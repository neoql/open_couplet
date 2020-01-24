import os
import logging
import itertools
import json

import torch
import torch.nn as nn
import torch.jit as jit

from tqdm import tqdm
from collections import deque
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from open_couplet.models.seq2seq import Seq2seqModel
from open_couplet.tokenizer import Seq2seqTokenizer
from train.data import RandomBatchSampler, Seq2seqCollectWrapper


class Seq2seqTrainer(object):
    def __init__(self,
                 train_set: Dataset,
                 dev_set: Dataset,
                 tokenizer: Seq2seqTokenizer,
                 save_dir: str,
                 logging_dir: str,
                 logging_every: int,
                 save_eval_every: int,
                 max_ckpt_num: int,
                 logger: logging.Logger):

        self.train_set = train_set
        self.dev_set = dev_set

        self.logging_dir = logging_dir
        self.save_dir = save_dir

        self.logging_every = logging_every
        self.save_eval_every = save_eval_every

        self.tokenizer = tokenizer
        self.logger = logger
        self.ckpt_manager = CheckpointManager(save_dir, max_ckpt_num, model_class=Seq2seqModel)

    def train(self, model,
              learning_rate: float,
              num_epochs: int,
              batch_size: int,
              max_grad_norm: float,
              early_stop: bool = False,
              max_attempt_times: int = 5,
              use_cuda: bool = True,
              resume: bool = False):

        tb_writer = SummaryWriter(self.logging_dir)

        loss_fn = nn.NLLLoss(ignore_index=self.tokenizer.pad_token_id, reduction='mean')

        collect_wrapper = Seq2seqCollectWrapper(self.tokenizer, enforce_sorted=True)

        if resume:
            self.ckpt_manager.resume()
            latest_ckpt_dir, latest_ckpt = self.ckpt_manager.get_latest_checkpoint()
            self.logger.info(f'resume from checkpoint "{latest_ckpt_dir}"')
            model = latest_ckpt.model
            trainer_states = latest_ckpt.trainer_states

            global_step = trainer_states['global_step']
            tr_loss, logging_loss = trainer_states['tr_loss'], trainer_states['logging_loss']
            tr_acc, logging_acc = trainer_states['tr_acc'], trainer_states['logging_acc']

            optimizer = trainer_states['optimizer']
        else:
            global_step = 0
            tr_loss, logging_loss = 0.0, 0.0
            tr_acc, logging_acc = 0.0, 0.0

            optimizer = Adam(model.parameters(), lr=learning_rate)

        if use_cuda:
            model = model.cuda()
            loss_fn = loss_fn.cuda()

        steps_per_epoch = len(self.train_set)

        attempt_times = 0
        early_stop_flag = False

        for epoch in range(num_epochs):
            if resume and global_step//steps_per_epoch < epoch:
                continue

            sampler = RandomBatchSampler(self.train_set, batch_size=batch_size)
            train_dl = DataLoader(self.train_set, batch_sampler=sampler, collate_fn=collect_wrapper)
            batch_iter = tqdm(enumerate(train_dl), total=len(train_dl), desc=f'Epoch-{epoch+1}', unit=' step')

            if resume:
                batch_iter = itertools.dropwhile(lambda n, _: n == global_step % steps_per_epoch, batch_iter)

            for step, batch in batch_iter:
                if use_cuda:
                    batch = tuple(t.cuda() for t in batch)

                x1, x2, y, x1_len = batch

                log_prob, attn_weights = model(x1, x2, x1_len, enforce_sorted=True)
                loss = loss_fn(log_prob.flatten(0, 1), y.flatten(0, 1))

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                model.zero_grad()

                loss_val = loss.item()
                acc = self.compute_accuracy(log_prob.detach(), y.detach()).item()

                batch_iter.set_postfix({'Loss': loss_val, 'Accuracy': acc})

                tr_loss = tr_loss + loss_val
                tr_acc = tr_loss + acc

                if global_step % self.logging_every == 0:
                    tb_writer.add_scalar("train/Loss", (tr_loss - logging_loss)/self.logging_every, global_step)
                    tb_writer.add_scalar("train/Accuracy", (tr_acc - logging_acc)/self.logging_every, global_step)

                    logging_loss, logging_acc = tr_loss, tr_acc

                if global_step % self.save_eval_every == 0:
                    states = self.evaluate(model, loss_fn, self.dev_set)
                    batch_iter.write(f'Step-{global_step} evaluate result: f{states}')

                    tb_writer.add_scalar("eval/Loss", states['dev_loss'], global_step)
                    tb_writer.add_scalar('eval/Accuracy', states['dev_acc'], global_step)

                    states.update({
                        'loss': loss_val,
                        'accuracy': acc
                    })

                    min_dev_loss = self.ckpt_manager.min_dev_loss
                    self.ckpt_manager.add_checkpoint(model, optimizer, states, global_step)

                    if early_stop and states['dev_loss'] > min_dev_loss:
                        attempt_times += 1
                    else:
                        attempt_times = 0

                    if attempt_times > max_attempt_times:
                        break

                global_step += 1
            if early_stop_flag:
                self.logger.info('early stop!')
                break

        best_ckpt_dir, best_ckpt = self.ckpt_manager.get_best_checkpoint()
        best_states = {
            'Loss': best_ckpt.trainer_states['loss'],
            'Accuracy': best_ckpt.trainer_states['accuracy'],
            'Dev Loss': best_ckpt.trainer_states['dev_loss'],
            'Dev Accuracy': best_ckpt.trainer_states['dev_acc']
        }
        self.logger.info(f'Best checkpoint "{best_ckpt_dir}": {best_states}')

        latest_ckpt_dir, latest_ckpt = self.ckpt_manager.get_latest_checkpoint()
        latest_states = {
            'Loss': latest_ckpt.trainer_states['loss'],
            'Accuracy': latest_ckpt.trainer_states['accuracy'],
            'Dev Loss': latest_ckpt.trainer_states['dev_loss'],
            'Dev Accuracy': latest_ckpt_dir.trainer_states['dev_acc']
        }
        self.logger.info(f'Latest checkpoint "{latest_ckpt_dir}": {latest_states}')

    def compute_accuracy(self, log_prob: torch.Tensor, y: torch.Tensor):
        mask = y != self.tokenizer.pad_token_id
        golden_y = y.masked_select(mask)
        pred_y = log_prob.argmax(-1).masked_select(mask)
        return (pred_y == golden_y).sum() / pred_y.size(0)

    @torch.no_grad()
    def evaluate(self, model, loss_fn, dev_set, use_cuda=True):
        collect_wrapper = Seq2seqCollectWrapper(self.tokenizer, enforce_sorted=True)
        sampler = RandomBatchSampler(dev_set, batch_size=128)
        dev_dl = DataLoader(dev_set, collate_fn=collect_wrapper, batch_sampler=sampler)

        tr_loss, tr_acc = 0.0, 0.0

        for batch in dev_dl:
            if use_cuda:
                batch = tuple(t.cuda() for t in batch)

            x1, x2, y, x1_len = batch

            log_prob, _, attn_weights = model(x1, x2, x1_len, enforce_sorted=True)
            loss = loss_fn(log_prob, y)
            acc = self.compute_accuracy(log_prob, y)

            tr_loss += loss.item()
            tr_acc += acc.item()

        return {'dev_loss': tr_loss/sampler.num_batches, 'dev_acc': tr_acc/sampler.num_batches}


class Checkpoint(object):
    JIT_MODEL_NAME = 'model.pt'
    TRAINER_STATES_NAME = 'trainer_states.pkl'

    def __init__(self, model, trainer_states):
        self.model = model
        self.trainer_states = trainer_states

    def save(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if isinstance(self.model, jit.ScriptModule):
            self.model.save(os.path.join(dirname, self.JIT_MODEL_NAME))
        else:
            self.model.save_trained(dirname)

        torch.save(self.trainer_states, os.path.join(dirname, self.TRAINER_STATES_NAME))

    @classmethod
    def load(cls, dirname, model_class=None):
        if os.path.exists(os.path.join(dirname, cls.JIT_MODEL_NAME)):
            model = jit.load(os.path.join(dirname, cls.JIT_MODEL_NAME), map_location='cpu')
        else:
            assert model_class is not None
            model = model_class.from_trained(dirname)
        trainer_states = torch.load(os.path.join(dirname, cls.TRAINER_STATES_NAME), map_location='cpu')
        return cls(model, trainer_states)


class CheckpointManager(object):
    CKPT_PREFIX = 'checkpoint_'

    def __init__(self, root_dir, max_ckpt_num, model_class=None):
        self.root_dir = root_dir
        self.max_ckpt_num = max_ckpt_num
        self.model_class = model_class

        self.min_dev_loss = float('inf')
        self.min_dev_loss_ckpt = None

        self.checkpoints = deque(maxlen=max_ckpt_num)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

    def resume(self):
        prefix_len = len(self.CKPT_PREFIX)
        ckpt_dirs = [path for path in os.listdir(self.root_dir) if path[:prefix_len] == self.CKPT_PREFIX]
        ckpt_dirs.sort(key=lambda ckpt_dir: int(ckpt_dir[prefix_len:]))

        self.checkpoints.extend(ckpt_dirs)

        if not os.path.exists(os.path.join(self.root_dir, 'best_checkpoint.json')):
            return

        with open(os.path.join(self.root_dir, 'best_checkpoint.json'), 'r') as fp:
            for k, v in json.load(fp).items():
                setattr(self, k, v)

    def get_latest_checkpoint(self):
        ckpt_dir = self.checkpoints[-1]
        return ckpt_dir, Checkpoint.load(ckpt_dir, self.model_class)

    def get_best_checkpoint(self):
        ckpt_dir = self.min_dev_loss_ckpt
        return ckpt_dir, Checkpoint.load(ckpt_dir, self.model_class)

    def add_checkpoint(self, model, optimizer, states, global_step):
        trainer_states = {
            'optimizer': optimizer,
            'global_step': global_step,
        }

        trainer_states.update(states)
        ckpt_dir = os.path.join(self.root_dir, self.CKPT_PREFIX + str(global_step))

        ckpt = Checkpoint(model, trainer_states)
        ckpt.save(ckpt_dir)

        if len(self.checkpoints) == self.max_ckpt_num:
            rm_ckpt_dir = self.checkpoints.popleft()
            if rm_ckpt_dir != self.min_dev_loss_ckpt:
                os.rmdir(rm_ckpt_dir)
        self.checkpoints.append(ckpt_dir)

        if states['dev_loss'] <= self.min_dev_loss:
            self.min_dev_loss = states['dev_loss']
            if self.min_dev_loss_ckpt not in self.checkpoints:
                os.rmdir(self.min_dev_loss_ckpt)
            self.min_dev_loss_ckpt = ckpt_dir
            with open(os.path.join(self.root_dir, 'best_checkpoint.json'), 'w') as fp:
                json.dump({
                    'min_dev_loss', self.min_dev_loss,
                    'min_dev_loss_ckpt', self.min_dev_loss_ckpt,
                }, fp)

        return ckpt_dir, ckpt
