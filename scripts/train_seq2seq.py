#!/usr/bin/env python

import argparse

import torch

from open_couplet.config import Seq2seqConfig
from open_couplet.tokenizer import Seq2seqTokenizer
from open_couplet.models.seq2seq import Seq2seqModel

from train.trainer import Seq2seqTrainer
from train.data import ParallelDataset


import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_file', required=True, help="Path of vocabulary file")

    # Model hyperparameters
    parser.add_argument('--hidden_size', required=True, type=int, help="Hidden layer vector size")
    parser.add_argument('--rnn_layers', required=True, type=int, help="Number of stacked rnn hidden layers")
    parser.add_argument('--cnn_kernel_size', required=True, type=int, help="CNN kernel size")
    parser.add_argument('--dropout_p', required=True, type=float, help="Dropout probability")

    # Path of training set and development set
    parser.add_argument('--train_set_dir', required=True, help="Path of training path")
    parser.add_argument('--dev_set_dir', required=True, help="Path of development set")

    parser.add_argument('--save_dir', required=True, help="Path for saving checkpoints")
    parser.add_argument('--logging_dir', required=True, help="Path for logging")

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_grad_norm', type=float, help='Max grad norm for grad clipping')
    parser.add_argument('--early_stop', action='store_true', default=False, help='Enable early stop')
    parser.add_argument('--max_attempt_times', type=int, default=10, help='Max attempt for early stop')
    parser.add_argument('--resume', action='store_true', default=False, help="Resume training use latest checkpoint")
    parser.add_argument('--disable_cuda', action='store_true', default=False, help="Disable cuda")

    parser.add_argument('--logging_every', type=int, default=50, help="Steps for each logging interval")
    parser.add_argument('--save_eval_every', type=int, default=50,
                        help="Steps for each saving and evaluating model internal")
    parser.add_argument('--max_ckpt_num', type=int, default=30, help="Maximum checkpoints saved")

    return parser.parse_args()


def main():
    args = parse_args()

    use_cuda = torch.cuda.is_available() and not args.disable_cuda
    logger.info(f'Use CUDA: {use_cuda}')

    tokenizer = Seq2seqTokenizer.from_vocab(args.vocab_file)

    config = Seq2seqConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        rnn_layers=args.rnn_layers,
        cnn_kernel_size=args.cnn_kernel_size,
        dropout_p=args.dropout_p,
    )

    # model = jit.script(Seq2seqModel(config))
    model = Seq2seqModel(config)

    logger.info('Start loading training set ...')
    train_set = ParallelDataset(args.train_set_dir, pbar=True, sort=True)
    logger.info('The training set is loaded completely!')

    logger.info('Start loading dev set ...')
    dev_set = ParallelDataset(args.dev_set_dir, pbar=True, sort=True)
    logger.info('The dev set is loaded completely!')

    trainer = Seq2seqTrainer(
        train_set=train_set,
        dev_set=dev_set,
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        logging_dir=args.logging_dir,
        logging_every=args.logging_every,
        save_eval_every=args.save_eval_every,
        max_ckpt_num=args.max_ckpt_num,
        logger=logger,
    )

    trainer.train(
        model=model,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        early_stop=args.early_stop,
        max_attempt_times=args.max_attempt_times,
        use_cuda=use_cuda,
        resume=args.resume,
    )


if __name__ == '__main__':
    main()
