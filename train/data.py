import os
import torch

from typing import List, Tuple
from contextlib import ExitStack, contextmanager
from itertools import zip_longest

from tqdm import tqdm
from torch.utils.data import Dataset, Sampler

from open_couplet.tokenizer import Tokenizer
from open_couplet import utils


class ParallelDataset(Dataset):
    def __init__(self, dirname: str,
                 pbar: bool = False,
                 in_file_name: str = 'in.txt',
                 out_file_name: str = 'out.txt',
                 sep: str = ' ',
                 sort: bool = False):

        in_file = os.path.join(dirname, in_file_name)
        out_file = os.path.join(dirname, out_file_name)

        self.sep = sep
        self.parallel = self._load_parallel(in_file, out_file, pbar=pbar, sort=sort)

    @contextmanager
    def _open_parallel(self, in_file: str, out_file: str):
        fps = [open(in_file, 'r'), open(out_file, 'r')]

        with ExitStack() as stack:
            yield tuple(stack.enter_context(fp) for fp in fps)

    def _load_parallel(self, in_file: str, out_file: str, pbar: bool = False, sort: bool = False):
        parallel_data = []

        with self._open_parallel(in_file, out_file) as (in_fp, out_fp):
            para_iter = tqdm(zip_longest(in_fp, out_fp)) if pbar else zip_longest(in_fp, out_fp)
            for in_line, out_line in para_iter:
                assert in_line and out_line is not None
                src_item = self.clean_data(in_line)
                tgt_item = self.clean_data(out_line)
                assert len(src_item) == len(tgt_item)
                parallel_data.append((src_item, tgt_item))

        print('before')
        if sort:
            parallel_data.sort(key=lambda x: len(x[0]), reverse=True)
        print('after')
        return parallel_data

    # noinspection PyMethodMayBeStatic
    def clean_data(self, line):
        line = utils.replace_en_punctuation(line.strip())
        return line

    def __getitem__(self, index):
        return tuple(line.split(self.sep) for line in self.parallel[index])

    def __len__(self):
        return len(self.parallel)


class RandomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False):
        super(RandomBatchSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        last_batch_size = len(data_source) % batch_size
        self.last_batch_size = last_batch_size if last_batch_size > 0 and not drop_last else batch_size

    @property
    def num_batches(self):
        num = len(self.data_source) // self.batch_size
        if self.last_batch_size < self.batch_size:
            num += 1
        return num

    def __iter__(self):
        n = self.num_batches
        for index in torch.randperm(n).tolist():
            start = index * self.batch_size
            size = self.last_batch_size if index == n-1 else self.batch_size
            yield range(start, start+size)

    def __len__(self):
        return self.num_batches


class Seq2seqCollectWrapper(object):
    def __init__(self, tokenizer: Tokenizer, enforce_sorted: bool = True):
        self.tokenizer = tokenizer

        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.enforce_sorted = enforce_sorted

    def __call__(self, batch: List[Tuple[List[str], List[str]]]):
        if not self.enforce_sorted:
            batch.sort(key=lambda x: len(x[0]), reverse=True)

        fix_len = len(batch[0][0])+2

        src_list = []
        tgt_list = []
        length = []

        for src_item, tgt_item in batch:
            src_list.append(torch.tensor(self.tokenizer.convert_tokens_to_ids(
                self.padding([self.bos_token] + src_item + [self.eos_token], fix_len))))
            tgt_list.append(torch.tensor(self.tokenizer.convert_tokens_to_ids(
                self.padding([self.bos_token] + tgt_item + [self.eos_token], fix_len))))
            length.append(len(src_item)+2)

        src_tensor = torch.stack(src_list, dim=0)
        tgt_tensor = torch.stack(tgt_list, dim=0)
        length_tensor = torch.tensor(length)

        return src_tensor, tgt_tensor[:, :-1], tgt_tensor[:, 1:], length_tensor

    def padding(self, tokens, fix_len):
        return tokens + [self.pad_token] * (fix_len - len(tokens))
