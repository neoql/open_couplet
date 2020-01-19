#!/usr/bin/env python

import os
import argparse

from tqdm import tqdm
from open_couplet.tokenizer import Tokenizer
from open_couplet import utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('source_files', nargs='+')
    parser.add_argument('-o', '--output-file', required=True, help='Specifying an output file.')
    parser.add_argument('--add-cn-punctuations', action='store_true', default=False,
                        help='Add extra Chinese punctuations.')
    parser.add_argument('--unused-tokens', default=0, type=int, help='Number of unused tokens.')

    return parser.parse_args()


def chinese_punctuations():
    tokens = [
        '，', '。', '（', '）', '：', '；', '？', '！',
        '《', '》', '【', '】', '、', '～',
    ]
    return tokens


def mkdir_ifn_exists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def clean_data(line: str):
    line = utils.replace_en_punctuation(line.strip())
    return line


def main():
    args = parse_args()

    tokenizer = Tokenizer(unused_tokens=[f'[unused{i}]' for i in range(args.unused_tokens)])
    if args.add_cn_punctuations:
        tokenizer.add_tokens(chinese_punctuations())

    for filename in args.source_files:
        with open(filename, 'r') as fp:
            for line in tqdm(fp, desc=filename, unit=' line'):
                tokenizer.add_tokens(clean_data(line).split(' '))

    mkdir_ifn_exists(os.path.dirname(args.output_file))
    tokenizer.save_vocab(args.output_file)

    print(f'Save {tokenizer.vocab_size} tokens into "{args.output_file}" successfully!')


if __name__ == '__main__':
    main()
