#!/usr/bin/env python

import argparse
from open_couplet import CoupletBot


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, help="Model directory")
    parser.add_argument('--beam_size', type=int, default=16, help="Beam size for beam searching")
    parser.add_argument('--topk', type=int, default=1, help="Number of the down-parts which returned")

    return parser.parse_args()


def main():
    args = parse_args()
    bot = CoupletBot.load(args.model)

    while True:
        up_part = input("输入上联：")
        for down_part, score in bot.reply(up_part, beam_size=args.beam_size, topk=args.topk):
            print(down_part)


if __name__ == '__main__':
    main()
