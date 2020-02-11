import argparse
import torch

from open_couplet.tokenizer import Seq2seqTokenizer
from open_couplet.models.seq2seq import Seq2seqModel
from open_couplet.predictor import Seq2seqPredictor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, help="Model directory")
    parser.add_argument('--vocab_file', required=True, help="Vocab file path")
    parser.add_argument('--beam_size', type=int, default=16, help="Beam size for beam searching")

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = Seq2seqTokenizer.from_vocab(args.vocab_file)
    model = Seq2seqModel.from_trained(args.model)

    predictor = Seq2seqPredictor(model, tokenizer)
    predictor.eval()

    while True:
        src = input("输入上联: ")
        src = [tokenizer.bos_token] + list(src) + [tokenizer.eos_token]
        src_t = torch.tensor(tokenizer.convert_tokens_to_ids(src)).unsqueeze(0)
        tgt_t = predictor(src_t, torch.tensor([len(src)]), beam_size=args.beam_size)[0].squeeze(0)[0]
        tgt = ''.join(tokenizer.convert_ids_to_tokens(tgt_t.tolist()))
        print(tgt)


if __name__ == '__main__':
    main()
