import torch

from open_couplet.tokenizer import Seq2seqTokenizer
from open_couplet.models.seq2seq import Seq2seqModel
from open_couplet.predictor import Seq2seqPredictor


def main():
    tokenizer = Seq2seqTokenizer.from_vocab('experiment/vocab.txt')
    model = Seq2seqModel.from_trained('experiment/checkpoints_new/checkpoint_19099')

    predictor = Seq2seqPredictor(model, tokenizer)
    predictor.eval()

    while True:
        src = input("输入上联: ")
        src = [tokenizer.bos_token] + list(src) + [tokenizer.eos_token]
        src_t = torch.tensor(tokenizer.convert_tokens_to_ids(src)).unsqueeze(0)
        tgt_t = predictor(src_t, torch.tensor([len(src)]), beam_size=256)[0].squeeze(0)[0]
        tgt = ''.join(tokenizer.convert_ids_to_tokens(tgt_t.tolist()))
        print(tgt)


if __name__ == '__main__':
    main()
