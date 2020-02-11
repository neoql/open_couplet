import os
import torch

from typing import Union, Iterable, Optional
from open_couplet.tokenizer import Seq2seqTokenizer
from open_couplet.models.seq2seq import Seq2seqModel
from open_couplet.predictor import Seq2seqPredictor
from open_couplet import utils


class CoupletBot(object):
    VOCAB_FILE_NAME = 'vocab.txt'
    MAX_LEN = 30

    def __init__(self, model_dir: str, *, vocab_file: Optional[str] = None):
        self.model_dir = model_dir
        self.vocab_file = os.path.join(self.model_dir, self.VOCAB_FILE_NAME) \
            if vocab_file is None else vocab_file

        self._reset_tokenizer()
        self._reset_model()
        self._reset_predictor()

    def _reset_tokenizer(self):
        self._tokenizer = Seq2seqTokenizer.from_vocab(self.vocab_file)

    def _reset_model(self):
        self._model = Seq2seqModel.from_trained(self.model_dir)

    def _reset_predictor(self):
        self._predictor = Seq2seqPredictor(self._model, self._tokenizer)
        self._predictor.eval()

    def reply(self, up_part: Union[str, Iterable[str]],
              beam_size: int = 16,
              topk: int = 1,
              enforce_cleaned: bool = False):
        single_flag = isinstance(up_part, str)
        up_part = [up_part] if single_flag else up_part
        assert 0 < topk <= beam_size

        x1, length = self.tokenize(up_part, enforce_cleaned)

        predict = self._predictor
        tokenizer = self._tokenizer

        with torch.no_grad():
            y, scores = predict(x1, length, beam_size=beam_size, enforce_sorted=False)

        batch_size = y.size(0)
        down_part = []

        for i in range(batch_size):
            yi = y[i, :, :length[i]-1]
            si = scores[i]
            down_part.append([
                (''.join(tokenizer.convert_ids_to_tokens(yi[j].tolist())[:-1]), si[j].item())
                for j in range(topk)
            ])

        down_part = down_part[0] if single_flag else down_part

        return down_part

    def tokenize(self, up_part: Iterable[str], enforce_cleaned: bool = False):
        tokenizer = self._tokenizer

        inputs = []
        length = []

        for s in up_part:
            s = s if enforce_cleaned else self.cleaning_inputs(s)
            s = [tokenizer.bos_token] + list(s) + [tokenizer.eos_token]
            l = len(s)
            inputs.append(tokenizer.convert_tokens_to_ids(
                s + [tokenizer.pad_token] * (self.MAX_LEN - l)))
            length.append(l)

        x1 = torch.tensor(inputs)[:, :max(length)]
        length = torch.tensor(length)

        return x1, length

    @staticmethod
    def cleaning_inputs(inputs):
        return utils.replace_en_punctuation(inputs)
