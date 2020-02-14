import re
import os
import torch

from typing import Union, Sequence, Optional, Tuple
from open_couplet.tokenizer import Seq2seqTokenizer
from open_couplet.models.seq2seq import Seq2seqModel
from open_couplet.predictor import Seq2seqPredictor
from open_couplet import utils


class CoupletBot(object):
    VOCAB_FILE_NAME = 'vocab.txt'
    MAX_LEN = 30

    def __init__(self, model: Seq2seqModel, tokenizer: Seq2seqTokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self._reset_predictor()

    @classmethod
    def load(cls, model_dir: str, *, vocab_file: Optional[str] = None):
        vocab_file = os.path.join(model_dir, cls.VOCAB_FILE_NAME) \
            if vocab_file is None else vocab_file

        model = Seq2seqModel.from_trained(model_dir)
        tokenizer = Seq2seqTokenizer.from_vocab(vocab_file)

        return cls(model, tokenizer)

    def _reset_predictor(self):
        self._predictor = Seq2seqPredictor(self._model, self._tokenizer)
        self._predictor.eval()

    def reply(self, up_part: Union[str, Sequence[str]],
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

    def tokenize(self,
                 up_part: Sequence[str],
                 enforce_cleaned: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self._tokenizer

        inputs = []
        length = []

        fix_len = self.MAX_LEN + 2 if len(up_part) > 1 else len(up_part[0])
        max_len = -1

        for s in up_part:
            cleaned_str = s if enforce_cleaned else self.cleaning_inputs(s)
            cleaned_str = [tokenizer.bos_token] + list(cleaned_str) + [tokenizer.eos_token]
            l = len(cleaned_str)
            if l > self.MAX_LEN + 2:
                raise ValueError(f'"{s}" is too long. '
                                 f'Length of up-part should be shorter than {self.MAX_LEN}')
            inputs.append(tokenizer.convert_tokens_to_ids(
                cleaned_str + [tokenizer.pad_token] * (fix_len - l)))
            length.append(l)
            max_len = l if l > max_len else max_len

        x1 = torch.tensor(inputs)[:, :max_len]
        length = torch.tensor(length)

        return x1, length

    @staticmethod
    def cleaning_inputs(inputs):
        # delete all english characters, numbers and blank characters
        inputs = re.sub(r'[A-Za-z0-9\s+]', '', inputs)
        return utils.replace_en_punctuation(inputs)
