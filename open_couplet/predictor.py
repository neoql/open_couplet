import torch
import torch.nn as nn

from typing import Optional, Tuple, List
from open_couplet.models.seq2seq import Seq2seqModel
from open_couplet.tokenizer import Tokenizer
from open_couplet import utils


# noinspection PyMethodMayBeStatic
class Seq2seqPredictor(nn.Module):
    __constants__ = ['pad_token_id', 'bos_token_id', 'eos_token_id', 'vocab_size']

    def __init__(self, model: Seq2seqModel, tokenizer: Tokenizer):
        super(Seq2seqPredictor, self).__init__()

        self.vocab_size = tokenizer.vocab_size

        self.encode = model.encode
        self.decode = model.decode

        self.special_token_ids = tokenizer.special_token_ids

        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def forward(self, source: torch.Tensor, src_len: torch.Tensor, beam_size: int = 1):
        assert beam_size > 0
        k = beam_size

        batch_size, fix_len = source.size()
        src_without_bos = source[:, 1:]
        fix_len_less_one = fix_len - 1
        tgt_len = src_len - 1

        # pad_mask: (batch_size, fix_len)
        pad_mask: Optional[torch.Tensor] = source.eq(self.pad_token_id) \
            if src_len.min().item() != fix_len else None
        sub_pad_mask: Optional[torch.Tensor] = src_without_bos.eq(self.pad_token_id) \
            if src_len.min().item() != fix_len else None

        # Initialize the scores; for the first step,
        # scores: (batch_size * k, 1)
        scores = torch.full([batch_size * k, 1], fill_value=float('-inf')).to(source.device)
        scores.index_fill_(0, torch.tensor([i * k for i in range(0, batch_size)]), 0.0)

        # output: (batch_size, k, fix_len-1)
        output = torch.full([batch_size, k, fix_len_less_one], fill_value=self.pad_token_id).long().to(source.device)
        output[torch.arange(batch_size).long(), :, tgt_len - 1] = torch.tensor(self.eos_token_id)

        # Initialize input variable of decoder
        # input_var: (batch_size * k, 1)
        input_var = torch.full([batch_size * k, 1], self.bos_token_id).long().to(source.device)

        # ban_token_mask: (batch_size * k, vocab_size)
        ban_token_mask = self.gen_token_mask(
            batch_size, k, torch.tensor(self.special_token_ids))

        fh: Optional[torch.Tensor] = None
        cnn_mem: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # (batch_size, fix_len-1, fix_len-1)
        sent_pattern = utils.sentence_pattern(src_without_bos, pad_mask=sub_pad_mask)

        # known_token: (batch_size, fix_len-1)
        # known: (batch_size, k, fix_len-1)
        known_token = self._known_token(sent_pattern, tgt_len).to(source.device)
        known = torch.stack([known_token] * k, dim=1)
        # multi: (batch_size, fix_len-1)
        multi = sent_pattern.sum(dim=2) > 1

        # encode source
        # context: (batch_size, fix_len, hidden_size)
        # state: (layers, batch_size, hidden_size)
        context, state = self.encode(source, src_len)

        context = torch.stack([context] * k, dim=1).flatten(0, 1)
        state = torch.stack([state] * k, dim=2).flatten(1, 2)

        # attention_mask: (batch_size * k, 1, fix_len)
        attention_mask: Optional[torch.Tensor] = torch.stack([pad_mask] * k, dim=1).flatten(0, 1).unsqueeze(1) \
            if pad_mask is not None else None

        end_indices_list: Optional[List[torch.Tensor]] = attention_mask.chunk(fix_len, dim=-1)[1:] \
            if attention_mask is not None else None

        for i in range(fix_len_less_one):
            log_prob, (state, fh, cnn_mem), attn_weights = self.decode(
                input_var, context, state, fh, attention_mask, cnn_mem
            )

            # update scores
            # scores: (batch_size * k, vocab_size)
            last_scores = scores
            scores = scores + log_prob.squeeze(1)

            # ban tokens
            if known_token[:, i].sum() == 0:
                token_mask = ban_token_mask
            else:
                token_mask = self._token_mask(ban_token_mask, known[:, :, i], output[:, :, i]).to(source.device)
            scores.masked_fill_(token_mask, float('-inf'))

            # top-k
            # scores: (batch_size, k)
            # candidates: (batch_size, k)
            scores, candidates = scores.view(batch_size, -1).topk(k, dim=1)

            scores = scores.view(batch_size * k, 1)
            if end_indices_list is not None:
                scores = torch.where(end_indices_list[i].view(batch_size * k, -1), last_scores, scores)

            # compute rank indices
            # candidates are k * vocab_size + offset
            batch_indices = torch.arange(batch_size).view(batch_size, 1).long()
            k_indices = candidates / self.vocab_size
            if end_indices_list is not None:
                k_indices = torch.where(end_indices_list[i].view(batch_size, k),
                                        torch.arange(k).view(1, k).long(), k_indices)
            # combine_indices: (batch_size * k,)
            combine_indices = (batch_indices * k + k_indices).view(-1)

            # re-rank
            output = output[batch_indices, k_indices, :]
            state = state[:, combine_indices, :]
            fh = fh[combine_indices, :]
            # noinspection PyTypeChecker
            cnn_mem = (cnn_mem[0][combine_indices, :, :], cnn_mem[1][combine_indices, :, :])
            ban_token_mask = ban_token_mask[combine_indices, :]

            # decode symbol; update output; update input_var;
            # update ban_token_mask
            # symbol: (batch_size, k)
            symbol = candidates % self.vocab_size
            if end_indices_list is not None:
                symbol = torch.where(end_indices_list[i].view(batch_size, k), torch.tensor(self.pad_token_id), symbol)
            input_var = symbol.view(batch_size * k, 1)
            # multi: (batch_size, max_len)
            if multi[:, i].sum() > 0:
                output = self._update_output(output, sent_pattern[:, i], symbol)
            else:
                output[:, :, i] = symbol
            ban_token_mask[torch.arange(batch_size * k).long(), symbol.view(-1)] = torch.tensor(1, dtype=torch.bool)

        return output.view(batch_size, k, fix_len_less_one), scores.view(batch_size, k)

    def gen_token_mask(self, batch_size: int, k: int, tokens: torch.Tensor):
        """
        :param batch_size: int
        :param k: int
        :param tokens: list
        :return: (batch_size * k, vocab_size)
        """
        token_mask = torch.zeros(batch_size * k, self.vocab_size, dtype=torch.bool)
        token_mask[:, tokens] = torch.tensor(1, dtype=torch.bool)
        return token_mask

    def _known_token(self, pattern: torch.Tensor, length: torch.Tensor):
        """
        :param pattern: (batch_size, fix_len, fix_len)
        :param length: (batch_size,)
        :return: (batch_size, fix_len)
        """
        known_token = torch.tril(pattern, diagonal=-1).sum(dim=2) > 0
        known_token[torch.arange(length.size(0)).long(), length - 1] = torch.tensor(1, dtype=torch.bool)
        return known_token

    def _token_mask(self, ban_token_mask: torch.Tensor, known: torch.Tensor, tokens: torch.Tensor):
        """
        :param ban_token_mask: (batch_size * k, vocab_size)
        :param known: (batch_size, k)
        :param tokens: (batch_size, k)
        :return: (batch_size * k, vocab_size)
        """
        batch_size, k = tokens.size()
        token_mask = torch.ones(batch_size * k, self.vocab_size).long()
        token_mask[torch.arange(batch_size * k).long(), tokens.view(-1)] = torch.tensor(0).long()
        token_mask = torch.where(known.view(-1, 1), token_mask, ban_token_mask.long()).to(torch.bool)
        return token_mask

    def _update_output(self, output: torch.Tensor, indices: torch.Tensor, symbol: torch.Tensor):
        """
        :param output: (batch_size, k, max_len)
        :param indices: (batch_size, max_len) boolean
        :param symbol: (batch_size, k)
        :return: output
        """
        max_len = output.size(2)
        indices = indices.unsqueeze(1)

        return torch.where(indices, torch.stack([symbol] * max_len, dim=2), output)
