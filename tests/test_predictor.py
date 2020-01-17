import torch
import torch.jit as jit

from math import inf

from open_couplet.predictor import Seq2seqPredictor
from open_couplet.models.seq2seq import Seq2seqModel
from open_couplet.config import Seq2seqConfig
from open_couplet.tokenizer import Tokenizer
from open_couplet import utils


class TestSeq2seqPredictor(object):

    @classmethod
    def setup_class(cls):
        config = Seq2seqConfig(vocab_size=50, hidden_size=100)

        cls.tokenizer = Tokenizer()
        cls.model = Seq2seqModel(config)
        cls.predictor = Seq2seqPredictor(cls.model, cls.tokenizer)
        cls.predictor.vocab_size = 50

    def test_known_token(self):
        sentence = torch.tensor([
            [1, 2, 3, 4, 5, 0],
            [1, 2, 3, 3, 2, 0],
            [1, 2, 3, 2, 0, 0],
            [1, 1, 2, 2, 0, 0],
            [1, 2, 3, 0, 0, 0],
        ])

        length = torch.tensor([6, 6, 5, 5, 4])

        known_token = self.predictor._known_token(utils.sentence_pattern(sentence), length)
        golden_known_token = torch.tensor([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=torch.bool)

        for i, l in enumerate(length):
            assert torch.equal(known_token[i, :l], golden_known_token[i, :l])

    def test_token_mask(self):
        batch_size, k = 3, 2
        sos, pad, eos = self.tokenizer.bos_token_id, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id

        ban_token_mask = self.predictor.gen_token_mask(batch_size, k, torch.tensor([sos, pad, eos]))
        ban_token_mask[:, [5, 10, 15]] = 1

        known = torch.stack([torch.tensor([False, True, True])] * k, dim=1)
        tokens = torch.stack([
            torch.tensor([pad, 10, eos]),
            torch.tensor([pad, 15, eos]),
        ], dim=1)

        token_mask = self.predictor._token_mask(ban_token_mask, known, tokens)

        ban_token_mask = ban_token_mask.view(batch_size, k, -1)
        token_mask = token_mask.view(batch_size, k, -1)

        batch1_mask = ban_token_mask[0, :, :]
        batch2_mask = torch.ones(k, self.predictor.vocab_size).bool()
        batch2_mask[[0, 1], [10, 15]] = 0
        batch3_mask = torch.ones(k, self.predictor.vocab_size).bool()
        batch3_mask[:, eos] = 0

        # noinspection PyUnresolvedReferences
        assert torch.equal(token_mask[0, :, :], batch1_mask)
        assert torch.equal(token_mask[1, :, :], batch2_mask)
        assert torch.equal(token_mask[2, :, :], batch3_mask)

    def test_update_output(self):
        output = torch.arange(3*4*5).view(3, 4, 5)
        indices = torch.tensor([
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
        ], dtype=torch.bool)
        symbol = torch.tensor([
            [101, 102, 103, 104],
            [201, 202, 203, 204],
            [301, 302, 303, 304],
        ])

        batch_size = output.size(0)

        golden_output = output.clone()
        for n in range(batch_size):
            golden_output[n, :, indices[n]] = symbol[n].unsqueeze(1)

        output = self.predictor._update_output(output, indices, symbol)
        assert output.equal(golden_output)

    def test_predict_prob(self):
        self.model.eval()
        test_predictor = GoldenSeq2seqPredictor(self.model, self.predictor.vocab_size, self.tokenizer)
        jit_predictor = jit.script(self.predictor)

        sentence, length = sample_case1()
        eps = 1e-5

        golden_output, golden_scores = test_predictor.predict_prob(sentence, length, beam_size=1)

        with torch.no_grad():
            output, scores = self.predictor(sentence, length, beam_size=1)

        assert torch.norm(scores - golden_scores) < eps
        assert torch.equal(output, golden_output)

        with torch.no_grad():
            output, scores = jit_predictor(sentence, length, beam_size=1)

        assert torch.norm(scores - golden_scores) < eps
        assert torch.equal(output, golden_output)

        golden_output, golden_scores = test_predictor.predict_prob(sentence, length, beam_size=3)

        with torch.no_grad():
            output, scores = self.predictor(sentence, length, beam_size=3)

        assert torch.norm(scores - golden_scores, p=inf) < eps
        assert torch.equal(output, golden_output)

        with torch.no_grad():
            output, scores = jit_predictor(sentence, length, beam_size=3)

        assert torch.norm(scores - golden_scores, p=inf) < eps
        assert torch.equal(output, golden_output)


def sample_case1():
    sentence = torch.tensor([
        [1, 13, 14, 15, 16, 17, 2],
        [1, 13, 14, 15, 16, 13, 2],
        [1, 13, 14, 15, 16, 12, 0],
        [1, 13, 14, 13, 14,  2, 0],
        [1, 19, 18, 17,  2,  0, 0],
    ])

    length = torch.tensor([7, 7, 6, 6, 5])

    return sentence, length


class GoldenSeq2seqPredictor(object):
    def __init__(self, model: Seq2seqModel, vocab_size, tokenizer: Tokenizer):
        self.model = model
        self.encoder = model.encode
        self.decoder = model.decode

        self.vocab_size = vocab_size
        self.special_token_ids = tokenizer.special_token_ids
        self.sos = tokenizer.bos_token_id
        self.pad = tokenizer.pad_token_id
        self.eos = tokenizer.eos_token_id

    def predict_prob(self, source, length, beam_size=1):
        self.model.eval()
        with torch.no_grad():
            return self._beam_search(source, length, beam_size)

    def _beam_search(self, source, length, beam_size=1):
        # pad_mask: (batch_size, total_len)
        # sent_pattern: (batch_size, total_len, total_len)
        pad_mask = source.eq(self.pad)
        sub_pad_mask = source[:, 1:].eq(self.pad)
        sent_pattern = utils.sentence_pattern(source[:, 1:], pad_mask=sub_pad_mask)

        # encode source
        # context: (batch_size, max_len, 2 * enc_rnn_units)
        # state: (layers, batch_size, 2 * enc_rnn_units)
        context, state = self.encoder(source, length)

        batch_size, fix_len = source.size()

        output = []
        scores = []

        for i in range(batch_size):
            context_i = context[i]
            state_i = state[:, i, :]
            length_i = length[i].item()
            sent_pattern_i = sent_pattern[i]
            pad_mask_i = pad_mask[i]

            output_i, scores_i = self._beam_search_single(
                context=context_i,
                state=state_i,
                src_len=length_i,
                fix_len=fix_len,
                beam_size=beam_size,
                sent_pattern=sent_pattern_i,
                attention_mask=pad_mask_i,
            )

            output += [output_i]
            scores += [scores_i]

        output = torch.stack(output, dim=0)
        scores = torch.stack(scores, dim=0)

        return output, scores

    def _beam_search_single(self, context, state, src_len, fix_len, beam_size, sent_pattern, attention_mask):
        """
        :param context: (total_length, 2 * enc_rnn_units)
        :param state: (layers, 2 * enc_rnn_units)
        :param src_len: scalar
        :param fix_len: scalar
        :param beam_size: scalar
        :param sent_pattern: (total_length, total_length)
        :param attention_mask: (total_length,)
        :return:
            output: (beam_size, total_length)
            scores: (beam_size,)
        """
        fix_len_less_one = fix_len - 1
        tgt_len = src_len - 1

        out = torch.full([beam_size, fix_len_less_one], fill_value=self.pad).long()
        out[:, tgt_len - 1] = self.eos

        scores = torch.zeros(beam_size, 1).float()
        scores[1:, :] = -inf

        context = torch.stack([context] * beam_size, dim=0)
        state = torch.stack([state] * beam_size, dim=1)

        # known_token: (total_len,)
        known_token = self._known_token(sent_pattern.unsqueeze(0), torch.tensor([tgt_len])).view(-1)

        # ban_token_mask: (beam_size, vocab_size)
        ban_token_mask = self.gen_token_mask(1, beam_size, self.special_token_ids)

        # multi: (fix_len-1,)
        multi = sent_pattern.sum(dim=1) > 1

        # pad_mask: (beam_size, total_length)
        attention_mask = attention_mask.expand(beam_size, -1)
        fh = None
        cnn_mem = None

        input_var = torch.full([beam_size, 1], self.sos).long()
        for i in range(tgt_len):
            log_prob, (state, fh, cnn_mem), attn_weights = self.decoder(
                input_var, context, state, fh, attention_mask.unsqueeze(1), cnn_mem
            )

            # scores: (beam_size, vocab_size)
            scores = scores.view(-1, 1)
            scores = scores + log_prob.squeeze(1)

            # token_mask: (beam_size, vocab_size)
            if known_token[i]:
                token_mask = self.known_token_mask(out[:, i])
            else:
                token_mask = ban_token_mask

            scores.masked_fill_(token_mask, -inf)

            # scores: (beam_size,)
            # candidates: (beam_size,)
            scores, candidates = scores.view(-1).topk(beam_size)

            # scores: (beam_size, 1)
            scores.unsqueeze(1)

            k_indices = candidates // self.vocab_size

            # re-rank
            # out: (beam_size, fix_len)
            out = out[k_indices]
            state = state[:, k_indices, :]
            fh = fh[k_indices]
            cnn_mem = tuple(m[k_indices, :, :] for m in cnn_mem)
            ban_token_mask = ban_token_mask[k_indices]

            # symbol: (beam_size,)
            symbol = candidates % self.vocab_size
            input_var = symbol.unsqueeze(dim=1)

            if multi[i]:
                out[:, sent_pattern[i]] = symbol.unsqueeze(1)
            else:
                out[:, i] = symbol
            ban_token_mask[range(beam_size), symbol] = 1

        return out, scores

    def known_token_mask(self, tokens):
        """
        :param tokens: (beam_size,)
        :return: (batch_size * k, vocab_size)
        """
        beam_size = tokens.size(0)
        token_mask = torch.ones(beam_size, self.vocab_size).bool()
        token_mask[range(beam_size), tokens.view(-1)] = 0
        return token_mask

    def gen_token_mask(self, batch_size, k, tokens):
        """
        :param batch_size: int
        :param k: int
        :param tokens: list
        :return: (batch_size * k, vocab_size)
        """
        token_mask = torch.zeros(batch_size * k, self.vocab_size).bool()
        token_mask[:, tokens] = 1
        return token_mask

    @staticmethod
    def _known_token(pattern, length):
        """
        :param pattern: (batch_size, max_len, max_len)
        :param length: (batch_size,)
        :return: (batch_size, max_len)
        """
        known_token = torch.tril(pattern, diagonal=-1).sum(dim=2) > 0
        known_token[range(length.size(0)), length - 1] = 1
        return known_token
