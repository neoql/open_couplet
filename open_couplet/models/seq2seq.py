import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from open_couplet.config import Seq2seqConfig


def gelu(x):
    """ Implementation of the gelu activation function.
        Using OpenAI GPT's gelu (not exactly the same as BERT)
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return x * cdf


def position_encoding(pos_seq: torch.Tensor,
                      inv_freq: torch.Tensor,
                      batch_size: int = 1):
    # noinspection PyTypeChecker
    sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
    pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    pos_emb = pos_emb.unsqueeze(0)

    if batch_size > 1:
        pos_emb = pos_emb.expand(batch_size, -1, -1)

    return pos_emb


class CNN(nn.Module):
    """
    Convolution Neural Network.
    """

    __constants__ = ['dropout_p']

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 kernel_size: int = 3, mid: bool = True, dropout_p: float = 0.0):
        super(CNN, self).__init__()

        padding: Tuple[int, int] = (kernel_size // 2, (kernel_size - 1) // 2) if mid else (kernel_size-1, 0)
        self.dropout_p = dropout_p
        self.mid = mid

        self.pad = nn.ConstantPad1d(padding, 0.0)
        self.conv_1 = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, padding=0)
        self.conv_2 = nn.Conv1d(hidden_size, output_size, kernel_size=kernel_size, padding=0)

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()

    def forward(self, x: torch.Tensor,
                mem: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                mask: Optional[torch.Tensor] = None,
                ret_mem: int = 0):
        x = x.transpose(-2, -1)

        if mask is not None:
            x = x.masked_fill(mask, 0.0)

        if mem is not None:
            x = torch.cat([mem[0], x], dim=-1)
        else:
            x = self.pad(x)

        h = gelu(self.conv_1(x))

        if self.dropout_p > 0:
            h = F.dropout(h, p=self.dropout_p, training=self.training)

        if mask is not None:
            h = h.masked_fill(mask, 0.0)

        if mem is not None:
            h = torch.cat([mem[1], h], dim=-1)
        else:
            h = self.pad(h)

        y = gelu(self.conv_2(h))

        return y.transpose(-2, -1), (x[:, :, :ret_mem], h[:, :, -ret_mem:])


class Attention(nn.Module):
    __constants__ = ['hidden_size', 'clamp_len']

    def __init__(self, hidden_size: int, clamp_len: int = -1):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.clamp_len = clamp_len

        self.content_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.position_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self,
                query: torch.Tensor,
                key_val: torch.Tensor,
                mask: Optional[torch.Tensor]):
        """
        :param query: (batch_size, q_len, hidden_size)
        :param key_val: (batch_size, kv_len, hidden_size)
        :param mask: attention mask
        :return:
            output: (batch_size, q_len, hidden_size)
            weights: (batch_size, q_len, kv_len)
        """
        # qh, qr: (batch_size, q_len, hidden_size)
        qh = self.content_proj(query)
        qr = self.position_proj(query)

        # kh, v: (batch_size, kv_len, hidden_size)
        # kr: (batch_size, r_len, hidden_size)
        kh, v = key_val, key_val
        kr = self.relative_position_encoding(
            query.size(0), query.size(1), key_val.size(1)).to(query.device)

        # content_scores, position_scores: (batch_size, q_len, kv_len)
        content_scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        position_scores = torch.matmul(qr, kr.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        position_scores = self.rel_shift(position_scores, klen=key_val.size(1))

        scores = content_scores + position_scores

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, -1)
        return torch.matmul(weights, v), weights

    def relative_position_encoding(self, batch_size, q_len, kv_len):
        freq_seq = torch.arange(0, self.hidden_size, 2.0, dtype=torch.float)
        # noinspection PyTypeChecker
        inv_freq: torch.Tensor = 1.0 / torch.pow(10000, (freq_seq / self.hidden_size))

        begin, end = kv_len, -q_len

        pos_seq = torch.arange(begin, end, -1.0)

        if self.clamp_len > 0:
            pos_seq = pos_seq.clamp(-self.clamp_len, self.clamp_len)
        pos_encoding = position_encoding(pos_seq, inv_freq, batch_size)

        return pos_encoding

    # noinspection PyMethodMayBeStatic
    def rel_shift(self, x: torch.Tensor, klen: int):
        """
        :param x: (batch_size, q_len, r_len)
        :param klen: scalar
        :return: (batch_size, q_len, klen)
        """
        x_size = x.size()
        x = x.reshape(x_size[0], x_size[2], x_size[1])
        x = x[:, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2] - 1)

        return x.index_select(2, torch.arange(klen, device=x.device, dtype=torch.long))


# noinspection PyMethodMayBeStatic
class Encoder(nn.Module):
    __constants__ = ['hidden_size', 'rnn_layers', 'cnn_kernel_size', 'dropout_p']

    def __init__(self, embedding, hidden_size, rnn_layers=1, cnn_kernel_size=3, dropout_p=0.0):
        super(Encoder, self).__init__()

        self.embedding: nn.Embedding = embedding
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.dropout_p = dropout_p

        self.cnn = CNN(hidden_size, hidden_size, hidden_size, kernel_size=cnn_kernel_size, dropout_p=dropout_p)
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True
        )

        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)

        self.h_proj = nn.Linear(2 * hidden_size, hidden_size)

        self.reset_parameters()

    # noinspection DuplicatedCode
    def reset_parameters(self):
        self.cnn.reset_parameters()
        # self.rnn.reset_parameters()
        self.norm_1.reset_parameters()
        self.norm_2.reset_parameters()
        self.h_proj.reset_parameters()

        for w_ih, w_hh, b_ih, b_hh in self.rnn.all_weights:
            I.orthogonal_(w_ih)
            I.orthogonal_(w_hh)
            I.zeros_(b_ih)
            I.zeros_(b_hh)

    def forward(self, source: torch.Tensor,
                seq_len: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                enforce_sorted: bool = False):
        """
        :param source: (batch_size, src_len) source sequences
        :param seq_len: (batch_size,) sequence length of source sequences
        :param mask: pad mask
        :param enforce_sorted: if True, the source is expected to contain sequences sorted by
            length in a decreasing order. If False, this condition is not checked. Default: True.
        :return: context, hidden
            **context** (batch_size, src_len, hidden_size):
                tensor containing the encoded features of the source sequence
            **state** (layers, batch, hidden_size):
                tensor containing the features in the hidden state `h`
        """
        batch_size, src_len = source.size()

        # Embedding layer
        embedded = self.embedding(source)
        if self.dropout_p:
            embedded = F.dropout(embedded, p=self.dropout_p, training=self.training)

        # CNN layer
        cnn_out, _ = self.cnn(embedded, mask=mask)
        if self.dropout_p:
            cnn_out = F.dropout(cnn_out, p=self.dropout_p, training=self.training)
        cnn_out = self.norm_1(embedded + cnn_out)

        # RNN layer
        rnn_input = pack_padded_sequence(cnn_out, seq_len, batch_first=True, enforce_sorted=enforce_sorted)
        # noinspection PyTypeChecker
        context, state = self.rnn(rnn_input)
        # (batch_size, src_len, 2, hidden_size)
        context = pad_packed_sequence(context, batch_first=True)[0].view(batch_size, src_len, 2, -1)
        if self.dropout_p:
            context = F.dropout(context, p=self.dropout_p, training=self.training)
        context = self.norm_2(context[:, :, 0] + context[:, :, 1])

        # cat directions
        state = torch.tanh(self.h_proj(self._cat_directions(state)))

        return context, state

    def _cat_directions(self, h):
        """(layers * directions, batch, hidden_size) -> (layers, batch, directions * hidden_size)"""
        return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)


# noinspection DuplicatedCode
class Decoder(nn.Module):
    __constants__ = ['hidden_size', 'rnn_layers', 'cnn_kernel_size', 'dropout_p']

    def __init__(self, embedding: nn.Embedding, hidden_size: int, cnn_kernel_size: int = 3,
                 rnn_layers: int = 1, dropout_p: float = 0):
        super(Decoder, self).__init__()

        self.embedding = embedding
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.dropout_p = dropout_p

        self.cnn = CNN(hidden_size, hidden_size, hidden_size, kernel_size=cnn_kernel_size,
                       dropout_p=dropout_p, mid=False)
        self.rnn = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout_p,
            batch_first=True
        )

        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.norm_3 = nn.LayerNorm(hidden_size)

        self.attention = Attention(hidden_size=hidden_size)

        self.fc_1 = nn.Linear(hidden_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, embedding.num_embeddings)

        self.reset_parameters()

    # noinspection DuplicatedCode
    def reset_parameters(self):
        self.cnn.reset_parameters()
        # self.rnn.reset_parameters()
        self.norm_1.reset_parameters()
        self.norm_2.reset_parameters()
        self.norm_3.reset_parameters()
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()

        for w_ih, w_hh, b_ih, b_hh in self.rnn.all_weights:
            I.orthogonal_(w_ih)
            I.orthogonal_(w_hh)
            I.zeros_(b_ih)
            I.zeros_(b_hh)

    def forward(self,
                input_seq: torch.Tensor,
                context: torch.Tensor,
                state: torch.Tensor,
                fh: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cnn_mem: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        :param input_seq: (batch, tgt_len)
        :param context: (layers, batch, hidden_size)
        :param state: (batch_size, src_len, hidden_size)
        :param fh: (batch_size, hidden_size)
        :param attention_mask: (batch_size, src_fix_len)
        :param cnn_mem: (batch_size, hidden_size, mem_len)
        :return:
        """
        batch_size, tgt_len = input_seq.size()
        if fh is None:
            fh = torch.zeros(batch_size, self.hidden_size).to(input_seq.device)

        # Embedding layer
        # embedded: (batch_size, tgt_len, hidden_size)
        embedded = self.embedding(input_seq)
        if self.dropout_p:
            embedded = F.dropout(embedded, p=self.dropout_p, training=self.training)

        # CNN layer
        cnn_out, cnn_mem = self.cnn(embedded, mem=cnn_mem, ret_mem=self.cnn_kernel_size - 1)
        if self.dropout_p:
            cnn_out = F.dropout(cnn_out, p=self.dropout_p, training=self.training)
        cnn_out = self.norm_1(cnn_out + embedded)

        outputs = []
        attn_weights = []

        for i in range(tgt_len):
            # (batch_size, tgt_len, 2*hidden_size)
            input_step = torch.cat([cnn_out[:, i, :], fh], dim=-1).unsqueeze(1)

            rnn_out, state = self.rnn(input_step, state)
            rnn_out = self.norm_2(rnn_out)
            if self.dropout_p:
                rnn_out = F.dropout(rnn_out, p=self.dropout_p, training=self.training)

            attn_out, attn_w = self.attention(query=rnn_out, key_val=context, mask=attention_mask)
            if self.dropout_p:
                attn_out = F.dropout(attn_out, p=self.dropout_p, training=self.training)
            attn_out = self.norm_3(attn_out + rnn_out)

            out = gelu(self.fc_1(attn_out))
            if self.dropout_p:
                out = F.dropout(out, p=self.dropout_p, training=self.training)
            fh = out.view(batch_size, self.hidden_size)

            outputs.append(out)
            attn_weights.append(attn_w)

        log_prob = F.log_softmax(self.fc_2(torch.cat(outputs, dim=1)), dim=-1)

        return log_prob, (state, fh, cnn_mem), torch.cat(attn_weights, dim=1)


# noinspection PyMethodMayBeStatic
class Seq2seqModel(nn.Module):
    def __init__(self, config: Seq2seqConfig):
        super(Seq2seqModel, self).__init__()
        self.config = config

        embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.encode = Encoder(embedding, config.hidden_size, config.rnn_layers, config.cnn_kernel_size,
                               config.dropout_p)
        self.decode = Decoder(embedding, config.hidden_size, config.cnn_kernel_size,
                              config.rnn_layers, config.dropout_p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x1_len: torch.Tensor, enforce_sorted: bool = False):
        fix_len = x1.size(1)
        mask = self.attention_mask(x1_len, fix_len) \
            if x1_len.min().item() != fix_len else None

        context, state = self.encode(x1, x1_len, mask=mask, enforce_sorted=enforce_sorted)
        log_prob, _, attn_weights = self.decode(x2, context, state, attention_mask=mask)

        return log_prob, attn_weights

    def attention_mask(self, length: torch.Tensor, fix_len: int):
        mask = (torch.arange(0, fix_len).to(length.device) >= length.unsqueeze(-1))
        return mask.unsqueeze(1)

    def save_trained(self, dirname):
        self.config.save_config(os.path.join(dirname, 'config.json'))
        torch.save(self.state_dict(), os.path.join(dirname, 'model.bin'))

    @classmethod
    def from_trained(cls, dirname):
        config = Seq2seqConfig.from_config(os.path.join(dirname, 'config.json'))
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(dirname, 'model.bin'), map_location='cpu'))
        return model
