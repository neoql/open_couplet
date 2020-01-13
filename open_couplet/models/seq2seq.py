import math
import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F

from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .attention import ScaledDotProductAttention


def _rnn(rnn_cell, *args, **kwargs):
    rnn_cell = rnn_cell.lower()

    if rnn_cell == 'lstm':
        rnn = nn.LSTM
    elif rnn_cell == 'gru':
        rnn = nn.GRU
    else:
        raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

    return rnn(*args, **kwargs)


def gelu(x):
    """ Implementation of the gelu activation function.
        Using OpenAI GPT's gelu (not exactly the same as BERT)
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return x * cdf


class CNN(nn.Module):
    """
    CNN with separable convolution.
    """

    __constants__ = ['dropout_p']

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 kernel_size: int = 3, mid: bool = True, dropout_p: float = 0.0):
        super(CNN, self).__init__()

        padding: Tuple[int, int] = (kernel_size//2, (kernel_size - 1)//2) if mid else (kernel_size, 0)
        self.dropout_p = dropout_p
        self.mid = mid

        self.pad = nn.ConstantPad1d(padding, 0.0)
        self.conv_1 = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, padding=0)
        self.conv_2 = nn.Conv1d(hidden_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor, mem: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, ret_mem: int = 0):
        x = x.transpose(-2, -1)

        if mem is not None:
            x = torch.cat([mem[0], x], dim=-1)
        else:
            x = self.pad(x)

        h = gelu(self.conv_1(x))

        if self.dropout_p > 0:
            h = F.dropout(h, p=self.dropout_p, training=self.training)

        if mem is not None:
            h = torch.cat([mem[1], h], dim=-1)
        else:
            h = self.pad(h)

        y = self.conv_2(h)

        return y.transpose(-2, -1), (x[:, :, :ret_mem], h[:, :, :ret_mem])


class Encoder(nn.Module):
    __constants__ = ['units', 'rnn_layers', 'dropout_p', 'rnn_cell']

    def __init__(self, embedding, units, rnn_layers=1, cnn_kernel_size=3, dropout_p=0.0, rnn_cell='gru'):
        super(Encoder, self).__init__()
        rnn_cell = rnn_cell.lower()

        self.embedding: nn.Embedding = embedding
        self.units = units
        self.rnn_layers = rnn_layers
        self.dropout_p = dropout_p
        self.rnn_cell = rnn_cell

        self.cnn = CNN(units, units, units, kernel_size=cnn_kernel_size, dropout_p=dropout_p)
        self.rnn = _rnn(
            rnn_cell=rnn_cell,
            input_size=units,
            hidden_size=units,
            num_layers=rnn_layers,
            dropout=dropout_p,
            bidirectional=True,
        )

        self.norm_1 = nn.LayerNorm(units)
        self.norm_2 = nn.LayerNorm(units)

        self.h_proj = nn.Linear(2 * units, units, bias=False)
        if rnn_cell == 'lstm':
            self.c_proj = nn.Linear(2 * units, units, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, source, seq_len, enforce_sorted=False):
        """
        :param source: (batch_size, src_len) source sequences
        :param seq_len: (batch_size,) sequence length of source sequences
        :param enforce_sorted: if True, the source is expected to contain sequences sorted by
            length in a decreasing order. If False, this condition is not checked. Default: True.
        :return: context, hidden
            **context** (batch_size, src_len, units * 2):
                tensor containing the encoded features of the source sequence
            **state** (layers, batch, units * 2):
                tensor containing the features in the hidden state `h`
        """
        batch_size, src_len = source.size()

        # Embedding layer
        embedded = self.embedding(source)
        if self.dropout_p:
            embedded = F.dropout(embedded, p=self.dropout_p, training=self.training)

        # CNN layer
        cnn_out, _ = self.cnn(embedded)
        if self.dropout_p:
            cnn_out = F.dropout(cnn_out, p=self.dropout_p, training=self.training)
        cnn_out = self.norm_1(embedded + cnn_out)

        # RNN layer
        rnn_input = pack_padded_sequence(cnn_out, seq_len, batch_first=True, enforce_sorted=enforce_sorted)
        context, state = self.rnn(rnn_input)
        # (batch_size, src_len, 2, units)
        context, _ = pad_packed_sequence(context, batch_first=True).view(batch_size, src_len, 2, -1)
        if self.dropout_p:
            context = F.dropout(context, p=self.dropout_p, training=self.training)
        context = self.norm_2(context[:, :, 0] + context[:, :, 1] + cnn_out)

        # cat directions
        if isinstance(state, tuple):  # LSTM
            state = tuple([self._cat_directions(h) for h in state])
            state = (self.h_proj(state[0]), self.c_proj(state[1]))
        else:
            state = self.h_proj(self._cat_directions(state))

        return context, state

    @staticmethod
    def _cat_directions(h):
        """(layers * directions, batch, hidden_size) -> (layers, batch, directions * hidden_size)"""
        return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)


# noinspection DuplicatedCode
class Decoder(nn.Module):
    __constants__ = ['units', 'rnn_layers', 'cnn_kernel_size', 'dropout_p']

    def __init__(self, embedding: nn.Embedding, units: int, cnn_kernel_size: int = 3,
                 rnn_layers: int = 1, dropout_p: float = 0, rnn_cell: str = 'gru'):
        super(Decoder, self).__init__()

        self.embedding = embedding
        self.units = units
        self.rnn_layers = rnn_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.dropout_p = dropout_p
        self.rnn_cell = rnn_cell.lower()

        self.cnn = CNN(units, units, units, kernel_size=cnn_kernel_size,
                       dropout_p=dropout_p, mid=False)
        self.rnn = _rnn(
            rnn_cell=rnn_cell,
            input_size=units * 2,
            hidden_size=units,
            num_layers=rnn_layers,
            dropout=dropout_p,
        )

        self.norm_1 = nn.LayerNorm(units)
        self.norm_2 = nn.LayerNorm(units)
        self.norm_3 = nn.LayerNorm(units)

        self.q_proj = nn.Linear(units, units, bias=False)
        self.attention = ScaledDotProductAttention()

        self.fc_1 = nn.Linear(units, units)
        self.fc_2 = nn.Linear(units, embedding.num_embeddings)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input_seq: torch.Tensor,
                context: torch.Tensor, state: torch.Tensor, fh: Optional[torch.Tensor] = None,
                pad_mask: Optional[torch.Tensor] = None, cnn_mem: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        :param input_seq: (batch, tgt_len)
        :param context: (layers, batch, units)
        :param state: (batch_size, src_len, units)
        :param fh: (batch_size, units)
        :param pad_mask: (batch_size, src_fix_len)
        :param cnn_mem: (batch_size, units, mem_len)
        :return:
        """
        batch_size, tgt_len = input_seq.size()
        if fh is None:
            fh = torch.zeros(batch_size, self.units).to(input_seq.device)

        # Embedding layer
        # embedded: (batch_size, tgt_len, units)
        embedded = self.embedding(input_seq)
        if self.dropout_p:
            embedded = F.dropout(embedded, p=self.dropout_p, training=self.training)

        # CNN layer
        cnn_out, cnn_mem = self.cnn(embedded, mem=cnn_mem, ret_mem=self.cnn_kernel_size-1)
        if self.dropout_p:
            cnn_out = F.dropout(cnn_out, p=self.dropout_p, training=self.training)
        cnn_out = self.norm_1(cnn_out + embedded)

        outputs = []
        attn_weights = []

        for i in range(tgt_len):
            # (batch_size, tgt_len, 2*units)
            input_step = torch.cat([cnn_out[:, i, :], fh], dim=-1).unsqueeze(1)

            rnn_out, state = self.rnn(input_step, state)
            if self.dropout_p:
                rnn_out = F.dropout(rnn_out, p=self.dropout_p, training=self.training)
            rnn_out = self.norm_2(cnn_out[:, i, :].unsqueeze(1) + rnn_out)

            attn_out, attn_w = self.attention(q=self.q_proj(rnn_out), k=context, v=context, mask=pad_mask)
            attn_out = self.norm_3(attn_out + rnn_out)

            out = gelu(self.fc_1(attn_out))
            if self.dropout_p:
                out = F.dropout(out, p=self.dropout_p, training=self.training)
            fh = out

            outputs.append(out)
            attn_weights.append(attn_w)

        prob = F.softmax(self.fc_2(torch.cat(outputs, dim=1)), dim=-1)

        return prob, (state, fh, cnn_mem), torch.cat(attn_weights, dim=1)


class Seq2seq(nn.Module):
    def __init__(self, vocab_size, units, rnn_layers=1, cnn_kernel_size=3, dropout_p=0.0, rnn_cell='gru'):
        super(Seq2seq, self).__init__()

        embedding = nn.Embedding(vocab_size, units)

        self.encoder = Encoder(embedding, units, rnn_layers, cnn_kernel_size, dropout_p, rnn_cell)
        self.decoder = jit.script(Decoder(embedding, units, cnn_kernel_size, rnn_layers, dropout_p, rnn_cell))

    def forward(self):
        pass
