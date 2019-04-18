from functools import partial

import torch
from math import sqrt
from torch import Tensor
from torch.nn import Module, GRU, Embedding, Parameter, Sequential, Linear, Sigmoid, Tanh, GRUCell, ELU
from torch.nn.functional import dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models import mask_from_sequence_lengths, masked_softmax
from models.attentions import DotAttention
from models.dropout import VariationalDropout, variational_dropout


def create_embedding_layer(vectors: Tensor, padding_idx) -> Embedding:
    """
    Returns an Embedding module from a tensor.
    In addition, torchtext creates 2 more indexes usually stored as index 0 and 1 for
    unknown words and padding.
    This method ensures that those 2 indexes are correctly mapped.
    :param vectors:
    :param padding_idx:
    :return:
    """
    # Padding is stored at index 0
    vectors_with_padding = torch.cat((
        torch.zeros(2, vectors.size(-1)),
        vectors

    ), dim=0)

    return Embedding.from_pretrained(vectors_with_padding, freeze=True, padding_idx=padding_idx)


def forward_packed(f, x, x_l, h0=None, padding_value: float = 0.) -> Tensor:
    """
    Forward sequence X through the function f. Before behind explicitly forwarded,
    X is packed through PyTorch's PackSequence allowing faster execution with CuDNN.
    After forwarding, X is unpacked and padded back to original state.
    :param f: The recurrent cell to forward
    :param x: Tensor[S x B x *] -> The sequence to be forwarded
    :param x_l: Tensor[B] -> Length for each of the sequences
    :param h0: Tensor[(nb_directions * nb_layers) x B x *]
    :param padding_value: float
    :return:
    """
    # Pack sequence
    x_packed = pack_padded_sequence(x, x_l, batch_first=True, enforce_sorted=False)

    # Forward through f
    f.flatten_parameters()
    h, _ = f(x_packed, h0)

    # Unpack
    return pad_packed_sequence(h, batch_first=True, padding_value=padding_value)[0]


def weights_init(x, slope: float):
    if isinstance(x, GRU):
        for n, p in x.named_parameters():
            if 'bias_ih' in n:
                torch.nn.init.normal_(p, 0, 0.01)
            elif 'bias_hh' in n:
                torch.nn.init.normal_(p, -1, 0.01)

    elif isinstance(x, GRUCell):
        torch.nn.init.xavier_normal_(x.weight_hh)
        torch.nn.init.orthogonal_(x.weight_ih)
        torch.nn.init.normal_(x.bias_ih, 0., 0.01)
        torch.nn.init.normal_(x.bias_hh, -1, 0.01)

    elif isinstance(x, Linear):
        torch.nn.init.kaiming_normal_(x.weight, slope)

        if x.bias is not None:
            torch.nn.init.zeros_(x.bias)


class ContextToQuestion(Module):
    """
    This module represent the Gated Attention-Based Recurrent Network as described
    in the section 3.2 of the paper.
    It basically mixes the context of the question to the document.

    To avoid huge memory usage, attention is a dot product as introduce in Vaswani & al 2017.
    """
    def __init__(self, in_size, hidden_size, slope: float = 0.01, dropout_p=0.2, padding_value: float = 0.):
        super(ContextToQuestion, self).__init__()

        self._dropout_p = dropout_p
        self._padding_value = padding_value

        self._attention = DotAttention(in_size, hidden_size, dropout_p, slope)
        self._gate = Sequential(
            VariationalDropout(dropout_p),
            Linear(2 * in_size, 2 * in_size, bias=False),
            Sigmoid()
        )

        self._h0 = Parameter(torch.nn.init.normal_(torch.empty(2, 1, hidden_size), 0, 0.1))
        self._h = GRU(2 * in_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, u_p, u_q, u_p_l, u_q_l, u_q_mask):
        u_c_t = self._attention(u_p, u_q, u_q_mask)
        u_c_t = torch.cat((u_p, u_c_t), dim=-1)

        # Gate
        u_c_t = u_c_t * self._gate(u_c_t)

        # GRU
        h0 = self._h0.repeat(1, u_p.size(0), 1)
        h0 = h0 + (torch.randn_like(h0) * sqrt(0.1))

        return forward_packed(self._h, u_c_t, u_p_l, h0, self._padding_value)


class SelfMatcher(Module):
    def __init__(self, in_size: int, hidden_size: int, mask_diag: bool = True,
                 slope: float = 0.01, dropout_p: float = 0., padding_value: int = 0):

        super(SelfMatcher, self).__init__()

        self._dropout_p = dropout_p
        self._padding_value = padding_value

        self._attention = DotAttention(in_size, hidden_size, dropout_p, slope, mask_diag)
        self._gate = Sequential(
            VariationalDropout(dropout_p),
            Linear(2 * in_size, 2 * in_size, bias=False),
            Sigmoid()
        )

        self._h0 = Parameter(torch.nn.init.normal_(torch.zeros(2, 1, hidden_size), 0, 0.1))
        self._h = GRU(2 * in_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, v, v_l, v_mask):
        # Attention
        u_c_t = self._attention(v, v, v_mask)

        u_c_t = torch.cat((v, u_c_t), dim=-1)

        # Gate
        u_c_t = u_c_t * self._gate(u_c_t)
        u_c_t = variational_dropout(u_c_t, self._dropout_p, self.training)

        # GRU
        h0 = self._h0.repeat(1, v.size(0), 1)
        h0 = h0 + (torch.randn_like(h0) * sqrt(0.1))
        return forward_packed(self._h, u_c_t, v_l, h0, self._padding_value)


class QuestionPooling(Module):

    def __init__(self, in_size, attention_size, dropout_p=0.2):
        super().__init__()

        self._w_q = Sequential(
            VariationalDropout(dropout_p),
            Linear(in_size, attention_size),  # Do include bias term to simulate VQ_r (Eq. 11)
            Tanh(),
            Linear(attention_size, 1)
        )

    def forward(self, u_q, u_q_mask):
        s_j = self._w_q(u_q).squeeze(-1)
        _, a_i = masked_softmax(s_j, u_q_mask)
        return torch.bmm(a_i.unsqueeze(1), u_q).squeeze(1)


class Pointer(Module):
    def __init__(self, in_size, attention_size, slope: float, dropout_p=0.2):
        super(Pointer, self).__init__()

        self._dropout_p = dropout_p
        self._w_p_h_0 = Sequential(
            VariationalDropout(dropout_p),
            Linear(in_size, attention_size, bias=False),
            ELU(slope, True)
        )

        self._w_p_h_1 = Sequential(
            VariationalDropout(dropout_p),
            Linear(in_size, attention_size, bias=False),
            ELU(slope, True)
        )

        self._w_p_h_2 = Sequential(
            VariationalDropout(dropout_p),
            Linear(attention_size, 1, bias=False),
            ELU(slope, True)
        )

        self.h = GRUCell(in_size, in_size)

    def forward(self, context, context_mask, state):
        # Cache variables
        dropout_p, training = self._dropout_p, self.training

        # Dropout over the state
        state_ = dropout(state, dropout_p, training)

        # Start prediction
        logits_s, c = self._pointer(context, state_, context_mask)

        # Go through the GRU Cell to generate next state
        c_ = dropout(c,  dropout_p, training)
        state_ = dropout(self.h(c_, state), dropout_p, training)

        # Dropout
        logits_e, _ = self._pointer(context, state_, context_mask)

        return logits_s, logits_e

    def _pointer(self, context, state, context_mask):
        # Projections
        s_h = self._w_p_h_0(context) + self._w_p_h_1(state).unsqueeze(1)
        s_j = self._w_p_h_2(torch.tanh(s_h)).squeeze(-1)

        logits, a_i = masked_softmax(s_j, context_mask)
        c_t = torch.bmm(a_i.unsqueeze(-1).transpose(1, 2), context).squeeze(1)

        return logits, c_t


class RNet(Module):
    def __init__(self, word_embedding: Tensor, encoder_dim: int,
                 encoder_layers: int, attention_size: int,
                 slope: float = 0.01, dropout_p: float = 0.0, padding_idx: int = 1):
        super().__init__()

        self._dropout_p = dropout_p
        self._word_emb = create_embedding_layer(word_embedding, padding_idx)

        self._word_enc_h0 = Parameter(torch.zeros(2 * encoder_layers, 1, encoder_dim))
        self._word_enc = GRU(self._word_emb.embedding_dim, encoder_dim,
                             encoder_layers, batch_first=True,
                             dropout=dropout_p, bidirectional=True)

        self._c2q_attn = ContextToQuestion(2 * encoder_dim, attention_size, slope, dropout_p, padding_idx)
        self._c2c_attn = SelfMatcher(2 * encoder_dim, attention_size, True, slope, dropout_p, padding_idx)
        self._que_attn = QuestionPooling(2 * encoder_dim, attention_size, dropout_p)
        self._pointer  = Pointer(2 * encoder_dim, attention_size, dropout_p)

        self.apply(partial(weights_init, slope=slope))

    def forward(self, context_w, context_c, context_l, question_w, questions_c, question_l):
        # Cache variables
        drop_p, training = self._dropout_p, self.training

        # Compute masks
        context_mask, question_mask = mask_from_sequence_lengths(context_l), mask_from_sequence_lengths(question_l)

        # Embed words
        context_e, question_e = self._word_emb(context_w), self._word_emb(question_w)
        context_e, question_e = variational_dropout(context_e, drop_p, training), \
                                variational_dropout(question_e, drop_p, training)

        # Encode words
        word_enc_h0 = self._word_enc_h0.repeat(1, context_e.size(0), 1)
        word_enc_h0 = word_enc_h0 + (torch.randn_like(word_enc_h0) * sqrt(0.1))
        context_h = forward_packed(self._word_enc, context_e, context_l, word_enc_h0, self._word_emb.padding_idx)
        question_h = forward_packed(self._word_enc, question_e, question_l, word_enc_h0, self._word_emb.padding_idx)

        # Match the query and the context together
        context_q = self._c2q_attn(context_h, question_h, context_l, question_l, question_mask)

        # Self matching
        context_sm = self._c2c_attn(context_q, context_l, context_mask)

        # Question pooling
        question_attn = self._que_attn(question_h, question_mask)

        # Pointer to answer start / answer end
        p_start, p_end = self._pointer(context_sm, context_mask, question_attn)

        # Mask padding to avoid computing gradient over
        return masked_softmax(p_start, context_mask)[0], masked_softmax(p_end, context_mask)[0]



