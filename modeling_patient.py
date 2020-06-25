#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dongyu Zhang
"""

from pytorch_transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertLayerNorm
import torch
from torch import nn
from torch.nn import init, Parameter
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
import numpy as np


class PatientLevelEmbedding(nn.Module):
    def __init__(self, config):
        super(PatientLevelEmbedding, self).__init__()
        self.config = config
        assert self.config.embed_mode in ["all", "note", "chunk", "no"]
        if self.config.embed_mode == "all":
            self.note_embedding = nn.Embedding(self.config.max_note_position_embedding, self.config.hidden_size)
            self.chunk_embedding = nn.Embedding(self.config.max_chunk_position_embedding, self.config.hidden_size)
            self.combine_embed_rep = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size)
        elif self.config.embed_mode == "note":
            self.note_embedding = nn.Embedding(self.config.max_note_position_embedding, self.config.hidden_size)
            self.combine_embed_rep = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        elif self.config.embed_mode == "chunk":
            self.chunk_embedding = nn.Embedding(self.config.max_chunk_position_embedding, self.config.hidden_size)
            self.combine_embed_rep = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        else:
            pass
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs, new_note_ids=None, new_chunk_ids=None):
        if self.config.embed_mode == "all":
            note_embeds = self.note_embedding(new_note_ids)
            chunk_embeds = self.chunk_embedding(new_chunk_ids)
            output = self.combine_embed_rep(torch.cat((inputs, note_embeds, chunk_embeds), 2))
        elif self.config.embed_mode == "note":
            note_embeds = self.note_embedding(new_note_ids)
            output = self.combine_embed_rep(torch.cat((inputs, note_embeds), 2))
        elif self.config.embed_mode == "chunk":
            chunk_embeds = self.chunk_embedding(new_chunk_ids)
            output = self.combine_embed_rep(torch.cat((inputs, chunk_embeds), 2))
        elif self.config.embed_mode == "no":
            output = inputs
        else:
            raise ValueError("The embed mode: {} is not supported".format(self.config.embed_mode))
        if self.config.embed_mode != "no":
            output = self.LayerNorm(output)
            output = self.dropout(output)
        return output


class SelfDefineBert(nn.Module):
    def __init__(self):
        super(SelfDefineBert, self).__init__()

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class PatientLevelBert(SelfDefineBert):
    def __init__(self, config):
        super(PatientLevelBert, self).__init__()
        self.config = config
        self.embeddings = PatientLevelEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def forward(self, inputs, new_note_ids=None, new_chunk_ids=None):
        device = inputs.device
        input_shape = inputs.size()[0:2]
        attention_mask = torch.ones(input_shape, device=device)
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        encoder_extended_attention_mask = None
        head_mask = [None] * self.config.num_hidden_layers
        embedding_output = self.embeddings(inputs=inputs,
                                           new_note_ids=new_note_ids,
                                           new_chunk_ids=new_chunk_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,)
        return outputs


class PatientLevelBertForSequenceClassification(SelfDefineBert):
    def __init__(self, config, num_labels):
        super(PatientLevelBertForSequenceClassification, self).__init__()
        self.config = config
        self.patient_bert = PatientLevelBert(config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        self.apply(self.init_weights)

    def forward(self, inputs, new_note_ids=None, new_chunk_ids=None, labels=None):
        outputs = self.patient_bert(inputs, new_note_ids, new_chunk_ids)
        pooled_output = outputs[1]
        pooled_output2 = self.dropout(pooled_output)
        logits = self.classifier(pooled_output2)
        pred = torch.sigmoid(logits).squeeze(1)
        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(pred, labels.float())
            return loss, pred
        else:
            return pred


class LSTMLayer(SelfDefineBert):
    def __init__(self, config, num_labels):
        super(LSTMLayer, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(self.config.hidden_size,
                            self.config.hidden_size // 2,
                            self.config.lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.embeddings = PatientLevelEmbedding(config)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        self.apply(self.init_weights)

    def forward(self, inputs, new_note_ids=None, new_chunk_ids=None, labels=None):
        device = inputs.device
        batch_size = inputs.size()[0]
        hidden = (torch.zeros((self.config.lstm_layers * 2, batch_size, self.config.hidden_size // 2), device=device),
                  torch.zeros((self.config.lstm_layers * 2, batch_size, self.config.hidden_size // 2), device=device))
        new_input = self.embeddings(inputs, new_note_ids, new_chunk_ids)
        lstm_output, hidden = self.lstm(new_input, hidden)
        loss_fct = BCELoss()
        drop_input = lstm_output[0, -1, :]
        class_input = self.dropout(drop_input)
        logits = self.classifier(class_input)
        pred = torch.sigmoid(logits)
        if labels is not None:
            loss = loss_fct(pred, labels.float().view(1))
            return loss, pred
        else:
            return pred


class TLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, config, batch_first=True, bidirectional=True):
        super(TLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.c1 = torch.Tensor([1]).float()
        self.c2 = torch.Tensor([np.e]).float()
        self.ones = torch.ones([1, self.hidden_size]).float()
        self.register_buffer('c1_const', self.c1)
        self.register_buffer('c2_const', self.c2)
        self.register_buffer("ones_const", self.ones)
        # Input Gate Parameter
        self.Wi = Parameter(torch.normal(0.0, config.initializer_range, size=(self.input_size, self.hidden_size)))
        self.Ui = Parameter(torch.normal(0.0, config.initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bi = Parameter(torch.zeros(self.hidden_size))
        # Forget Gate Parameter
        self.Wf = Parameter(torch.normal(0.0, config.initializer_range, size=(self.input_size, self.hidden_size)))
        self.Uf = Parameter(torch.normal(0.0, config.initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bf = Parameter(torch.zeros(self.hidden_size))
        # Output Gate Parameter
        self.Wog = Parameter(torch.normal(0.0, config.initializer_range, size=(self.input_size, self.hidden_size)))
        self.Uog = Parameter(torch.normal(0.0, config.initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bog = Parameter(torch.zeros(self.hidden_size))
        # Cell Layer Parameter
        self.Wc = Parameter(torch.normal(0.0, config.initializer_range, size=(self.input_size, self.hidden_size)))
        self.Uc = Parameter(torch.normal(0.0, config.initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bc = Parameter(torch.zeros(self.hidden_size))
        # Decomposition Layer Parameter
        self.W_decomp = Parameter(
            torch.normal(0.0, config.initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.b_decomp = Parameter(torch.zeros(self.hidden_size))

    def TLSTM_unit(self, prev_hidden_memory, inputs, times):
        prev_hidden_state, prev_cell = prev_hidden_memory
        x = inputs
        t = times
        T = self.map_elapse_time(t)
        C_ST = torch.tanh(torch.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = torch.mul(T, C_ST)
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input Gate
        i = torch.sigmoid(torch.matmul(x, self.Wi) +
                          torch.matmul(prev_hidden_state, self.Ui) + self.bi)
        # Forget Gate
        f = torch.sigmoid(torch.matmul(x, self.Wf) +
                          torch.matmul(prev_hidden_state, self.Uf) + self.bf)
        # Output Gate
        o = torch.sigmoid(torch.matmul(x, self.Wog) +
                          torch.matmul(prev_hidden_state, self.Uog) + self.bog)
        # Candidate Memory Cell
        C = torch.sigmoid(torch.matmul(x, self.Wc) +
                          torch.matmul(prev_hidden_state, self.Uc) + self.bc)
        # Current Memory Cell
        Ct = f * prev_cell + i * C

        # Current Hidden State
        current_hidden_state = o * torch.tanh(Ct)

        return current_hidden_state, Ct

    def map_elapse_time(self, t):
        T = torch.div(self.c1_const, torch.log(t + self.c2_const))
        T = torch.matmul(T, self.ones_const)
        return T

    def forward(self, inputs, times):
        device = inputs.device
        if self.batch_first:
            batch_size = inputs.size()[0]
            inputs = inputs.permute(1, 0, 2)
            times = times.transpose(0, 1)
        else:
            batch_size = inputs.size()[1]
        prev_hidden = torch.zeros((batch_size, self.hidden_size), device=device)
        prev_cell = torch.zeros((batch_size, self.hidden_size), device=device)
        seq_len = inputs.size()[0]
        hidden_his = []
        for i in range(seq_len):
            prev_hidden, prev_cell = self.TLSTM_unit((prev_hidden, prev_cell), inputs[i], times[i])
            hidden_his.append(prev_hidden)
        hidden_his = torch.stack(hidden_his)
        if self.bidirectional:
            second_hidden = torch.zeros((batch_size, self.hidden_size), device=device)
            second_cell = torch.zeros((batch_size, self.hidden_size), device=device)
            second_inputs = torch.flip(inputs, [0])
            second_times = torch.flip(times, [0])
            second_hidden_his = []
            for i in range(seq_len):
                if i == 0:
                    time = times[i]
                else:
                    time = second_times[i-1]
                second_hidden, second_cell = self.TLSTM_unit((second_hidden, second_cell), second_inputs[i], time)
                second_hidden_his.append(second_hidden)
            second_hidden_his = torch.stack(second_hidden_his)
            hidden_his = torch.cat((hidden_his, second_hidden_his), dim=2)
            prev_hidden = torch.cat((prev_hidden, second_hidden), dim=1)
            prev_cell = torch.cat((prev_cell, second_cell), dim=1)
        if self.batch_first:
            hidden_his = hidden_his.permute(1, 0, 2)
        return hidden_his, (prev_hidden, prev_cell)


class TLSTMLayer(SelfDefineBert):

    def __init__(self, config, num_labels):
        super(TLSTMLayer, self).__init__()
        self.config = config
        self.tlstm = TLSTM(self.config.hidden_size,
                           self.config.hidden_size // 2,
                           self.config,
                           batch_first=True,
                           bidirectional=True)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.embeddings = PatientLevelEmbedding(config)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        self.apply(self.init_weights)

    def forward(self, inputs, times, new_note_ids=None, new_chunk_ids=None, labels=None):
        new_input = self.embeddings(inputs, new_note_ids, new_chunk_ids)
        lstm_output, hidden = self.tlstm(new_input, times.float())
        loss_fct = BCEWithLogitsLoss()
        drop_input = lstm_output[0, -1, :]
        class_input = self.dropout(drop_input)
        logits = self.classifier(class_input)
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
        pred = torch.sigmoid(logits)
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        pred = torch.where(torch.isinf(pred), torch.zeros_like(pred), pred)
        if labels is not None:
            loss = loss_fct(logits, labels.float().view(1))
            return loss, pred
        else:
            return pred


class FTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, config, batch_first=True, bidirectional=True):
        super(FTLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.c1 = torch.Tensor([1]).float()
        self.c2 = torch.Tensor([np.e]).float()
        self.c3 = torch.Tensor([0.]).float()
        self.ones = torch.ones([1, self.hidden_size]).float()
        self.register_buffer('c1_const', self.c1)
        self.register_buffer('c2_const', self.c2)
        self.register_buffer('c3_const', self.c3)
        self.register_buffer("ones_const", self.ones)
        # Input Gate Parameter
        self.Wi = Parameter(torch.normal(0.0, config.initializer_range, size=(self.input_size, self.hidden_size)))
        self.Ui = Parameter(torch.normal(0.0, config.initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bi = Parameter(torch.zeros(self.hidden_size))
        # Forget Gate Parameter
        self.Wf = Parameter(torch.normal(0.0, config.initializer_range, size=(self.input_size, self.hidden_size)))
        self.Uf = Parameter(torch.normal(0.0, config.initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bf = Parameter(torch.zeros(self.hidden_size))
        # Output Gate Parameter
        self.Wog = Parameter(torch.normal(0.0, config.initializer_range, size=(self.input_size, self.hidden_size)))
        self.Uog = Parameter(torch.normal(0.0, config.initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bog = Parameter(torch.zeros(self.hidden_size))
        # Cell Layer Parameter
        self.Wc = Parameter(torch.normal(0.0, config.initializer_range, size=(self.input_size, self.hidden_size)))
        self.Uc = Parameter(torch.normal(0.0, config.initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.bc = Parameter(torch.zeros(self.hidden_size))
        # Decomposition Layer Parameter
        self.W_decomp = Parameter(
            torch.normal(0.0, config.initializer_range, size=(self.hidden_size, self.hidden_size)))
        self.b_decomp = Parameter(torch.zeros(self.hidden_size))
        # Decay Parameter
        self.W_decay_1 = Parameter(torch.tensor([[0.33]]))
        self.W_decay_2 = Parameter(torch.tensor([[0.33]]))
        self.W_decay_3 = Parameter(torch.tensor([[0.33]]))
        self.a = Parameter(torch.tensor([1.0]))
        self.b = Parameter(torch.tensor([1.0]))
        self.m = Parameter(torch.tensor([0.02]))
        self.k = Parameter(torch.tensor([2.9]))
        self.d = Parameter(torch.tensor([4.5]))
        self.n = Parameter(torch.tensor([2.5]))

    def FTLSTM_unit(self, prev_hidden_memory, inputs, times):
        prev_hidden_state, prev_cell = prev_hidden_memory
        x = inputs
        t = times
        T = self.map_elapse_time(t)
        C_ST = torch.tanh(torch.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = torch.mul(T, C_ST)
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input Gate
        i = torch.sigmoid(torch.matmul(x, self.Wi) +
                          torch.matmul(prev_hidden_state, self.Ui) + self.bi)
        # Forget Gate
        f = torch.sigmoid(torch.matmul(x, self.Wf) +
                          torch.matmul(prev_hidden_state, self.Uf) + self.bf)
        # Output Gate
        o = torch.sigmoid(torch.matmul(x, self.Wog) +
                          torch.matmul(prev_hidden_state, self.Uog) + self.bog)
        # Candidate Memory Cell
        C = torch.sigmoid(torch.matmul(x, self.Wc) +
                          torch.matmul(prev_hidden_state, self.Uc) + self.bc)
        # Current Memory Cell
        Ct = f * prev_cell + i * C

        # Current Hidden State
        current_hidden_state = o * torch.tanh(Ct)

        return current_hidden_state, Ct

    def map_elapse_time(self, t):
        T_1 = torch.div(self.c1_const, torch.mul(self.a, torch.pow(t, self.b)))
        T_2 = self.k - torch.mul(self.m, t)
        T_3 = torch.div(self.c1_const, (self.c1_const + torch.pow(torch.div(t, self.d), self.n)))
        T = torch.mul(self.W_decay_1, T_1) + torch.mul(self.W_decay_2, T_2) + torch.mul(self.W_decay_3, T_3)
        T = torch.max(T, self.c3_const)
        T = torch.min(T, self.c1_const)
        T = torch.matmul(T, self.ones_const)
        return T

    def forward(self, inputs, times):
        device = inputs.device
        if self.batch_first:
            batch_size = inputs.size()[0]
            inputs = inputs.permute(1, 0, 2)
            times = times.transpose(0, 1)
        else:
            batch_size = inputs.size()[1]
        prev_hidden = torch.zeros((batch_size, self.hidden_size), device=device)
        prev_cell = torch.zeros((batch_size, self.hidden_size), device=device)
        seq_len = inputs.size()[0]
        hidden_his = []
        for i in range(seq_len):
            prev_hidden, prev_cell = self.FTLSTM_unit((prev_hidden, prev_cell), inputs[i], times[i])
            hidden_his.append(prev_hidden)
        hidden_his = torch.stack(hidden_his)
        if self.bidirectional:
            second_hidden = torch.zeros((batch_size, self.hidden_size), device=device)
            second_cell = torch.zeros((batch_size, self.hidden_size), device=device)
            second_inputs = torch.flip(inputs, [0])
            second_times = torch.flip(times, [0])
            second_hidden_his = []
            for i in range(seq_len):
                if i == 0:
                    time = times[i]
                else:
                    time = second_times[i-1]
                second_hidden, second_cell = self.FTLSTM_unit((second_hidden, second_cell), second_inputs[i], time)
                second_hidden_his.append(second_hidden)
            second_hidden_his = torch.stack(second_hidden_his)
            hidden_his = torch.cat((hidden_his, second_hidden_his), dim=2)
            prev_hidden = torch.cat((prev_hidden, second_hidden), dim=1)
            prev_cell = torch.cat((prev_cell, second_cell), dim=1)
        if self.batch_first:
            hidden_his = hidden_his.permute(1, 0, 2)
        return hidden_his, (prev_hidden, prev_cell)


class FTLSTMLayer(SelfDefineBert):

    def __init__(self, config, num_labels):
        super(FTLSTMLayer, self).__init__()
        self.config = config
        self.ftlstm = FTLSTM(self.config.hidden_size,
                           self.config.hidden_size // 2,
                           self.config,
                           batch_first=True,
                           bidirectional=True)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.embeddings = PatientLevelEmbedding(config)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        self.apply(self.init_weights)

    def forward(self, inputs, times, new_note_ids=None, new_chunk_ids=None, labels=None):
        new_input = self.embeddings(inputs, new_note_ids, new_chunk_ids)
        lstm_output, hidden = self.ftlstm(new_input, times.float())
        loss_fct = BCEWithLogitsLoss()
        drop_input = lstm_output[0, -1, :]
        class_input = self.dropout(drop_input)
        logits = self.classifier(class_input)
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
        pred = torch.sigmoid(logits)
        pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
        pred = torch.where(torch.isinf(pred), torch.zeros_like(pred), pred)
        if labels is not None:
            loss = loss_fct(logits, labels.float().view(1))
            return loss, pred
        else:
            return pred
