import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from cuda import cuda_wrapper
import torch.nn.init as init


class Attention(nn.Module):

    def __init__(self, attn_q_dim=600, attn_rel_dim=600, attn_hid_dim=100, max_seq_len=13, dropout=0.0):
        """
        """
        nn.Module.__init__(self)
        self.dropout = dropout
        self.rel_dim = attn_rel_dim
        self.hidden_dim = attn_hid_dim
        self.q_dim = attn_q_dim
        self.max_seq_len = max_seq_len
        self.q_linear = nn.Linear(in_features=self.q_dim, out_features=self.hidden_dim)
        self.rel_linear = nn.Linear(in_features=self.rel_dim, out_features=self.hidden_dim)
        if self.dropout > 0:
            self.v = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim, out_features=max_seq_len),
                nn.Dropout(self.dropout)
            )
        else:
            self.v = nn.Linear(in_features=self.hidden_dim, out_features=max_seq_len)

        self.tanh = nn.Tanh()

    def forward(self, q_lstm_out, rel_repr):
        """
        energy = v * tanh(V_q * q + V_rel * r)
        """
        if bool(q_lstm_out.shape[0] != rel_repr.shape[0]):
            pass
        score = self.tanh(self.q_linear(q_lstm_out) + self.rel_linear(rel_repr))
        energy = self.v(score)
        return energy





class Question_Encoder(nn.Module):
    '''
    return representation of word and dependency from lstm output
    '''
    def __init__(self, q_word_emb, q_dep_emb, q_word_emb_dim=300, q_dep_emb_dim=300, lstm_hidden_dim=300):
        nn.Module.__init__(self)
        self.q_word_emb_dim = q_word_emb_dim
        self.q_dep_emb_dim = q_dep_emb_dim
        self.q_word_emb = nn.Embedding.from_pretrained(q_word_emb, freeze=False)
        self.q_word_emb_grad = nn.Embedding.from_pretrained(q_word_emb, freeze=False)
        self.q_dep_emb = nn.Embedding.from_pretrained(q_dep_emb, freeze=False)
        self.q_word_lstm = nn.LSTM(input_size=q_word_emb_dim, hidden_size=lstm_hidden_dim, bidirectional=True,
                                   batch_first=True)
        self.q_dep_lstm = nn.LSTM(input_size=q_dep_emb_dim, hidden_size=lstm_hidden_dim, bidirectional=True,
                                  batch_first=True)

        # init.orthogonal_(self.q_word_lstm)
        # init.orthogonal_(self.q_dep_lstm)

    def forward(self, q_word_seq, q_word_lengths, q_dep_seq, q_dep_lengths):
        '''
        :return: hidden and last state of question word lstm, and question dependency lstm
        '''

        q_word_embedded = self.q_word_emb(q_word_seq.long()).float()
        q_dep_embedded = self.q_dep_emb(q_dep_seq.long()).float()
        if q_word_embedded.shape == 2:
            q_word_embedded = q_word_embedded.view(1, q_word_embedded.shape[0], q_word_embedded.shape[1]).float()
            q_dep_embedded = q_dep_embedded.view(1, q_dep_embedded.shape[0], q_dep_embedded.shape[1]).float()
        q_word_pack = pack_padded_sequence(q_word_embedded, q_word_lengths, batch_first=True)
        q_dep_pack = pack_padded_sequence(q_dep_embedded, q_dep_lengths, batch_first=True)
        q_w_lstm_packed_out, (q_w_lstm_hid, _) = self.q_word_lstm(q_word_pack)
        q_w_lstm_out, q_word_lengths = pad_packed_sequence(q_w_lstm_packed_out, batch_first=True, padding_value=0)
        q_w_lstm_last = q_w_lstm_out[:, -1, :] # last hidden state
        q_dep_lstm_packed_out, (q_dep_lstm_hid, _) = self.q_dep_lstm(q_dep_pack)
        q_dep_lstm_out, q_dep_lengths = pad_packed_sequence(q_dep_lstm_packed_out, batch_first=True, padding_value=0)
        q_dep_lstm_last = q_dep_lstm_out[:, -1, :]
        return q_w_lstm_last, q_w_lstm_out, q_dep_lstm_last, q_dep_lstm_out


class Relation_Constraint_Encoder(nn.Module):

    def __init__(self, rel_word_emb, rel_id_emb, constraint_word_emb, constraint_id_emb, word_emb_dim=300, id_emb_dim=300):
        nn.Module.__init__(self)
        self.word_emb_dim = word_emb_dim
        self.id_emb_dim = id_emb_dim
        self.rel_word_emb = nn.Embedding.from_pretrained(rel_word_emb, freeze=False)
        self.rel_id_emb = nn.Embedding.from_pretrained(rel_id_emb, freeze=False)
        if constraint_id_emb is not None:
            self.constraint_word_emb = nn.Embedding.from_pretrained(constraint_word_emb, freeze=False)
            self.constraint_id_emb = nn.Embedding.from_pretrained(constraint_id_emb, freeze=False)

    def forward(self, rel_word, rel_word_lengths, rel_id, constraint_word=None, constraint_word_lengths=None, constraint_id=None,
                constraint_id_lengths=None, use_constraint=False, pooling=False):
        '''
        :return: max pool(average_emb of rel and constraint)
        '''
        if use_constraint and pooling:
            batch = rel_word.shape[0]
            rel_word_embedded = self.rel_word_emb(rel_word.long()).float()
            rel_word_avg = cuda_wrapper(torch.Tensor(batch, self.word_emb_dim))
            for i in range(batch):
                rel_word_avg[i] = rel_word_embedded[i][:rel_word_lengths[i], :].float().mean(dim=0)
            rel_id_embedded = self.rel_id_emb(rel_id.long()).reshape(batch, self.word_emb_dim).float()
            batch, cons_num, cons_seq_len = constraint_word.shape
            pooled_results = cuda_wrapper(torch.zeros(batch, self.word_emb_dim + self.id_emb_dim))
            # process per batch
            for batch_idx in range(batch):
                # count effective constraint number
                effective_constraint_num = 0
                for cons_num_idx in range(cons_num):
                    if not bool(constraint_id[batch_idx][cons_num_idx][0] == 0):
                        effective_constraint_num += 1
                    else:
                        break

                if effective_constraint_num == 0:
                    pooled_results[batch_idx] = torch.cat([rel_word_avg[batch_idx], rel_id_embedded[batch_idx]])
                else:
                    # fill into pooled_results
                    constraint_word_avg = cuda_wrapper(torch.zeros(effective_constraint_num, self.word_emb_dim))
                    for cons_num_idx in range(effective_constraint_num):
                        effective_cons_word_num = 0
                        # count effective cons word number first
                        for cons_word_idx in range(cons_seq_len):
                            if not bool(constraint_word[batch_idx][cons_num_idx][cons_word_idx] == 0):
                                effective_cons_word_num += 1
                            else:
                                break # will be zero anyway after this because of padding
                        this_cons_word_embedded = self.constraint_word_emb(constraint_word[batch_idx][cons_num_idx][:effective_cons_word_num].long()).float()
                        constraint_word_avg[cons_num_idx] = this_cons_word_embedded.mean(dim=0)


                    constraint_id_embedded = self.constraint_id_emb(constraint_id[batch_idx][:effective_constraint_num].long()).float().reshape(effective_constraint_num, self.id_emb_dim)
                    constraint_repr = torch.cat([constraint_word_avg, constraint_id_embedded], dim=1)
                    rel_repr = torch.cat([rel_word_avg[batch_idx], rel_id_embedded[batch_idx]], dim=0).reshape(1, self.word_emb_dim + self.id_emb_dim)
                    concat_repr = torch.cat([rel_repr, constraint_repr], dim=0)
                    pooled_results[batch_idx] = concat_repr.max(dim=0)[0]
            return pooled_results
        else:
            batch = rel_word.shape[0]
            rel_word_embedded = self.rel_word_emb(rel_word.long()).float()
            rel_word_avg = cuda_wrapper(torch.Tensor(batch, self.word_emb_dim))
            for i in range(batch):
                rel_word_avg[i] = rel_word_embedded[i][:rel_word_lengths[i], :].float().mean(dim=0)
            rel_id_embedded = self.rel_id_emb(rel_id.long()).reshape(batch, self.word_emb_dim).float()
            res = torch.cat([rel_word_avg, rel_id_embedded], dim=1)

            return res

class Relation_Constraint_Encoder_NOIDEMB(nn.Module):

    def __init__(self, rel_word_emb, constraint_word_emb, word_emb_dim=300):
        nn.Module.__init__(self)
        self.word_emb_dim = word_emb_dim
        self.rel_word_emb = nn.Embedding.from_pretrained(rel_word_emb, freeze=False)
        if constraint_word_emb is not None:
            self.constraint_word_emb = nn.Embedding.from_pretrained(constraint_word_emb, freeze=False)

    def forward(self, rel_word, rel_word_lengths, constraint_word=None, constraint_word_lengths=None, constraint_id=None,
                constraint_id_lengths=None, use_constraint=False, pooling=False):
        '''
        :return: max pool(average_emb of rel and constraint)
        '''
        if use_constraint and pooling:
            batch = rel_word.shape[0]
            rel_word_embedded = self.rel_word_emb(rel_word.long()).float()
            rel_word_avg = cuda_wrapper(torch.Tensor(batch, self.word_emb_dim))
            for i in range(batch):
                rel_word_avg[i] = rel_word_embedded[i][:rel_word_lengths[i], :].float().mean(dim=0)
            batch, cons_num, cons_seq_len = constraint_word.shape
            pooled_results = cuda_wrapper(torch.zeros(batch, self.word_emb_dim))
            # process per batch
            for batch_idx in range(batch):
                # count effective constraint number
                effective_constraint_num = 0
                for cons_num_idx in range(cons_num):
                    if not bool(constraint_id[batch_idx][cons_num_idx][0] == 0):
                        effective_constraint_num += 1
                    else:
                        break

                if effective_constraint_num == 0:
                    pooled_results[batch_idx] = torch.cat([rel_word_avg[batch_idx]])
                else:
                    # fill into pooled_results
                    constraint_word_avg = cuda_wrapper(torch.zeros(effective_constraint_num, self.word_emb_dim))
                    for cons_num_idx in range(effective_constraint_num):
                        effective_cons_word_num = 0
                        # count effective cons word number first
                        for cons_word_idx in range(cons_seq_len):
                            if not bool(constraint_word[batch_idx][cons_num_idx][cons_word_idx] == 0):
                                effective_cons_word_num += 1
                            else:
                                break # will be zero anyway after this because of padding
                        this_cons_word_embedded = self.constraint_word_emb(constraint_word[batch_idx][cons_num_idx][:effective_cons_word_num].long()).float()
                        constraint_word_avg[cons_num_idx] = this_cons_word_embedded.mean(dim=0)

                    constraint_repr = torch.cat([constraint_word_avg], dim=1)
                    rel_repr = torch.cat([rel_word_avg[batch_idx]], dim=0).reshape(1, self.word_emb_dim)
                    concat_repr = torch.cat([rel_repr, constraint_repr], dim=0)
                    pooled_results[batch_idx] = concat_repr.max(dim=0)[0]
            return pooled_results
        else:
            batch = rel_word.shape[0]
            rel_word_embedded = self.rel_word_emb(rel_word.long()).float()
            rel_word_avg = cuda_wrapper(torch.Tensor(batch, self.word_emb_dim))
            for i in range(batch):
                rel_word_avg[i] = rel_word_embedded[i][:rel_word_lengths[i], :].float().mean(dim=0)
            res = torch.cat([rel_word_avg], dim=1)
            return res



class ComplexWebQSP_Model(nn.Module):
    def __init__(self, q_word_emb, q_dep_emb, rel_word_emb, rel_id_emb, use_attn=False, use_constraint=False,
                 constraint_word_emb=None, constraint_id_emb=None, use_el_score=False, use_rel_id=True,
                 q_word_emb_dim=300, q_dep_emb_dim=300, lstm_hidden_dim=300,
                 word_emb_dim=300, id_emb_dim=300,
                 attn_q_dim=600, attn_rel_dim=600, attn_hid_dim=100, max_seq_len=13, dropout=0.0,
                 linear_hid_dim=1024, output_dim=2,):

        nn.Module.__init__(self)
        self.use_attn = use_attn
        self.use_constraint = use_constraint
        self.use_el_score = use_el_score
        self.use_rel_id = use_rel_id
        self.q_encoder = Question_Encoder(q_word_emb=q_word_emb, q_dep_emb=q_dep_emb, q_word_emb_dim=q_word_emb_dim,
                                          q_dep_emb_dim=q_dep_emb_dim, lstm_hidden_dim=lstm_hidden_dim)
        if use_attn:
            if self.use_rel_id:
                assert attn_rel_dim == id_emb_dim + word_emb_dim
            else:
                assert attn_rel_dim == word_emb_dim

            self.attn = Attention(attn_q_dim=attn_q_dim, attn_rel_dim=attn_rel_dim, attn_hid_dim=attn_hid_dim,
                                  max_seq_len=max_seq_len, dropout=dropout)
        if use_rel_id:
            print 'use id emb...'
            self.query_graph_encoder = Relation_Constraint_Encoder(rel_word_emb=rel_word_emb, rel_id_emb=rel_id_emb,
                                                                   word_emb_dim=word_emb_dim, id_emb_dim=id_emb_dim,
                                                                   constraint_word_emb=constraint_word_emb,
                                                                   constraint_id_emb=constraint_id_emb)

        else:
            print 'no id emb...'
            self.query_graph_encoder = Relation_Constraint_Encoder_NOIDEMB(rel_word_emb=rel_word_emb,
                                                                           constraint_word_emb=constraint_word_emb, word_emb_dim=300)
        self.linear_hid_dim = linear_hid_dim
        self.output_dim = output_dim
        if self.use_el_score:
            if self.use_rel_id:
                self.linear_1 = nn.Linear(in_features=q_dep_emb_dim * 2 + q_word_emb_dim * 2 + word_emb_dim + id_emb_dim + 1,
                                          out_features=linear_hid_dim)
            else:
                self.linear_1 = nn.Linear(in_features=q_dep_emb_dim * 2 + q_word_emb_dim * 2 + word_emb_dim + 1,
                                          out_features=linear_hid_dim)
        else:
            if self.use_rel_id:
                self.linear_1 = nn.Linear(in_features=q_dep_emb_dim * 2 + q_word_emb_dim * 2 + word_emb_dim + id_emb_dim,
                                          out_features=linear_hid_dim)
            else:
                self.linear_1 = nn.Linear(in_features=q_dep_emb_dim * 2 + q_word_emb_dim * 2 + word_emb_dim,
                                          out_features=linear_hid_dim)
        self.act1 = nn.Sigmoid()
        self.linear_2 = nn.Linear(in_features=linear_hid_dim, out_features=output_dim)
        self.act2 = nn.Sigmoid()

    def forward(self, padded_q_word_seq, q_word_lengths, padded_q_dep_seq, q_dep_lengths, padded_rel_words_seq,
                rel_word_lengths, rel_ids, padded_constraint_words_seq=None, constraint_word_lengths=None,
                padded_constraint_ids=None, constraint_id_lengths=None, el_scores=None, pooling=True):
        q_w_lstm_last, q_w_lstm_hid, q_dep_lstm_last, q_dep_lstm_hid = self.q_encoder(padded_q_word_seq, q_word_lengths,
                                                                                      padded_q_dep_seq, q_dep_lengths)
        query = None
        if self.use_rel_id:
            query = self.query_graph_encoder(rel_word=padded_rel_words_seq, rel_word_lengths=rel_word_lengths,
                                             constraint_word=padded_constraint_words_seq, rel_id=rel_ids,
                                             use_constraint=self.use_constraint,
                                             constraint_word_lengths=constraint_word_lengths,
                                             constraint_id=padded_constraint_ids,
                                             constraint_id_lengths=constraint_id_lengths,
                                             pooling=pooling)
        else:
            query = self.query_graph_encoder(
                rel_word=padded_rel_words_seq, rel_word_lengths=rel_word_lengths, constraint_word=padded_constraint_words_seq,
                constraint_word_lengths=constraint_word_lengths, constraint_id=padded_constraint_ids,
                constraint_id_lengths=constraint_id_lengths, use_constraint=self.use_constraint, pooling=pooling)

        context = None
        if self.use_attn:
            energy = self.attn(q_w_lstm_last, query)
            batch_size = q_word_lengths.shape[0]
            # mask first
            mask = []
            for i in range(batch_size):
                mask.append([0] * q_word_lengths[i].item() + [1] * (energy.shape[-1] - q_word_lengths[i]).item())
            # mask input seq length
            mask = cuda_wrapper(torch.tensor(mask).byte())  # batch * maxseq
            if bool(mask.shape[1] != energy.shape[1]):
                assert StandardError("Wrong size in energy and mask")
            masked_energy = energy.masked_fill(mask, 0)
            deducted_energy = masked_energy[:, :q_word_lengths[0]]
            weights = F.softmax(deducted_energy, dim=-1)
            weights = weights.reshape(batch_size, 1, weights.shape[1])  # size: batch * 1 * maxseq_b
            context = torch.bmm(weights, q_w_lstm_hid).reshape(batch_size, -1)
        else:
            context = query

        linear_in = None
        if self.use_el_score:
            linear_in = torch.cat([context, q_dep_lstm_last, query, el_scores], dim=1)
        else:
            linear_in = torch.cat([context, q_dep_lstm_last, query], dim=1)

        linear1_o = self.linear_1(linear_in)
        linear1_a = self.act1(linear1_o)
        linear2_o = self.linear_2(linear1_a)
        if self.output_dim == 2:
            return linear2_o # Note cross entropy contains softmax step already
        linear2_act = self.act2(linear2_o)
        return linear2_act
