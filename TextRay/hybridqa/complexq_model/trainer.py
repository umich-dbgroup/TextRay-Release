import torch
import torch.nn as nn
from nn_Modules.model import ComplexWebQSP_Model
from utils import Progbar
from data_loader import ComplexWebSP_Train_Data
from torch.utils import data
import cPickle
import random
import os
import numpy as np
from nn_Modules.cuda import obj_to_tensor,cuda_wrapper
from nn_Modules.loss import Dynamic_Cross_Entropy_Loss
import argparse
import math
import sys
import argparse


class ComplexWebQSP_Trainer:
    '''
    Ingredients
    question: q_word, q_dep
    rel: rel_word, rel_id
    constraints: cons_word, cons_ids
    '''
    def __init__(self, dataset, q_word_emb, q_dep_emb, rel_word_emb, rel_id_emb, use_attn=False, use_constraint=False,
                 constraint_word_emb=None, constraint_id_emb=None, always_pooling=False, use_el_score=False,
                 use_prior_weights=False, q_word_emb_dim=300, q_dep_emb_dim=300, lstm_hidden_dim=300, rel_word_emb_dim=300,
                 rel_id_emb_dim=300, attn_q_dim=600, attn_rel_dim=600, attn_hid_dim=100, dropout=0.0, max_seq_len=13,
                 linear_hid_dim=1024, output_dim=2, max_epoch=20, optim='adam', lr=0.001, reward_threshold=0.5,
                 pooling_threshold=1, batch_size=64, momentum=0, weight_decay=0, lr_gamma=0.1):
        self.sudo_batch = 1
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.reward_threshold = cuda_wrapper(torch.tensor(reward_threshold))
        self.pooling_threshold = pooling_threshold
        self.use_constraint = use_constraint
        self.use_attn = use_attn
        self.always_pooling = always_pooling
        self.use_el_score = use_el_score
        self.use_prior_weights = use_prior_weights
        self.data_loader = dataset
        self.model = cuda_wrapper(ComplexWebQSP_Model(
            q_word_emb=q_word_emb, q_dep_emb=q_dep_emb, rel_word_emb=rel_word_emb,
            rel_id_emb=rel_id_emb, use_constraint=use_constraint, use_attn=use_attn,
            constraint_id_emb=constraint_id_emb, constraint_word_emb=constraint_word_emb,
            q_word_emb_dim=q_word_emb_dim, q_dep_emb_dim=q_dep_emb_dim,
            lstm_hidden_dim=lstm_hidden_dim, word_emb_dim=rel_word_emb_dim,
            id_emb_dim=rel_id_emb_dim, attn_q_dim=attn_q_dim, attn_rel_dim=attn_rel_dim,
            attn_hid_dim=attn_hid_dim, max_seq_len=max_seq_len, dropout=dropout,
            linear_hid_dim=linear_hid_dim, output_dim=output_dim, use_rel_id=True))
        if optim.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=lr_gamma, milestones=[3, 5, 7])
        sys.stderr.write("Train model config:\n q_word_emb: {}, q_dep_emb: {}\n".format(q_word_emb.shape, q_dep_emb.shape))


    def unpack_data_dict(self, data_dict):
        if self.use_constraint:
            return data_dict['padded_q_word'], data_dict['q_word_lengths'], data_dict['padded_q_dep'], data_dict['q_dep_lengths'], \
                   data_dict['padded_rel_words'], data_dict['rel_word_lengths'], data_dict['rel_ids'], data_dict['constraint_word'], \
                   data_dict['constraint_word_lengths'], data_dict['constraint_ids'], data_dict['constraint_id_lengths'], \
                   data_dict['prior_weights'], data_dict['labels']
        else:
            return data_dict['padded_q_word'], data_dict['q_word_lengths'], data_dict['padded_q_dep'], data_dict['q_dep_lengths'], \
                   data_dict['padded_rel_words'], data_dict['rel_word_lengths'], data_dict['rel_ids'], None, \
                   None, None, None, data_dict['prior_weights'], data_dict['labels']

    def eval(self):
        criterion = Dynamic_Cross_Entropy_Loss()
        criterion.eval()
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for idx in range(len(self.data_loader)):
                data_dict = self.data_loader.get_one_batch(idx)
                padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, batch_rel_ids, \
                padded_cons_words, cons_word_lengths, padded_cons_id, cons_id_lengths, batch_prior_weights, \
                batch_labels = self.unpack_data_dict(data_dict)
                if not self.data_loader.cpu_data:
                    out = self.model(padded_q_word_seq=padded_q_word, q_word_lengths=q_word_lengths,
                                     padded_q_dep_seq=padded_q_dep, q_dep_lengths=q_dep_lengths,
                                     padded_rel_words_seq=padded_rel_words, rel_word_lengths=rel_word_lengths,
                                     rel_ids=batch_rel_ids, padded_constraint_words_seq=padded_cons_words,
                                     constraint_word_lengths=cons_word_lengths, padded_constraint_ids=padded_cons_id,
                                     constraint_id_lengths=cons_id_lengths, pooling=self.use_constraint)
                else:
                    out = self.model(padded_q_word_seq=cuda_wrapper(padded_q_word), q_word_lengths=q_word_lengths,
                                     padded_q_dep_seq=cuda_wrapper(padded_q_dep), q_dep_lengths=q_dep_lengths,
                                     padded_rel_words_seq=cuda_wrapper(padded_rel_words),
                                     rel_word_lengths=rel_word_lengths, rel_ids=cuda_wrapper(batch_rel_ids),
                                     padded_constraint_words_seq=cuda_wrapper(padded_cons_words),
                                     constraint_word_lengths=cons_word_lengths,
                                     padded_constraint_ids=cuda_wrapper(padded_cons_id),
                                     constraint_id_lengths=cons_id_lengths, pooling=self.use_constraint)
                if self.use_prior_weights:
                    loss += criterion.forward(out, cuda_wrapper(batch_labels.long()), cuda_wrapper(batch_prior_weights)).item()
                else:
                    loss += criterion.forward(out, cuda_wrapper(batch_labels.long()), None).item()

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        return loss


    def train(self, save_dir=None):
        criterion = Dynamic_Cross_Entropy_Loss()
        padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, \
        batch_rel_ids, padded_cons_words, cons_word_lengths, padded_cons_id, cons_id_lengths, batch_prior_weights, \
        batch_labels = None, None, None, None, None, None, None, None, None, None, None, None, None
        epoch_loss_history = [float("inf"), ]
        sys.stderr.write('Max Epoch: {}\n'.format(self.max_epoch))
        for epoch in range(self.max_epoch):
            self.scheduler.step(epoch)
            progbar = Progbar(len(self.data_loader), file=sys.stderr)
            prog_idx = 0
            shuffled_indices = random.sample(range(len(self.data_loader)), len(self.data_loader))
            random.shuffle(shuffled_indices)
            for curr_index in shuffled_indices:
                data_dict = self.data_loader.get_one_batch(curr_index)
                padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, batch_rel_ids, \
                padded_cons_words,cons_word_lengths, padded_cons_id, cons_id_lengths, batch_prior_weights, batch_labels \
                    = self.unpack_data_dict(data_dict)
                # print "curr_idx: {}".format(curr_index)
                # print "padded_q_word: {}".format(padded_q_word)
                self.model.zero_grad()
                self.optimizer.zero_grad()
                # train
                if epoch <= self.pooling_threshold:
                    if self.data_loader.use_entity_type:
                        self.model.q_encoder.q_word_emb.weight[50:].requres_grad = False
                    else:
                        self.model.q_encoder.q_word_emb.weight[4:].requres_grad = False

                '''don't need VAR(use_constraint) because model has been initialized'''
                out = None # placeholder
                if not self.data_loader.cpu_data:
                    out = self.model(padded_q_word_seq=padded_q_word, q_word_lengths=q_word_lengths,
                                     padded_q_dep_seq=padded_q_dep, q_dep_lengths=q_dep_lengths,
                                     padded_rel_words_seq=padded_rel_words, rel_word_lengths=rel_word_lengths,
                                     rel_ids=batch_rel_ids, padded_constraint_words_seq=padded_cons_words,
                                     constraint_word_lengths=cons_word_lengths, padded_constraint_ids=padded_cons_id,
                                     constraint_id_lengths=cons_id_lengths, pooling=self.pooling_criterion(epoch))
                else:
                    out = self.model(padded_q_word_seq=cuda_wrapper(padded_q_word), q_word_lengths=q_word_lengths,
                                     padded_q_dep_seq=cuda_wrapper(padded_q_dep), q_dep_lengths=q_dep_lengths,
                                     padded_rel_words_seq=cuda_wrapper(padded_rel_words),
                                     rel_word_lengths=rel_word_lengths, rel_ids=cuda_wrapper(batch_rel_ids),
                                     padded_constraint_words_seq=cuda_wrapper(padded_cons_words),
                                     constraint_word_lengths=cons_word_lengths,
                                     padded_constraint_ids=cuda_wrapper(padded_cons_id),
                                     constraint_id_lengths=cons_id_lengths, pooling=self.pooling_criterion(epoch))
                loss = None # placeholder
                if self.use_prior_weights:
                    loss = criterion.forward(out, cuda_wrapper(batch_labels.long()), cuda_wrapper(batch_prior_weights))
                else:
                    loss = criterion.forward(out, cuda_wrapper(batch_labels.long()), None)
                loss.backward()
                # print "epoch: {}, iter: {}, loss: {}".format(epoch, prog_idx, loss.item())
                self.optimizer.step()
                if epoch <= self.pooling_threshold:
                    self.model.q_encoder.q_word_emb.weight.requires_grad = True
                if self.use_constraint:
                    self.model.query_graph_encoder.constraint_word_emb.weight.requires_grad = True
                    self.model.query_graph_encoder.constraint_id_emb.weight.requires_grad = True
                progbar.update(prog_idx + 1, [("loss", loss.item())])
                prog_idx += 1
            #epoch_loss = self.eval()
            #epoch_loss_history.append(epoch_loss)
            #sys.stderr.write("Epoch: {}, Loss: {}\n".format(epoch, epoch_loss))

            #print "Epoch: {}, Loss: {}".format(epoch, epoch_loss)
            if epoch == self.max_epoch - 1 or epoch % 3 == 2:
                epoch_loss = self.eval()
                sys.stderr.write("Epoch: {}, Loss: {}\n".format(epoch, epoch_loss))
            if save_dir is not None:
                check_point = {
                    #'loss': epoch_loss,
                    'state_dict': self.model.state_dict()
                }
                torch.save(check_point, os.path.join(save_dir, str(epoch)))
            # if self.early_stopping(epoch_loss_history[:-1], epoch_loss):
            #     break

    def early_stopping(self, loss_history, curr_loss):
        if curr_loss > loss_history[-1] and curr_loss > loss_history[-2]:
            return True
        if len(loss_history) > 6 and curr_loss > loss_history[-1]:
            return True
        return False

    def pooling_criterion(self, epoch_num):
        if self.always_pooling:
            return True
        return epoch_num > self.pooling_threshold


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_lr', type=float, default=0.0005)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--device', default=0)
    parser.add_argument('--save_dir', type=str, default='/public/ComplexWebQuestions_Resources/0418_nodep')
    args = parser.parse_args()
    start_lr = args.start_lr
    lr_gamma = args.lr_gamma
    weight_decay = args.weight_decay
    device = int(args.device)
    save_dir = args.save_dir
    torch.cuda.set_device(device)

    dataset_path = '/public/ComplexWebQuestions_Resources/intermediate_data/rescaled_max_priors_derived_0.5_train_processed.json'
    vocab_prefix = '/public/ComplexWebQuestions_Resources/vocab'
    q_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'question_word_to_idx_2.pickle'), 'rb'))
    q_dep_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'question_dep_to_idx_2.pickle'), 'rb'))
    rel_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'rel_word_to_idx.pickle'), 'rb'))
    rel_id_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'rel_id_to_idx_word.pickle'), 'rb'))
    q_word_emb = cPickle.load(open(os.path.join(vocab_prefix, 'question_word_emb_tensor_2'), 'rb'))
    q_dep_emb = cPickle.load(open(os.path.join(vocab_prefix, 'question_dep_emb_tensor_2')))
    rel_word_emb = cPickle.load(open(os.path.join(vocab_prefix, 'rel_word_emb_word_tensor'), 'rb'))
    rel_id_emb = cPickle.load(open(os.path.join(vocab_prefix, 'rel_id_emb_word_tensor'), 'rb'))
    constraint_id_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_id_to_idx.pickle'), 'rb'))
    constraint_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_word_to_idx.pickle'), 'rb'))
    constraint_id_emb = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_id_emb_tensor'), 'rb'))
    constraint_word_emb = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_word_emb_tensor'), 'rb'))
    max_epoch = 6
    use_attn = True
    use_constraint = True
    use_dep = False
    use_cache = True
    cache_prefix = 'labelpr_train'
    print "Config, start_lr: {}, lr_gamma: {}, device: {}".format(start_lr, lr_gamma, device)
    data_loader = ComplexWebSP_Train_Data(
        dataset_path=dataset_path, q_word_to_idx=q_word_to_idx, q_dep_to_idx=q_dep_to_idx, rel_word_to_idx=rel_word_to_idx,
        rel_id_to_idx=rel_id_to_idx, use_constraint=use_constraint, max_constraint_word=10, max_constraint_num=4, use_cache=True,
        constraint_word_to_idx=constraint_word_to_idx, constraint_id_to_idx=constraint_id_to_idx, use_dep=use_dep, cache_prefix=cache_prefix
    )
    print "Finish constructing data loader.."
    trainer = ComplexWebQSP_Trainer(
        dataset=data_loader, q_word_emb=q_word_emb, q_dep_emb=q_dep_emb, rel_word_emb=rel_word_emb, rel_id_emb=rel_id_emb,
        use_attn=use_attn, use_constraint=use_constraint, constraint_id_emb=constraint_id_emb, attn_hid_dim=500,
        constraint_word_emb=constraint_word_emb, max_seq_len=data_loader.get_max_qw_len(),
        lr=start_lr, lr_gamma=lr_gamma, weight_decay=weight_decay, max_epoch=max_epoch)
    trainer.train(save_dir=save_dir)
