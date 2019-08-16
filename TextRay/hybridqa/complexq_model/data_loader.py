import torch
import cPickle
import json
from torch.utils.data.dataset import Dataset
from nn_Modules.cuda import cuda_wrapper, obj_to_tensor
import numpy as np
import os
import random
import sys
import csv
from preprocess.preprocess_utils import Relation_Executor
from tqdm import tqdm

class ComplexWebSP_Train_Data:
    def __init__(self, dataset_path, q_word_to_idx, q_dep_to_idx, rel_word_to_idx, rel_id_to_idx,
                 constraint_word_to_idx=None, constraint_id_to_idx=None, use_constraint=False,
                 max_constraint_word=10, max_constraint_num=4, reward_threshold=0.5, batch_size=64,
                 use_entity_type=False, cache_dir='/public/ComplexWebQuestions_Resources/cache', use_cache=True,
                 use_el_score=False, use_dep=True, use_prior_weights=False, cache_prefix='', enforce_cpu=False):
        self.use_constraint = use_constraint
        self.reward_threshold = reward_threshold
        self.use_entity_type = use_entity_type
        self.use_el_score = use_el_score
        self.use_prior_weights = use_prior_weights
        self.use_dep = use_dep
        self.cache_prefix = cache_prefix
        self.enforce_cpu = enforce_cpu
        self.batch_size = batch_size
        self.q_word_to_idx = q_word_to_idx
        self.q_dep_to_idx = q_dep_to_idx
        self.rel_word_to_idx = rel_word_to_idx
        self.rel_id_to_idx = rel_id_to_idx
        self.constraint_word_to_idx = constraint_word_to_idx
        self.constraint_id_to_idx = constraint_id_to_idx
        self.max_constraint_word = max_constraint_word
        self.max_constraint_num = max_constraint_num
        self.max_q_word_len, self.max_qword_ref, self.max_dep_len, self.max_dep_ref = None, None, None, None
        self.qid_to_word_seq = {} # format: qid->entity->seq
        self.qid_to_dep_seq = {}
        self.qid_to_entity_scores = {}
        self.rel_to_word = {}
        self.rel_to_id = {}
        self.constraint_to_word = {}
        self.constraint_to_id = {}
        self.batch_training_data = []
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.cpu_data = False
        with open(dataset_path, 'r') as f:
            self.questions = json.load(f)
        self.init_question_dicts()
        self.init_rel_dicts()
        self.transform_to_batch_data()

    def init_question_dicts(self):
        # qid to seqs
        if self.use_cache and (self.cache_prefix == 'fast_train' or self.cache_prefix == 'labelpr_train' or
                               self.cache_prefix == 'modellb_train' or self.cache_prefix == 'ablation') and \
                os.path.exists(os.path.join(self.cache_dir, 'train_qid_to_word_seq.pickle')) and \
           os.path.exists(os.path.join(self.cache_dir, 'train_qid_to_dep_seq.pickle')):
            self.qid_to_word_seq = cPickle.load(
                open(os.path.join(self.cache_dir, 'train_qid_to_word_seq.pickle'), 'rb'))
            self.qid_to_dep_seq = cPickle.load(
                open(os.path.join(self.cache_dir, 'train_qid_to_dep_seq.pickle'), 'rb'))
            sys.stderr.write('Fast train config, Finish loading question mapping from cache\n')
        elif self.use_cache and os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_word_seq.pickle')) and \
            os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_dep_seq.pickle')):
            self.qid_to_word_seq = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_word_seq.pickle'), 'rb'))
            self.qid_to_dep_seq = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_dep_seq.pickle'), 'rb'))
            sys.stderr.write('Finish loading question mapping from cache\n')
        else:
            print "Start init question dict..."
            for question in tqdm(self.questions):
                if question['ID'] not in self.qid_to_word_seq.keys():
                    self.qid_to_word_seq[question['ID']] = {}
                if question['ID'] not in self.qid_to_dep_seq.keys():
                    self.qid_to_dep_seq[question['ID']] = {}
                if self.use_entity_type:
                    for entity in question['entity_to_seqs'].keys():
                        seq_dict = question['entity_to_seqs'][entity]
                        self.qid_to_word_seq[question['ID']][entity] = [self.tokenTOidx(self.q_word_to_idx, i, '<WORD_UNKNOWN>')
                                                                        for i in seq_dict['WordSeqType']]
                        self.qid_to_dep_seq[question['ID']][entity] = [self.tokenTOidx(self.q_dep_to_idx, i, '<DEP_UNKNOWN>')
                                                                        for i in seq_dict['DepSeqType']]
                else:
                    for entity in question['entity_to_seqs'].keys():
                        seq_dict = question['entity_to_seqs'][entity]
                        self.qid_to_word_seq[question['ID']][entity] = [self.tokenTOidx(self.q_word_to_idx, i, '<WORD_UNKNOWN>')
                                                                        for i in seq_dict['WordSeq']]
                        if self.use_dep:
                            self.qid_to_dep_seq[question['ID']][entity] = [self.tokenTOidx(self.q_dep_to_idx, i, '<DEP_UNKNOWN>')
                                                                            for i in seq_dict['DepSeq']]
                        else:
                            self.qid_to_dep_seq[question['ID']][entity] = [self.tokenTOidx(self.q_dep_to_idx, i, '<DEP_UNKNOWN>')
                                                                            for i in seq_dict['WordSeq']]
            if self.use_cache:
                try:
                    cPickle.dump(self.qid_to_word_seq, open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_word_seq.pickle'), 'wb+'))
                    cPickle.dump(self.qid_to_dep_seq, open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_dep_seq.pickle'), 'wb+'))
                except:
                    pass

        # qid to entity scores
        if self.use_el_score and self.use_cache and os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_entity_scores.pickle')):
            self.qid_to_entity_scores = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_entity_scores.pickle'), 'rb'))
        else:
            for question in self.questions:
                if question['ID'] not in self.qid_to_entity_scores.keys():
                    self.qid_to_entity_scores[question['ID']] = {}
                for entity in question['entity_to_seqs'].keys():
                    self.qid_to_entity_scores[question['ID']][entity] = question['entity_to_seqs'][entity]['Score']
            if self.use_cache:
                try:
                    cPickle.dump(self.qid_to_entity_scores, open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_entity_scores.pickle'), 'wb+'))
                except:
                    pass


        max_q_word_len, max_q_dep_len, max_qword_ref, max_qdep_ref = 0, 0, None, None
        for idx in range(len(self.questions)):
            if self.use_entity_type:
                ques = self.questions[idx]
                for entity in ques['entity_to_seqs'].keys():
                    seq_dict = ques['entity_to_seqs'][entity]
                    if len(seq_dict['WordSeqType']) > max_q_word_len:
                        max_q_word_len, max_qword_ref = len(seq_dict['WordSeqType']), seq_dict['WordSeqType']
                    if len(seq_dict['DepSeqType']) > max_q_dep_len:
                        max_q_dep_len, max_qdep_ref = len(seq_dict['DepSeqType']), seq_dict['DepSeqType']
            else:
                ques = self.questions[idx]
                for entity in ques['entity_to_seqs'].keys():
                    seq_dict = ques['entity_to_seqs'][entity]
                    if len(seq_dict['WordSeq']) > max_q_word_len:
                        max_q_word_len, max_qword_ref = len(seq_dict['WordSeq']), seq_dict['WordSeq']
                    if len(seq_dict['DepSeq']) > max_q_dep_len:
                        max_q_dep_len, max_qdep_ref = len(seq_dict['DepSeq']), seq_dict['DepSeq']

        self.max_q_word_len = max_q_word_len
        self.max_dep_len = max_q_dep_len
        self.max_qword_ref = obj_to_tensor([self.tokenTOidx(self.q_word_to_idx, i, '<WORD_UNKNOWN>') for i in
                                            max_qword_ref], enforce_cpu=self.enforce_cpu)
        self.max_dep_ref = obj_to_tensor([self.tokenTOidx(self.q_dep_to_idx, i, '<DEP_UNKNOWN>')
                                          for i in max_qdep_ref], enforce_cpu=self.enforce_cpu)

        sys.stderr.write("Finish initializing question representation mapping...\n")

    def init_rel_dicts(self):
        self.rel_to_word[tuple([])] = [self.tokenTOidx(self.rel_word_to_idx, "<REL_WORD_UNKNOWN>")]
        self.rel_to_id[tuple([])] = self.tokenTOidx(self.rel_id_to_idx, "<REL_ID_UNKNOWN>")
        if self.use_cache and (self.cache_prefix == 'fast_train' or self.cache_prefix == 'modelpr_train' or
                               self.cache_prefix == 'labelpr_train' or self.cache_prefix == 'ablation') and \
                os.path.exists(os.path.join(self.cache_dir, 'train_rel_to_id.pickle')):
            self.rel_to_id = cPickle.load(open(os.path.join(self.cache_dir, 'train_rel_to_id.pickle'), 'rb'))
            self.rel_to_word = cPickle.load(open(os.path.join(self.cache_dir, 'train_rel_to_word.pickle'), 'rb'))
            self.constraint_to_word = cPickle.load(open(os.path.join(self.cache_dir, 'train_constraint_to_word.pickle'), 'rb'))
            self.constraint_to_id = cPickle.load(open(os.path.join(self.cache_dir, 'train_constraint_to_id.pickle'), 'rb'))
            sys.stderr.write('Fast train config, finish laoding rel from cache...\n')
        elif self.use_cache and os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_id.pickle')):
            self.rel_to_id = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_id.pickle'), 'rb'))
            self.rel_to_word = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_word.pickle'), 'rb'))
            self.constraint_to_word = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_constraint_to_word.pickle'), 'rb'))
            self.constraint_to_id = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_constraint_to_id.pickle'), 'rb'))
            sys.stderr.write('Finish loading relation mapping from cache...\n')
        else:
            print "Start init rel dict..."
            for question in tqdm(self.questions):
                candidate_paths = question['CandidatePaths']
                for start_entity in candidate_paths:
                    cand_chains = candidate_paths[start_entity]
                    for cand_chain in cand_chains:
                        rel = tuple(cand_chain['relations'])
                        if rel not in self.rel_to_word.keys():
                            if len(cand_chain['relation_words']) == 0: # special case for no relation words (only stopwords)
                                self.rel_to_word[rel] = [self.tokenTOidx(self.rel_word_to_idx, "<REL_WORD_UNKNOWN>", "<REL_WORD_UNKNOWN>")]
                            else:
                                self.rel_to_word[rel] = [self.tokenTOidx(self.rel_word_to_idx, i, "<REL_WORD_UNKNOWN>") for i in cand_chain['relation_words']]

                        if rel not in self.rel_to_id.keys():
                            self.rel_to_id[rel] = self.tokenTOidx(self.rel_id_to_idx, rel)
                        '''initilaize constraints dictionary'''
                        if self.use_constraint:
                            if "constraints" in cand_chain.keys():
                                constraints = cand_chain["constraints"]
                                for constraint in constraints:
                                    cons_rel = constraint["relation"]
                                    if cons_rel not in self.constraint_to_id.keys(): # TODO Check None Cons_Rel in preprocess
                                        self.constraint_to_id[cons_rel] = self.tokenTOidx(self.constraint_id_to_idx, cons_rel, "<CONSTRAINT_ID_PAD>")
                                    cons_rel_words = constraint["relation_words"]
                                    if len(cons_rel_words) == 0: # special case for no constraint relation words
                                        self.constraint_to_word[cons_rel] = \
                                            [self.tokenTOidx(self.constraint_word_to_idx, "<CONSTRAINT_WORD_UNKNOWN>", "<CONSTRAINT_WORD_UNKNOWN>")] \
                                            * self.max_constraint_word
                                    else:
                                        self.constraint_to_word[cons_rel] = \
                                            [self.tokenTOidx(self.constraint_word_to_idx, i, "<CONSTRAINT_WORD_UNKNOWN>") \
                                             for i in cons_rel_words] + [0] * (self.max_constraint_word - len(cons_rel_words))
            if self.use_cache and not os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_id.pickle')):
                try:
                    cPickle.dump(self.rel_to_id, open(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_id.pickle'), 'wb+'))
                    cPickle.dump(self.rel_to_word, open(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_word.pickle'), 'wb+'))
                    cPickle.dump(self.constraint_to_word, open(os.path.join(self.cache_dir, self.cache_prefix + '_constraint_to_word.pickle'), 'wb+'))
                    cPickle.dump(self.constraint_to_id, open(os.path.join(self.cache_dir, self.cache_prefix + '_constraint_to_id.pickle'), 'wb+'))
                except:
                    pass
        sys.stderr.write("Finish initializing relation representation mapping...\n")

    def tokenTOidx(self, token_dict, token, unknown=None):
        if token in token_dict.keys():
            return token_dict[token]
        else:
            return token_dict[unknown]


    def generate_one_question_batch(self, ques_examples):
        '''
        :param ques_examples: list of dictionaries,
        keys: keys: qid, topic, rel, constraints(if use constraint), reward, train_label
        :return: number of training examples
        batch add to self.batch_training_data
        '''
        unpacked_q_word_seqs, unpacked_q_dep_seqs, unpacked_rel_words, rel_ids, unpacked_constraint_words, \
        unpacked_constraint_ids, el_scores, prior_weights, labels = [], [], [], [], [], [], [], [], []
        # shuffle index to avoid all the positive examples appear simultaneously
        shuffled_indices = random.sample(range(len(ques_examples)), len(ques_examples))
        offset = 0
        training_example_num = 0
        for curr_idx in shuffled_indices:
            returned_example = ques_examples[curr_idx]
            if returned_example['topic'] not in self.qid_to_word_seq[returned_example['qid']].keys():
                continue
            if returned_example['wordseq'] is not None:
                # support constraint entity replacement
                unpacked_q_word_seqs.append(obj_to_tensor([self.tokenTOidx(self.q_word_to_idx, i, '<WORD_UNKNOWN>')
                                                           for i in returned_example['wordseq']], enforce_cpu=self.enforce_cpu))
                unpacked_q_dep_seqs.append(obj_to_tensor([self.tokenTOidx(self.q_dep_to_idx, i, '<DEP_UNKNOWN>')
                                                           for i in returned_example['depseq']], enforce_cpu=self.enforce_cpu))
            else:
                unpacked_q_word_seqs.append(obj_to_tensor(self.qid_to_word_seq[returned_example['qid']][returned_example['topic']],
                                                          enforce_cpu=self.enforce_cpu))
                unpacked_q_dep_seqs.append(obj_to_tensor(self.qid_to_dep_seq[returned_example['qid']][returned_example['topic']],
                                                         enforce_cpu=self.enforce_cpu))
            unpacked_rel_words.append(obj_to_tensor(self.rel_to_word[returned_example['rel']], enforce_cpu=self.enforce_cpu))
            rel_ids.append(obj_to_tensor(self.rel_to_id[returned_example['rel']], enforce_cpu=self.enforce_cpu))

            if self. use_el_score:
                el_scores.append(obj_to_tensor(returned_example['entity_score'], enforce_cpu=self.enforce_cpu))
            labels.append(obj_to_tensor(returned_example['train_label'], enforce_cpu=self.enforce_cpu))

            if self.use_prior_weights:
                prior_weights.append(obj_to_tensor(returned_example['prior_weights'], enforce_cpu=self.enforce_cpu))

            if self.use_constraint: # construct constraint tensor
                constraint_rel_tensor, constraint_id_tensor = None, None
                if type(returned_example['constraints']) is not tuple:
                    raise TypeError("Constraint Type Err")
                if len(returned_example['constraints']) == 0:
                    '''no constraints'''
                    constraint_rel_tensor = obj_to_tensor([[0] * self.max_constraint_word] * self.max_constraint_num,
                                                          enforce_cpu=self.enforce_cpu)
                    constraint_id_tensor = obj_to_tensor([[0]] * self.max_constraint_num, enforce_cpu=self.enforce_cpu)
                    unpacked_constraint_words.append(constraint_rel_tensor)
                    unpacked_constraint_ids.append(constraint_id_tensor)
                else:
                    constraint_rel_tensor = []
                    constraint_id_tensor = []
                    for cons in returned_example['constraints']:
                        constraint_rel_tensor.append(self.constraint_to_word[cons])
                        constraint_id_tensor.append([self.constraint_to_id[cons]])
                    cons_added = len(returned_example['constraints'])
                    while cons_added < self.max_constraint_num:
                        constraint_rel_tensor.append([0] * self.max_constraint_word)
                        constraint_id_tensor.append([0])
                        cons_added += 1
                    if not self.enforce_cpu:
                        unpacked_constraint_words.append(cuda_wrapper(torch.tensor(constraint_rel_tensor)))
                        unpacked_constraint_ids.append(cuda_wrapper(torch.tensor(constraint_id_tensor)))
                    else:
                        unpacked_constraint_words.append(torch.tensor(constraint_rel_tensor))
                        unpacked_constraint_ids.append(torch.tensor(constraint_id_tensor))
            else:
                unpacked_constraint_words.append(torch.Tensor(0)) # placeholder
                unpacked_constraint_ids.append(torch.Tensor(0)) # placeholder

            training_example_num += 1
            offset = (offset + 1) % self.batch_size

            if offset == 0 or training_example_num == len(ques_examples):
                # construct a batch of data
                padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, \
                batch_rel_ids, padded_cons_words, cons_word_lengths, padded_cons_id, cons_id_lengths, batch_el_scores, \
                batch_prior_weights, batch_labels = \
                    self.prepare_per_batch_data(unpacked_q_word_seqs, unpacked_q_dep_seqs, unpacked_rel_words, rel_ids,
                                            unpacked_constraint_words, unpacked_constraint_ids, el_scores, prior_weights, labels)

                # add to batch data
                batch_obj = {
                    'padded_q_word': padded_q_word,
                    'q_word_lengths': q_word_lengths,
                    'padded_q_dep': padded_q_dep,
                    'q_dep_lengths': q_dep_lengths,
                    'padded_rel_words': padded_rel_words,
                    'rel_word_lengths': rel_word_lengths,
                    'rel_ids': batch_rel_ids,
                    'prior_weights': None,
                    'labels': batch_labels
                }

                if self.use_constraint:
                    batch_obj['constraint_word'] = padded_cons_words
                    batch_obj['constraint_word_lengths'] = cons_word_lengths
                    batch_obj['constraint_ids'] = padded_cons_id
                    batch_obj['constraint_id_lengths'] = cons_id_lengths

                if self.use_el_score:
                    batch_obj['el_scores'] = batch_el_scores

                if self.use_prior_weights:
                    batch_obj['prior_weights'] = prior_weights

                self.batch_training_data.append(batch_obj)
                '''remember to re-initizlize everything'''
                unpacked_q_word_seqs, unpacked_q_dep_seqs, unpacked_rel_words, rel_ids, unpacked_constraint_words, \
                unpacked_constraint_ids, el_scores, prior_weights, labels = [], [], [], [], [], [], [], [], []

        return training_example_num

    def sample_data(self, pos_pool, neg_pool, std_pos_num=128, std_neg_num=192):
        if len(neg_pool) == 0:
            ref_example = pos_pool[0]
            dummy_example = {
                'qid': ref_example['qid'],
                'depseq': None,
                'wordseq': None,
                'topic': ref_example['topic'],
                'constraints': tuple([]),
                'rel': tuple([]),
                'train_label': 0,
                'prior_weights': random.uniform(0, 1)
            }
            if self.use_el_score:
                dummy_example['entity_score'] = self.qid_to_entity_scores[ref_example['qid']][ref_example['topic']]
            neg_pool.append(dummy_example)
        this_ques_examples = []

        # sample negative examples
        shuffled_indices = random.sample(range(max(len(neg_pool), std_neg_num)), std_neg_num)
        for idx in shuffled_indices:
            this_ques_examples.append(neg_pool[idx % len(neg_pool)])

        # sample positive examples
        shuffled_indices = random.sample(range(max(len(pos_pool), std_pos_num)), std_pos_num)
        for idx in shuffled_indices:
            this_ques_examples.append(pos_pool[idx % len(pos_pool)])

        return this_ques_examples

    # def sample_data(self, pos_pool, candidate_pos_pool, neg_pool, std_pos_num=128, std_neg_num=192):
    #     if len(neg_pool) == 0:
    #         ref_example = pos_pool[0]
    #         dummy_example = {
    #             'qid': ref_example['qid'],
    #             'depseq': None,
    #             'wordseq': None,
    #             'topic': ref_example['topic'],
    #             'constraints': tuple([]),
    #             'rel': tuple([]),
    #             'reward': 0,  # f1 score
    #             'train_label': 0,
    #         }
    #         if self.use_el_score:
    #             dummy_example['entity_score'] = self.qid_to_entity_scores[ref_example['qid']][ref_example['topic']]
    #         neg_pool.append(dummy_example)
    #
    #     this_ques_examples = []
    #     if len(pos_pool) != 0:
    #         for sample in candidate_pos_pool:
    #             sample['train_label'] = 0
    #         neg_pool = neg_pool + candidate_pos_pool
    #         shuffled_indices = random.sample(range(max(std_pos_num, len(pos_pool))), std_pos_num)
    #         for idx in shuffled_indices:
    #             this_ques_examples.append(pos_pool[idx % len(pos_pool)])
    #         # let candidate pos first go to negative
    #         all_cand_pos, curr_neg_num = (candidate_pos_pool >= std_neg_num), min(std_neg_num, len(candidate_pos_pool))
    #         shuffled_indices = random.sample(range(min(std_neg_num, len(candidate_pos_pool))), min(std_neg_num, len(candidate_pos_pool)))
    #         for idx in shuffled_indices:
    #             this_ques_examples.append(candidate_pos_pool[idx % len(candidate_pos_pool)])
    #         if not all_cand_pos:
    #             shuffled_indices = random.sample(range(max(std_neg_num, len(neg_pool))), std_neg_num - len(neg_pool))
    #             for idx in shuffled_indices:
    #                 this_ques_examples.append(neg_pool[idx % len(neg_pool)])
    #     else:
    #         # find best in candidate
    #         ref_example = candidate_pos_pool[0]
    #         dummy_example = {
    #             'qid': ref_example['qid'],
    #             'depseq': None,
    #             'wordseq': None,
    #             'topic': ref_example['topic'],
    #             'constraints': tuple([]),
    #             'rel': tuple([]),
    #             'reward': 1.0,  # f1 score
    #             'train_label': 1,
    #         }
    #         if self.use_el_score:
    #             dummy_example['entity_score'] = self.qid_to_entity_scores[ref_example['qid']][ref_example['topic']]
    #
    #         pos_pool.append(dummy_example)
    #         candidate_best_indices = set()
    #         best_examples = []
    #         best_reward = .0
    #         for idx in range(len(candidate_pos_pool)):
    #             cand_best = candidate_pos_pool[idx]
    #             if cand_best['reward'] > best_reward:
    #                 candidate_best_indices = {idx, }
    #                 best_examples = [cand_best, ]
    #                 best_reward = cand_best['reward']
    #             elif cand_best['reward'] == best_reward: # support multiple best examples for training
    #                 best_examples.append(cand_best)
    #                 candidate_best_indices.add(idx)
    #         # remove
    #         curr_neg_num = 0
    #         for idx in range(len(candidate_pos_pool)):
    #             curr_example = candidate_pos_pool[idx]
    #             if idx not in candidate_best_indices:
    #                 if curr_neg_num < std_neg_num:
    #                     this_ques_examples.append(curr_example)
    #                     curr_neg_num += 1
    #                 curr_example['train_label'] = 0
    #                 neg_pool.append(curr_example)
    #             else:
    #                 curr_example['train_label'] = 1
    #                 pos_pool.append(curr_example)
    #         shuffled_indices = random.sample(range(max(std_pos_num, len(pos_pool))), std_pos_num)
    #         for idx in shuffled_indices:
    #             this_ques_examples.append(pos_pool[idx % len(pos_pool)])
    #         if curr_neg_num < std_neg_num:
    #             shuffled_indices = random.sample(range(max(std_neg_num, len(neg_pool))), std_neg_num - curr_neg_num)
    #             for idx in shuffled_indices:
    #                 this_ques_examples.append(neg_pool[idx % len(neg_pool)])
    #     return this_ques_examples

    def contruct_one_question_training_dicts(self, question):
        '''
        construct a list of dictionaries for one question
        keys: qid, topic, rel, constraints(if use constraint), reward, train_label
        '''
        pos_pool, neg_pool = [], []
        candidate_paths = question['CandidatePaths']
        for start_entity in candidate_paths:
            cand_chains = candidate_paths[start_entity]
            for cand_chain in cand_chains:
                training_dict = {
                    'qid': question['ID'],
                    'depseq': None,
                    'wordseq': None,
                    'topic': start_entity,
                    'rel': tuple(cand_chain['relations']),
                    'train_label': cand_chain['derived_label']
                }
                if self.use_el_score:
                    training_dict['entity_score'] = self.qid_to_entity_scores[question['ID']][start_entity]

                if self.use_prior_weights:
                    training_dict['prior_weights'] = self.cal_loss_weight(derived_label=cand_chain['derived_label'],
                                                                          prior_score=cand_chain['scaled_prior_match_score'])

                if self.use_constraint:
                    constraints = []
                    if 'constraints' in cand_chain.keys():
                        for cons_dict in cand_chain['constraints']:
                            constraints.append(cons_dict['relation'])
                    training_dict['constraints'] = tuple(constraints)
                    if 'Cons_DepSeq' in cand_chain.keys():
                        if self.use_dep:
                            training_dict['depseq'] = cand_chain['Cons_DepSeq']
                        else:
                            training_dict['depseq'] = cand_chain['Cons_WordSeq']
                        training_dict['wordseq'] = cand_chain['Cons_WordSeq']

                if training_dict['train_label'] == 1:
                    pos_pool.append(training_dict)
                else:
                    neg_pool.append(training_dict)

        if len(pos_pool) == 0:
            print "NO POS POOL {}".format(question['ID'])
            raise AttributeError()
        this_ques_examples = self.sample_data(pos_pool, neg_pool)
        return this_ques_examples

    def transform_to_batch_data(self):
        '''
        construct a list of dictionaries
        keys: qid, topic, rel, constraints(if use constraint), reward
        '''
        total_num = 0
        qid_to_question_examples = {}
        if self.use_cache and os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_training_examples.pickle')):
            qid_to_question_examples = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_training_examples.pickle'), 'rb'))
            sys.stderr.write('Finish loading training examples from cache...\n')
        else:
            cnt = 0
            for question in self.questions:
                try:
                    ques_examples = self.contruct_one_question_training_dicts(question)
                    qid_to_question_examples[question['ID']] = ques_examples
                    cnt += 1
                except Exception as e:
                    if e is AttributeError():
                        continue
                    else:
                        print 'Exception when creating samples: {}\n'.format(question['ID'])
                        print e
                        continue
            # if not os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_training_examples.pickle')):
            #     try:
            #         cPickle.dump(qid_to_question_examples, open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_training_examples.pickle'), 'wb+'))
            #     except:
            #         pass

        if self.use_cache and os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_training_tensors.pickle')):
            self.cpu_data = True
            sys.stderr.write('Start loading batch tensors from cache...\n')
            self.batch_training_data = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_training_tensors.pickle'), 'rb'))
            sys.stderr.write('Finish loading batch training tensors from cache...\n')
            return


        cnt = 0
        for qid in qid_to_question_examples.keys():
            question_example_num = self.generate_one_question_batch(qid_to_question_examples[qid])
            total_num += question_example_num
            cnt += 1
            if cnt % 1000 == 0:
                sys.stderr.write('Finish generating {} question data\n'.format(cnt))


        sys.stderr.write("Training Examples in total: {}\n".format(total_num))
        if self.use_cache and self.enforce_cpu:
            if not os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_training_tensors.pickle')):
                try:
                    cPickle.dump(self.batch_training_data, open(os.path.join(self.cache_dir, self.cache_prefix + '_training_tensors.pickle'), 'wb+'))
                    sys.stderr.write('Finish caching batch training tensors...\n')
                    self.cpu_data = True
                except:
                    pass
        # if self.use_cache:
        #     if not os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_training_tensors.pickle')):
        #         try:
        #             cPickle.dump(self.batch_training_data, open(os.path.join(self.cache_dir, self.cache_prefix + '_training_tensors.pickle'), 'wb+'))
        #             sys.stderr.write('Finish caching batch training tensors...\n')
        #         except:
        #             pass

    def prepare_per_batch_data(self, unpacked_q_word_seqs, unpacked_q_dep_seqs, unpacked_rel_words, rel_ids,
                               unpacked_constraint_words, unpacked_constraint_ids, el_scores, prior_weights, labels):
        unpacked_q_word_seqs.append(self.max_qword_ref)
        unpacked_q_dep_seqs.append(self.max_dep_ref)
        padded_q_word, q_word_lengths, _ = self.pad_sequence(unpacked_q_word_seqs, True)
        padded_q_dep, q_dep_lengths, _ = self.pad_sequence(unpacked_q_dep_seqs, True)
        padded_rel_words, rel_word_lengths, sorted_indices = self.pad_sequence(unpacked_rel_words, False)
        batch_rel_ids = torch.zeros(len(sorted_indices))
        if not self.enforce_cpu:
            batch_rel_ids = cuda_wrapper(batch_rel_ids)
        for new_idx in range(len(sorted_indices)):
            batch_rel_ids[new_idx] = rel_ids[sorted_indices[new_idx]]
        padded_constraint_words, cons_word_lengths = None, None # placeholder
        padded_constraint_ids, cons_id_lengths = None, None # placeholder
        batch_el_scores = None # placeholder
        batch_prior_weights = None
        batch_labels = torch.zeros(len(sorted_indices))
        if not self.enforce_cpu:
            batch_labels = cuda_wrapper(batch_labels)
        for new_idx in range(len(sorted_indices)):
            batch_labels[new_idx] = labels[sorted_indices[new_idx]]

        if self.use_constraint:
            '''Note: currently it is fake lengths, just for placeholder usage'''
            cons_word_lengths, cons_id_lengths = [3] * len(sorted_indices), [3] * len(sorted_indices)
            padded_constraint_words = torch.zeros(len(sorted_indices), self.max_constraint_num, self.max_constraint_word)
            padded_constraint_ids = torch.zeros(len(sorted_indices), self.max_constraint_num, 1)
            if not self.enforce_cpu:
                padded_constraint_words = cuda_wrapper(padded_constraint_words)
                padded_constraint_ids = cuda_wrapper(padded_constraint_ids)
            for new_idx in range(len(sorted_indices)):
                padded_constraint_words[new_idx] = unpacked_constraint_words[sorted_indices[new_idx]]
                padded_constraint_ids[new_idx] = unpacked_constraint_ids[sorted_indices[new_idx]]

        if self.use_el_score:
            batch_el_scores = torch.zeros(len(sorted_indices))
            if not self.enforce_cpu:
                batch_el_scores = cuda_wrapper(batch_el_scores)
            for new_idx in range(len(sorted_indices)):
                batch_el_scores[new_idx] = el_scores[sorted_indices[new_idx]]

        if self.use_prior_weights:
            batch_prior_weights = torch.zeros(len(sorted_indices))
            if not self.enforce_cpu:
                batch_prior_weights = cuda_wrapper(batch_prior_weights)
            for new_idx in range(len(sorted_indices)):
                batch_prior_weights[new_idx] = prior_weights[sorted_indices[new_idx]]

        return padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, \
               batch_rel_ids, padded_constraint_words, cons_word_lengths, padded_constraint_ids, cons_id_lengths, batch_el_scores, \
               batch_prior_weights, batch_labels


    def pad_sequence(self, batch, remove_reference=True):
        '''
        :param batch: with reference to do padding
        '''
        sorted_indices = sorted(range(len(batch)), key=lambda x: batch[x].shape[0], reverse=True)
        sequences = [batch[i] for i in sorted_indices]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        lengths = np.array([len(x) for x in sequences])
        if remove_reference:
            return sequences_padded[1:], lengths[1:], sorted_indices
        return sequences_padded, lengths, sorted_indices

    def check_shape(self, padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words,
                    rel_word_lengths, batch_rel_ids, padded_constraint_words, constraint_word_lengths,
                    padded_constraint_ids, constraint_id_lengths, batch_labels):
        if padded_q_word.shape[0] != self.batch_size or padded_q_dep.shape[0] != self.batch_size or padded_rel_words.shape[0] != self.batch_size \
                or batch_rel_ids.shape[0] != self.batch_size or batch_labels.shape[0] != self.batch_size:
            return False
        if self.use_constraint:
            if padded_constraint_words.shape[0] != self.batch_size or padded_constraint_ids.shape[0] != self.batch_size:
                return False
        if len(q_word_lengths) != self.batch_size or len(q_dep_lengths) != self.batch_size or len(rel_word_lengths) != self.batch_size:
            return False
        if self.use_constraint:
            if len(constraint_word_lengths) != self.batch_size or len(constraint_id_lengths) != self.batch_size:
                return False
        return True

    def cal_loss_weight(self, derived_label, prior_score):
        if derived_label == 1:
            return prior_score
        else:
            return 1 - prior_score

    def get_max_qdep_len(self):
        return self.max_dep_len

    def get_max_qw_len(self):
        return self.max_q_word_len

    def __len__(self):
        return len(self.batch_training_data)

    def get_one_batch(self, idx):
        return self.batch_training_data[idx]


class ComplexWebSP_Test_Data:
    def __init__(self, dataset_path, q_word_to_idx, q_dep_to_idx, rel_word_to_idx, rel_id_to_idx, cache_dir,
                 constraint_word_to_idx=None, constraint_id_to_idx=None, use_constraint=False,
                 max_constraint_word=7, max_constraint_num=3, reward_threshold=0.5, batch_size=64,
                 use_entity_type=False, use_cache=True, use_rel_id=True, use_dep=True, cache_prefix='test'):
        self.rel_exe = Relation_Executor()
        self.use_rel_id = use_rel_id
        self.use_cache = use_cache
        self.use_dep = use_dep
        self.cache_dir = cache_dir
        self.cache_prefix = cache_prefix
        self.use_constraint = use_constraint
        self.reward_threshold = reward_threshold
        self.use_entity_type = use_entity_type
        self.batch_size = batch_size
        self.q_word_to_idx = q_word_to_idx
        self.q_dep_to_idx = q_dep_to_idx
        self.rel_word_to_idx = rel_word_to_idx
        self.rel_id_to_idx = rel_id_to_idx
        self.constraint_word_to_idx = constraint_word_to_idx
        self.constraint_id_to_idx = constraint_id_to_idx
        self.max_constraint_word = max_constraint_word
        self.max_constraint_num = max_constraint_num
        self.max_q_word_len, self.max_qword_ref, self.max_dep_len, self.max_dep_ref = None, None, None, None
        self.ques_info_lists = []
        self.qid_to_training_tensors_per_question = {}
        self.qid_to_test_indices = {}
        self.oov_relation_num = 0
        self.oov_relation_set = set()
        with open(dataset_path, 'r') as f:
            self.questions = json.load(f)

        self.init_question_dicts()
        self.init_rel_dicts()
        self.construct_examples_per_question()
        self.construct_tensors_per_question()

    def init_question_dicts(self):
        if self.use_cache and os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_word_seq.pickle')):
            self.qid_to_word_seq = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_word_seq.pickle'), 'rb'))
            self.qid_to_dep_seq = cPickle.load(open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_dep_seq.pickle'), 'rb'))
        else:
            self.qid_to_word_seq = {}
            self.qid_to_dep_seq = {}

            for question in self.questions:
                if question['ID'] not in self.qid_to_word_seq.keys():
                    self.qid_to_word_seq[question['ID']] = {}
                if question['ID'] not in self.qid_to_dep_seq.keys():
                    self.qid_to_dep_seq[question['ID']] = {}
                if self.use_entity_type:
                    for entity in question['entity_to_seqs'].keys():
                        seq_dict = question['entity_to_seqs'][entity]
                        self.qid_to_word_seq[question['ID']][entity] = [
                            self.tokenTOidx(self.q_word_to_idx, i, '<WORD_UNKNOWN>')
                            for i in seq_dict['WordSeqType']]
                        self.qid_to_dep_seq[question['ID']][entity] = [
                            self.tokenTOidx(self.q_dep_to_idx, i, '<DEP_UNKNOWN>')
                            for i in seq_dict['DepSeqType']]
                else:
                    for entity in question['entity_to_seqs'].keys():
                        seq_dict = question['entity_to_seqs'][entity]
                        self.qid_to_word_seq[question['ID']][entity] = [
                            self.tokenTOidx(self.q_word_to_idx, i, '<WORD_UNKNOWN>')
                            for i in seq_dict['WordSeq']]
                        if self.use_dep:
                            self.qid_to_dep_seq[question['ID']][entity] = [
                                self.tokenTOidx(self.q_dep_to_idx, i, '<DEP_UNKNOWN>')
                                for i in seq_dict['DepSeq']]
                        else:
                            self.qid_to_dep_seq[question['ID']][entity] = [
                            self.tokenTOidx(self.q_word_to_idx, i, '<WORD_UNKNOWN>')
                            for i in seq_dict['WordSeq']]
            if self.use_cache:
                cPickle.dump(self.qid_to_word_seq, open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_word_seq.pickle'), 'wb+'))
                cPickle.dump(self.qid_to_dep_seq, open(os.path.join(self.cache_dir, self.cache_prefix + '_qid_to_dep_seq.pickle'), 'wb+'))

        max_q_word_len, max_q_dep_len, max_qword_ref, max_qdep_ref = 0, 0, None, None
        for idx in range(len(self.questions)):
            if self.use_entity_type:
                ques = self.questions[idx]
                for entity in ques['entity_to_seqs'].keys():
                    seq_dict = ques['entity_to_seqs'][entity]
                    if len(seq_dict['WordSeqType']) > max_q_word_len:
                        max_q_word_len, max_qword_ref = len(seq_dict['WordSeqType']), seq_dict['WordSeqType']
                    if len(seq_dict['DepSeqType']) > max_q_dep_len:
                        max_q_dep_len, max_qdep_ref = len(seq_dict['DepSeqType']), seq_dict['DepSeqType']
            else:
                ques = self.questions[idx]
                for entity in ques['entity_to_seqs'].keys():
                    seq_dict = ques['entity_to_seqs'][entity]
                    if len(seq_dict['WordSeq']) > max_q_word_len:
                        max_q_word_len, max_qword_ref = len(seq_dict['WordSeq']), seq_dict['WordSeq']
                    if len(seq_dict['DepSeq']) > max_q_dep_len:
                        max_q_dep_len, max_qdep_ref = len(seq_dict['DepSeq']), seq_dict['DepSeq']

        self.max_q_word_len = max_q_word_len
        self.max_dep_len = max_q_dep_len
        print "max q word length: {}, max q dep length: {}".format(self.max_q_word_len, self.max_dep_len)
        self.max_qword_ref = obj_to_tensor([self.tokenTOidx(self.q_word_to_idx, i, '<WORD_UNKNOWN>') for i in
                                            max_qword_ref])
        self.max_dep_ref = obj_to_tensor([self.tokenTOidx(self.q_dep_to_idx, i, '<DEP_UNKNOWN>')
                                          for i in max_qdep_ref])
        print "Finish initializing question representation mapping..."

    def init_rel_dicts(self):
        if self.use_cache and os.path.exists(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_word.pickle')):
            self.rel_to_word = cPickle.load(
                open(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_word.pickle'), 'rb'))
            self.rel_to_id = cPickle.load(
                open(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_id.pickle'), 'rb'))
            self.constraint_to_word = cPickle.load(
                open(os.path.join(self.cache_dir, self.cache_prefix + '_constraint_to_word.pickle'), 'rb'))
            self.constraint_to_id = cPickle.load(
                open(os.path.join(self.cache_dir, self.cache_prefix + '_constraint_to_id.pickle'), 'rb'))
        else:
            self.rel_to_word = {}
            self.rel_to_id = {}
            self.constraint_to_word = {}
            self.constraint_to_id = {}

            for question in self.questions:
                candidate_paths = question['CandidatePaths']
                for start_entity in candidate_paths:
                    cand_chains = candidate_paths[start_entity]
                    for cand_chain in cand_chains:
                        rel = tuple(cand_chain['relations'])
                        if rel not in self.rel_to_word.keys():
                            if len(cand_chain[
                                       'relation_words']) == 0:  # special case for no relation words (only stopwords)
                                self.rel_to_word[rel] = [
                                    self.tokenTOidx(self.rel_word_to_idx, "<REL_WORD_UNKNOWN>", "<REL_WORD_UNKNOWN>")]
                            else:
                                self.rel_to_word[rel] = [self.tokenTOidx(self.rel_word_to_idx, i, "<REL_WORD_UNKNOWN>")
                                                         for i in cand_chain['relation_words']]

                        if rel not in self.rel_to_id.keys():
                            if self.use_rel_id:
                                if rel not in self.rel_id_to_idx.keys():
                                    continue
                                self.rel_to_id[rel] = self.tokenTOidx(self.rel_id_to_idx, rel)
                            else:
                                self.rel_to_id[rel] = 0 # placeholder for rel_to_id, adapt for downstream code
                        '''initilaize constraints dictionary'''
                        if self.use_constraint:
                            if "constraints" in cand_chain.keys():
                                constraints = cand_chain["constraints"]
                                for constraint in constraints:
                                    cons_rel = constraint["relation"]
                                    if cons_rel not in self.constraint_to_id.keys():  # TODO Check None Cons_Rel in preprocess
                                        self.constraint_to_id[cons_rel] = self.tokenTOidx(self.constraint_id_to_idx,
                                                                                          cons_rel,
                                                                                          "<CONSTRAINT_ID_PAD>")
                                    cons_rel_words = constraint["relation_words"]
                                    if len(cons_rel_words) == 0:  # special case for no constraint relation words
                                        self.constraint_to_word[cons_rel] = \
                                            [self.tokenTOidx(self.constraint_word_to_idx, "<CONSTRAINT_WORD_UNKNOWN>",
                                                             "<CONSTRAINT_WORD_UNKNOWN>")] \
                                            * self.max_constraint_word
                                    else:
                                        self.constraint_to_word[cons_rel] = \
                                            [self.tokenTOidx(self.constraint_word_to_idx, i,
                                                             "<CONSTRAINT_WORD_UNKNOWN>") \
                                             for i in cons_rel_words] + [0] * (
                                                        self.max_constraint_word - len(cons_rel_words))
            if self.use_cache:
                cPickle.dump(self.rel_to_id, open(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_id.pickle'), 'wb+'))
                cPickle.dump(self.rel_to_word, open(os.path.join(self.cache_dir, self.cache_prefix + '_rel_to_word.pickle'), 'wb+'))
                cPickle.dump(self.constraint_to_word, open(os.path.join(self.cache_dir, self.cache_prefix + '_constraint_to_word.pickle'), 'wb+'))
                cPickle.dump(self.constraint_to_id, open(os.path.join(self.cache_dir, self.cache_prefix + '_constraint_to_id.pickle'), 'wb+'))
        print "Finish initializing relation representation mapping..."

    def tokenTOidx(self, token_dict, token, unknown=None):
        if token in token_dict.keys():
            return token_dict[token]
        else:
            return token_dict[unknown]

    def write_sub1_file(self, file_path):
        sys.stderr.write('Start writing {}\n'.format(file_path))
        out_f = open(file_path, 'w+')
        csv_writer = csv.writer(out_f)
        if self.use_constraint:
            csv_writer.writerow(['index', 'qid', 'topic', 'relations', 'constraints'])
        else:
            csv_writer.writerow(['index', 'qid', 'topic', 'relations'])
        for qid in self.qid_to_training_examples.keys():
            examples = self.qid_to_training_examples[qid]
            for example in examples:
                if self.use_constraint:
                    csv_writer.writerow([example['index'], qid, example['topic'], example['relation'], example['constraints']])
                else:
                    csv_writer.writerow([example['index'], qid, example['topic'], example['relation']])
        out_f.close()

    def construct_examples_per_question(self):
        '''
        construct a list of dictionaries for one question
        keys: qid, topic, rel, constraints(if use constraint), reward, train_label
        '''
        tot_num = 0
        idx = 0
        if os.path.exists('temp_qid_to_training_examples.pickle'):
            self.qid_to_training_examples = cPickle.load(open('temp_qid_to_training_examples.pickle', 'rb'))
            print "Finish loading training examples.."
        else:
            self.qid_to_training_examples = {}
            for question in self.questions:
                # if question['ID'] == 'WebQTrn-2730_87deda135e48fdbc80ef86a95018f5c9':
                #     pass
                this_ques_examples = []
                candidate_paths = question['CandidatePaths']
                all_none = True # whether there is valid relation for one question
                for start_entity in candidate_paths:
                    cand_chains = candidate_paths[start_entity]
                    for cand_chain in cand_chains:
                        if self.use_rel_id:
                            if tuple(cand_chain['relations']) not in self.rel_id_to_idx.keys():
                                continue
                        curr_example = {
                            'index': idx,
                            'qid': question['ID'],
                            'wordseq': None,
                            'depseq': None,
                            'topic': start_entity,
                            'relation': tuple(cand_chain['relations']),
                        }
                        if self.use_constraint:
                            constraints = []
                            if 'constraints' in cand_chain.keys():
                                for cons_dict in cand_chain['constraints']:
                                    if cons_dict["relation"] not in self.constraint_id_to_idx.keys():
                                        continue
                                    constraints.append(cons_dict["relation"])
                            curr_example['constraints'] = tuple(constraints)
                            if 'Cons_DepSeq' in cand_chain.keys():
                                if self.use_dep:
                                    curr_example['depseq'] = cand_chain['Cons_DepSeq']
                                else:
                                    curr_example['depseq'] = cand_chain['Cons_WordSeq']
                        this_ques_examples.append(curr_example)
                        idx += 1
                        tot_num += 1
                        all_none = False
                if question['ID'] not in self.qid_to_training_examples.keys():
                    self.qid_to_training_examples[question['ID']] = []
                if not all_none:
                    self.qid_to_training_examples[question['ID']] = this_ques_examples
                else:
                    print("NO REL: {}".format(question['ID']))
            #cPickle.dump(self.qid_to_training_examples, open('temp_qid_to_training_examples.pickle', 'wb+'))

            print("Training Examples in total: {}".format(tot_num))

    def construct_tensors_per_question(self):
        unpacked_q_word_seqs, unpacked_q_dep_seqs, unpacked_rel_words, rel_ids, unpacked_constraint_words, \
        unpacked_constraint_ids, labels, indices = [], [], [], [], [], [], [], []
        padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, \
        batch_rel_ids, padded_cons_words, cons_word_lengths, padded_cons_id, cons_id_lengths, batch_indices \
            = None, None, None, None, None, None, None, None, None, None, None, None
        for qid in self.qid_to_training_examples.keys():
            if qid not in self.qid_to_test_indices.keys():
                self.qid_to_test_indices[qid] = []
            question_examples = self.qid_to_training_examples[qid]
            unpacked_q_word_seqs, unpacked_q_dep_seqs, unpacked_rel_words, rel_ids, unpacked_constraint_words, \
            unpacked_constraint_ids, labels, indices = [], [], [], [], [], [], [], []
            for example in question_examples:
                self.qid_to_test_indices[qid].append(example['index'])
                if type(example) is str:
                    raise TypeError('Example Type Err')
                if example['topic'] not in self.qid_to_word_seq[example['qid']].keys():
                    continue
                if example['wordseq'] is not None:
                    unpacked_q_word_seqs.append(obj_to_tensor([self.tokenTOidx(self.q_word_to_idx, i, '<WORD_UNKNOWN>')
                                                               for i in example['wordseq']]))
                    unpacked_q_dep_seqs.append(obj_to_tensor([self.tokenTOidx(self.q_dep_to_idx, i, '<WORD_UNKNOWN>')
                                                              for i in example['depseq']]))
                else:
                    unpacked_q_word_seqs.append(obj_to_tensor(self.qid_to_word_seq[example['qid']][example['topic']]))
                    unpacked_q_dep_seqs.append(obj_to_tensor(self.qid_to_dep_seq[example['qid']][example['topic']]))
                unpacked_rel_words.append(obj_to_tensor(self.rel_to_word[example['relation']]))
                rel_ids.append(obj_to_tensor(self.rel_to_id[example['relation']]))
                indices.append(obj_to_tensor(example['index']))
                if self.use_constraint:
                    constraint_rel_tensor, constraint_id_tensor = None, None
                    if len(example['constraints']) == 0:
                        '''no constraints'''
                        constraint_rel_tensor = obj_to_tensor([[0] * self.max_constraint_word] * self.max_constraint_num)
                        constraint_id_tensor = obj_to_tensor([[0]] * self.max_constraint_num)
                        unpacked_constraint_words.append(constraint_rel_tensor)
                        unpacked_constraint_ids.append(constraint_id_tensor)
                    else:
                        constraint_rel_tensor = []
                        constraint_id_tensor = []
                        for cons in example['constraints']:
                            constraint_rel_tensor.append(self.constraint_to_word[cons])
                            constraint_id_tensor.append([self.constraint_to_id[cons]])
                        cons_added = len(example['constraints'])
                        while cons_added < self.max_constraint_num:
                            constraint_rel_tensor.append([0] * self.max_constraint_word)
                            constraint_id_tensor.append([0])
                            cons_added += 1
                        unpacked_constraint_words.append(cuda_wrapper(torch.tensor(constraint_rel_tensor)))
                        unpacked_constraint_ids.append(cuda_wrapper(torch.tensor(constraint_id_tensor)))
                else:
                    unpacked_constraint_words.append(torch.Tensor(0))  # placeholder
                    unpacked_constraint_ids.append(torch.Tensor(0))  # placeholder

            # pad sequence
            if len(unpacked_rel_words) == 0:
                sys.stderr.write('No Topic Entities Hit {}\n'.format(qid))
                continue
            padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, \
            batch_rel_ids, padded_cons_words, cons_word_lengths, padded_cons_id, cons_id_lengths, batch_indices, _ = \
                self.prepare_batch_data(unpacked_q_word_seqs, unpacked_q_dep_seqs, unpacked_rel_words, rel_ids,
                                        unpacked_constraint_words, unpacked_constraint_ids, indices, None)

            self.qid_to_training_tensors_per_question[qid] = {
                'indices': batch_indices,
                'padded_q_word': padded_q_word,
                'q_word_lengths': q_word_lengths,
                'padded_q_dep': padded_q_dep,
                'q_dep_lengths': q_dep_lengths,
                'padded_rel_words': padded_rel_words,
                'rel_word_lengths': rel_word_lengths,
                'rel_ids': batch_rel_ids,
                'constraint_word': padded_cons_words,
                'constraint_word_lengths': cons_word_lengths,
                'constraint_ids': padded_cons_id,
                'constraint_id_lengths': cons_id_lengths}
        print "Finish constructing tensor..."

    def construct_data_dict_from_test_json(self, qid, candidate_paths):
        testing_examples = []
        idx = 0
        for start_entity in candidate_paths:
            cand_chains = candidate_paths[start_entity]
            for cand_chain in cand_chains:
                if tuple(cand_chain['relations']) not in self.rel_id_to_idx.keys():
                    '''check oov'''
                    continue
                if tuple(cand_chain['relations']) not in self.rel_to_word.keys():
                    rel = tuple(cand_chain['relations'])
                    rel_words = self.rel_exe.relation_list_word_seq(rel, domain=False)
                    if len(rel_words) == 0:  # special case for no relation words (only stopwords)
                        self.rel_to_word[rel] = [
                            self.tokenTOidx(self.rel_word_to_idx, "<REL_WORD_UNKNOWN>", "<REL_WORD_UNKNOWN>")]
                    else:
                        self.rel_to_word[rel] = [self.tokenTOidx(self.rel_word_to_idx, i, "<REL_WORD_UNKNOWN>")
                                                 for i in rel_words]
                    if rel not in self.rel_to_id.keys():
                        self.rel_to_id[rel] = self.tokenTOidx(self.rel_id_to_idx, rel)
                curr_example = {
                    'parent_index': int(cand_chain['parent_index']),
                    'index': idx,
                    'qid': qid,
                    'start entity': start_entity,
                    'topic entity': cand_chain['topic_entity'],
                    'relation': tuple(cand_chain['relations']),
                }
                if self.use_constraint:
                    constraints = []
                    if 'constraints' in cand_chain.keys():
                        for cons_dict in cand_chain['constraints']:
                            if cons_dict["relation"] not in self.constraint_id_to_idx.keys():
                                continue
                            constraints.append(cons_dict["relation"])
                    curr_example['constraints'] = tuple(constraints)
                    idx += 1
                testing_examples.append(curr_example)
        # construct data dict
        unpacked_q_word_seqs, unpacked_q_dep_seqs, unpacked_rel_words, rel_ids, unpacked_constraint_words, \
        unpacked_constraint_ids, labels = [], [], [], [], [], [], []
        padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, \
        batch_rel_ids, padded_cons_words, cons_word_lengths, padded_cons_id, cons_id_lengths \
            = None, None, None, None, None, None, None, None, None, None, None
        indices = []
        parent_indices = []
        for example in testing_examples:
            if self.use_constraint:
                self.csv_curr.writerow([example['parent_index'], example['index'], qid, example['start entity'],
                                        example['relation'], example['constraints']])
            else:
                self.csv_curr.writerow([example['parent_index'], example['index'], qid, example['start entity'],
                                        example['relation']])
            if type(example) is str:
                raise TypeError('Example Type Err')
            if example['topic entity'] not in self.qid_to_word_seq[example['qid']].keys():
                print "topic entity: {}".format(example['topic entity'])
                print self.qid_to_word_seq[example['qid']].keys()
                continue
            indices.append(example['index'])
            parent_indices.append(example['parent_index'])
            unpacked_q_word_seqs.append(obj_to_tensor(self.qid_to_word_seq[example['qid']][example['topic entity']]))
            unpacked_q_dep_seqs.append(obj_to_tensor(self.qid_to_dep_seq[example['qid']][example['topic entity']]))
            unpacked_rel_words.append(obj_to_tensor(self.rel_to_word[example['relation']]))
            rel_ids.append(obj_to_tensor(self.rel_to_id[example['relation']]))
            if self.use_constraint:
                constraint_rel_tensor, constraint_id_tensor = None, None
                if len(example['constraints']) == 0:
                    '''no constraints'''
                    constraint_rel_tensor = obj_to_tensor([[0] * self.max_constraint_word] * self.max_constraint_num)
                    constraint_id_tensor = obj_to_tensor([[0]] * self.max_constraint_num)
                    unpacked_constraint_words.append(constraint_rel_tensor)
                    unpacked_constraint_ids.append(constraint_id_tensor)
                else:
                    constraint_rel_tensor = []
                    constraint_id_tensor = []
                    for cons in example['constraints']:
                        constraint_rel_tensor.append(self.constraint_to_word[cons])
                        constraint_id_tensor.append([self.constraint_to_id[cons]])
                    cons_added = len(example['constraints'])
                    while cons_added < self.max_constraint_num:
                        constraint_rel_tensor.append([0] * self.max_constraint_word)
                        constraint_id_tensor.append([0])
                        cons_added += 1
                    unpacked_constraint_words.append(cuda_wrapper(torch.tensor(constraint_rel_tensor)))
                    unpacked_constraint_ids.append(cuda_wrapper(torch.tensor(constraint_id_tensor)))
            else:
                unpacked_constraint_words.append(torch.Tensor(0))  # placeholder
                unpacked_constraint_ids.append(torch.Tensor(0))  # placeholder

        if len(unpacked_q_word_seqs) == 0:
            return None

        # pad sequence
        padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, \
        batch_rel_ids, padded_cons_words, cons_word_lengths, padded_cons_id, cons_id_lengths, batch_indices, batch_parent_indices = \
            self.prepare_batch_data(unpacked_q_word_seqs, unpacked_q_dep_seqs, unpacked_rel_words, rel_ids,
                                    unpacked_constraint_words, unpacked_constraint_ids, indices, parent_indices)

        return {
            'indices': batch_indices,
            'parent_indices': batch_parent_indices,
            'padded_q_word': padded_q_word,
            'q_word_lengths': q_word_lengths,
            'padded_q_dep': padded_q_dep,
            'q_dep_lengths': q_dep_lengths,
            'padded_rel_words': padded_rel_words,
            'rel_word_lengths': rel_word_lengths,
            'rel_ids': batch_rel_ids,
            'constraint_word': padded_cons_words,
            'constraint_word_lengths': cons_word_lengths,
            'constraint_ids': padded_cons_id,
            'constraint_id_lengths': cons_id_lengths}

    def pad_sequence(self, batch, remove_reference=True):
        '''
        :param batch: with reference to do padding
        '''
        sorted_indices = sorted(range(len(batch)), key=lambda x: batch[x].shape[0], reverse=True)
        sequences = [batch[i] for i in sorted_indices]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        lengths = np.array([len(x) for x in sequences])
        if remove_reference:
            return sequences_padded[1:], lengths[1:], sorted_indices
        return sequences_padded, lengths, sorted_indices

    def prepare_batch_data(self, unpacked_q_word_seqs, unpacked_q_dep_seqs, unpacked_rel_words, rel_ids,
                           unpacked_constraint_words, unpacked_constraint_ids, indices=None, parent_indices=None):
        unpacked_q_word_seqs.append(self.max_qword_ref)
        unpacked_q_dep_seqs.append(self.max_dep_ref)
        padded_q_word, q_word_lengths, _ = self.pad_sequence(unpacked_q_word_seqs, True)
        padded_q_dep, q_dep_lengths, _ = self.pad_sequence(unpacked_q_dep_seqs, True)
        padded_rel_words, rel_word_lengths, sorted_indices = self.pad_sequence(unpacked_rel_words, False)
        # batch_rel_ids = obj_to_tensor(rel_ids)
        batch_rel_ids = cuda_wrapper(torch.zeros(len(sorted_indices)))
        batch_parent_indices = cuda_wrapper(torch.zeros(len(sorted_indices)))
        batch_indices = cuda_wrapper(torch.zeros(len(sorted_indices)))
        for new_idx in range(len(sorted_indices)):
            batch_rel_ids[new_idx] = rel_ids[sorted_indices[new_idx]]
        if indices is not None:
            for new_idx in range(len(sorted_indices)):
                batch_indices[new_idx] = indices[sorted_indices[new_idx]]
        if parent_indices is not None:
            for new_idx in range(len(sorted_indices)):
                batch_parent_indices[new_idx] = parent_indices[sorted_indices[new_idx]]
        padded_constraint_words, cons_word_lengths = None, None # placeholder
        padded_constraint_ids, cons_id_lengths = None, None # placeholder
        if self.use_constraint:
            '''Note: currently it is fake lengths, just for placeholder usage'''
            cons_word_lengths, cons_id_lengths = [3] * len(sorted_indices), [3] * len(sorted_indices)
            padded_constraint_words = cuda_wrapper(torch.zeros(len(sorted_indices), self.max_constraint_num, self.max_constraint_word))
            padded_constraint_ids = cuda_wrapper(torch.zeros(len(sorted_indices), self.max_constraint_num, 1))
            for new_idx in range(len(sorted_indices)):
                padded_constraint_words[new_idx] = unpacked_constraint_words[sorted_indices[new_idx]][:self.max_constraint_num]
                padded_constraint_ids[new_idx] = unpacked_constraint_ids[sorted_indices[new_idx]][:self.max_constraint_num]

        return padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, \
               batch_rel_ids, padded_constraint_words, cons_word_lengths, padded_constraint_ids, cons_id_lengths, \
               batch_indices, batch_parent_indices

    def get_max_qdep_len(self):
        return self.max_dep_len

    def get_max_qw_len(self):
        return self.max_q_word_len

    def open_csv_file(self, path):
        self.curr_fp = open(path, 'w+')
        self.csv_curr = csv.writer(self.curr_fp)
        if self.use_constraint:
            self.csv_curr.writerow(['parent_index', 'index', 'qid', 'topic', 'relations', 'constraints'])
        else:
            self.csv_curr.writerow(['parent_index', 'index', 'qid', 'topic', 'relations'])

    def close_csv_file(self):
        self.curr_fp.close()
        self.csv_curr = None


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    #dataset_path = '/public/ComplexWebQuestions_Resources/intermediate_data/rescaled_max_priors_derived_0.5_train_processed.json'
    #vocab_prefix = '/public/ComplexWebQuestions_Resources/vocab'
    dataset_path = '/z/zxycarol/ComplexWebQuestions_Resources/intermediate_data/dev_openie.json'
    vocab_prefix = '/z/zxycarol/ComplexWebQuestions_Resources/vocab'
    q_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'question_word_to_idx_2.pickle'), 'rb'))
    q_dep_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'question_dep_to_idx_2.pickle'), 'rb'))
    rel_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'rel_word_to_idx.pickle'), 'rb'))
    rel_id_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'rel_id_to_idx_word.pickle'), 'rb'))
    constraint_id_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_id_to_idx.pickle'), 'rb'))
    constraint_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_word_to_idx.pickle'), 'rb'))
    sys.stderr.write("Start constructing data loader...\n")
    data_loader = ComplexWebSP_Test_Data(
        dataset_path=dataset_path, q_word_to_idx=q_word_to_idx, q_dep_to_idx=q_dep_to_idx, rel_word_to_idx=rel_word_to_idx,
        rel_id_to_idx=rel_id_to_idx, use_constraint=False, max_constraint_word=10, max_constraint_num=4, cache_dir='/z/zxycarol/ComplexWebQuestions_Resources/cache',
        constraint_word_to_idx=constraint_word_to_idx, constraint_id_to_idx=constraint_id_to_idx,
        use_cache=False
    )
    for qid in data_loader.qid_to_training_tensors_per_question.keys():
        data_dict = data_loader.qid_to_training_tensors_per_question[qid]
    # shuffled_indices = random.sample(range(len(data_loader)), len(data_loader))
    # random.shuffle(shuffled_indices)
    # for idx in shuffled_indices:
    #     data_loader.get_one_batch(idx)
    # #
    # shuffled_indices = random.sample(range(len(data_loader)), len(data_loader))
    # random.shuffle(shuffled_indices)
    # for curr_index in shuffled_indices:
    #     data_dict = data_loader.get_one_batch(curr_index)
