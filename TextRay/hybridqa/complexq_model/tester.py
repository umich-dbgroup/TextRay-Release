import torch
import cPickle
import json
from torch.utils.data.dataset import Dataset
from nn_Modules.cuda import cuda_wrapper, obj_to_tensor
import numpy as np
import os
from nn_Modules.model import ComplexWebQSP_Model
from data_loader import ComplexWebSP_Test_Data
import subprocess
import sys
import torch.nn.functional as F

sys.path.insert(0, '/home/zxycarol/TextRay_Root/TextRay/hybridqa')
from preprocessing.complexq.tester import Tester_Interface


class Tester:
    def __init__(self, dataset_loader, q_word_to_idx, q_dep_to_idx, rel_word_to_idx, rel_id_to_idx, constraint_word_to_idx,
                 constraint_id_to_idx, q_word_emb, q_dep_emb, rel_word_emb, rel_id_emb, constraint_word_emb,
                 constraint_id_emb, device=0, test_batch_size=128, use_constraint=True, use_attn=True):
        torch.backends.cudnn.enabled = False
        torch.cuda.set_device(device)
        print "device: {}".format(torch.cuda.current_device())
        self.reward_threshold = 0.5
        self.test_batch_size = test_batch_size
        self.q_word_to_idx = q_word_to_idx
        self.q_dep_to_idx = q_dep_to_idx
        self.rel_word_to_idx = rel_word_to_idx
        self.rel_id_to_idx = rel_id_to_idx
        self.constraint_word_to_idx = constraint_word_to_idx
        self.constraint_id_to_idx = constraint_id_to_idx
        self.q_word_emb = q_word_emb
        self.q_dep_emb = q_dep_emb
        self.rel_word_emb = rel_word_emb
        self.rel_id_emb = rel_id_emb
        self.constraint_word_emb = constraint_word_emb
        self.constraint_id_emb = constraint_id_emb
        self.rel_idx_to_id = {}
        self.rel_idx_to_word = {}
        self.word_idx_to_q_word = {}
        self.constraint_idx_to_id = {}
        self.use_constraint = use_constraint
        self.use_attn = use_attn
        self.data_loader = dataset_loader
        for key in self.rel_id_to_idx.keys():
            val = self.rel_id_to_idx[key]
            self.rel_idx_to_id[val] = key
        for key in self.q_word_to_idx.keys():
            val = self.q_word_to_idx[key]
            self.word_idx_to_q_word[val] = key
        if self.use_constraint:
            for key in self.constraint_id_to_idx.keys():
                val = self.constraint_id_to_idx[key]
                self.constraint_idx_to_id[val] = key
        for key in self.rel_word_to_idx.keys():
            val = self.rel_word_to_idx[key]
            self.rel_idx_to_word[val] = key

        self.model = cuda_wrapper(ComplexWebQSP_Model(
            q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb,
            rel_id_emb=self.rel_id_emb, use_constraint=self.use_constraint,
            constraint_id_emb=self.constraint_id_emb, constraint_word_emb=self.constraint_word_emb, use_attn=self.use_attn
        ))
        sys.stderr.write("Testing model configuration:\n q_word_emb: {}, q_dep_emb: {}\n".format(
            self.model.q_encoder.q_word_emb.weight.shape, self.model.q_encoder.q_dep_emb.weight.shape))


    def sub1_prediction(self, model, data_dict, top_K=5):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, batch_rel_ids, \
        padded_cons_word, cons_word_lengths, padded_cons_id, cons_id_lengths \
            = data_dict['padded_q_word'], data_dict['q_word_lengths'], \
              data_dict['padded_q_dep'], data_dict['q_dep_lengths'], \
              data_dict['padded_rel_words'], data_dict['rel_word_lengths'], data_dict['rel_ids'], \
              data_dict['constraint_word'], data_dict['constraint_word_lengths'], \
              data_dict['constraint_ids'], data_dict['constraint_id_lengths']
        indices = data_dict['indices']
        out = model(padded_q_word_seq=padded_q_word, q_word_lengths=q_word_lengths,
                    padded_q_dep_seq=padded_q_dep, q_dep_lengths=q_dep_lengths,
                    padded_rel_words_seq=padded_rel_words, rel_word_lengths=rel_word_lengths,
                    padded_constraint_words_seq=padded_cons_word, constraint_word_lengths=cons_word_lengths,
                    padded_constraint_ids=padded_cons_id, constraint_id_lengths=cons_id_lengths,
                    rel_ids=batch_rel_ids, pooling=self.use_constraint)
        softmax_output = F.softmax(out)
        out_scores = softmax_output[:, 1]
        sorted_scores, sorted_idx = out_scores.sort(descending=True)
        top_k_result = []
        for number_idx in range(min(top_K, sorted_idx.shape[0])):
            if self.use_constraint:
                constraints = []
                # for cons_idx in range(padded_cons_id[sorted_idx[number_idx]][:cons_id_lengths[sorted_idx[number_idx]]].shape[0]):
                #     constraints.append(self.constraint_idx_to_id[int(padded_cons_id[sorted_idx[number_idx]][cons_idx].squeeze(dim=0))])
                top_k_result.append({
                    "index": indices[sorted_idx[number_idx]].item(),
                    "sub1_score": sorted_scores[number_idx].item(),
                    "sub1_relation": self.rel_idx_to_id[batch_rel_ids[sorted_idx[number_idx]].item()],
                    # "sub1_constraint": constraints
                })
            else:
                top_k_result.append({
                    "index": indices[sorted_idx[number_idx]].item(),
                    "sub1_score": sorted_scores[number_idx].item(),
                    "sub1_relation": self.rel_idx_to_id[batch_rel_ids[sorted_idx[number_idx]].item()],
                })
        return top_k_result

    def sub2_prediction(self, model, data_dict, top_K=5):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        padded_q_word, q_word_lengths, padded_q_dep, q_dep_lengths, padded_rel_words, rel_word_lengths, batch_rel_ids, \
        padded_cons_word, cons_word_lengths, padded_cons_id, cons_id_lengths \
            = data_dict['padded_q_word'], data_dict['q_word_lengths'], \
              data_dict['padded_q_dep'], data_dict['q_dep_lengths'], \
              data_dict['padded_rel_words'], data_dict['rel_word_lengths'], data_dict['rel_ids'], \
              data_dict['constraint_word'], data_dict['constraint_word_lengths'], \
              data_dict['constraint_ids'], data_dict['constraint_id_lengths']
        parent_indices = data_dict['parent_indices']
        indices = data_dict['indices']
        out = model(padded_q_word_seq=padded_q_word, q_word_lengths=q_word_lengths,
                    padded_q_dep_seq=padded_q_dep, q_dep_lengths=q_dep_lengths,
                    padded_rel_words_seq=padded_rel_words, rel_word_lengths=rel_word_lengths,
                    padded_constraint_words_seq=padded_cons_word, constraint_word_lengths=cons_word_lengths,
                    padded_constraint_ids=padded_cons_id, constraint_id_lengths=cons_id_lengths,
                    rel_ids=batch_rel_ids, pooling=self.use_constraint)
        softmax_scores = F.softmax(out)
        out_scores = softmax_scores[:, 1]
        sorted_scores, sorted_idx = out_scores.sort(descending=True)
        top_k_result = []
        for number_idx in range(sorted_idx.shape[0]):
            if self.use_constraint:
                constraints = []
                # for cons_idx in range(padded_cons_id[sorted_idx[number_idx]][:cons_id_lengths[sorted_idx[number_idx]]].shape[0]):
                #     constraints.append(self.constraint_idx_to_id[int(padded_cons_id[sorted_idx[number_idx]][cons_idx].squeeze(dim=0))])
                top_k_result.append({
                    "parent_index": parent_indices[sorted_idx[number_idx]].item(),
                    "index": indices[sorted_idx[number_idx]].item(),
                    "sub2_score": sorted_scores[number_idx].item(),
                    "sub2_relation": self.rel_idx_to_id[batch_rel_ids[sorted_idx[number_idx]].item()],
                    # "sub2_constraint": constraints
                })
            else:
                top_k_result.append({
                    "parent_index": parent_indices[sorted_idx[number_idx]].item(),
                    "index": indices[sorted_idx[number_idx]].item(),
                    "sub2_score": sorted_scores[number_idx].item(),
                    "sub2_relation": self.rel_idx_to_id[batch_rel_ids[sorted_idx[number_idx]].item()],
                })
        #print top_k_result
        return top_k_result


    def sub2_data_gen(self, tester_inferencer, sub1_result, output_path, qid):
        '''
        :param sub1_result: see return result in sub1_one_prediction,
        :param output_path: output prefix for written results
        :param qid: qid of questions, in case you need it
        '''
        # make output directory
        # if not os.path.exists(output_path):
        #     if subprocess.call('mkdir ' + output_path, shell=True) != 0:
        #         raise OSError("Cannot mkdir " + output_path)
        #     else:
        #         print "Successful mkdir " + output_path
        #sys.stderr.write("Start gen sub2 data for {} in {}\n".format(qid, output_path))
        tester_inferencer.generate_sub2_candidates(sub1_result, qid, output_path)

    def viz_model_prediction(self, model, model_name, running_output, sub1_path, sub2_path, test_inferencer, top_K=5, rough_estimate=False):
        model.eval()

        prediction_output_path = os.path.join(running_output, model_name + '_prediction.csv')
        # sub1 evaluation
        if not os.path.exists(running_output):
            if subprocess.call('mkdir ' + running_output, shell=True) != 0:
                raise OSError('Cannot mkdir ' + running_output)
            else:
                print "Successful mkdir " + running_output
        print "Finish writing sub1 look up file"
        for qid in self.data_loader.qid_to_training_tensors_per_question.keys():
            data_dict = self.data_loader.qid_to_training_tensors_per_question[qid]
            sub1_result = self.sub1_prediction(model, data_dict, top_K)
            json.dump(sub1_result, open(os.path.join(running_output, 'sub1_pred_{}.json'.format(qid)), 'w+'))
            if os.path.exists(os.path.join(running_output, qid + "_sub2_data.json")):
                # if qid == 'WebQTest-1923_e11655219d44e3762e0510f2bde1c077':
                #     self.sub2_data_gen(test_inferencer, sub1_result,
                #                        os.path.join(running_output, qid + "_sub2_data.json"), qid)
                #     pass
                continue
            self.sub2_data_gen(test_inferencer, sub1_result, os.path.join(running_output, qid + "_sub2_data.json"), qid)


        # sub2 evaluation
        qid_to_sub2_res = {}
        if os.path.exists(os.path.join(running_output, 'qid_to_sub2_res.json')):
            qid_to_sub2_res = json.load(open(os.path.join(running_output, 'qid_to_sub2_res.json')))
        else:
            self.data_loader.open_csv_file(sub2_path)
            for qid in self.data_loader.qid_to_training_tensors_per_question.keys():
                if not os.path.exists(os.path.join(running_output, qid + "_sub2_data.json")):
                    continue
                #print "Checking qid: {}".format(qid)
                sub2_candidate_paths = json.load(open(os.path.join(running_output, qid + "_sub2_data.json")))
                sub2_data_dict = self.data_loader.construct_data_dict_from_test_json(qid, sub2_candidate_paths)
                if sub2_data_dict is None:
                    print 'sub2 data dict empty: {}'.format(qid)
                    continue
                sub2_result = self.sub2_prediction(model, sub2_data_dict)
                json.dump(sub2_result, open(os.path.join(running_output, 'sub2_pred_{}.json'.format(qid)), 'w+'))
                qid_to_sub2_res[qid] = sub2_result
            self.data_loader.close_csv_file()
            print "qid_to_sub2_res: {}".format(len(qid_to_sub2_res))
            json.dump(qid_to_sub2_res, open(os.path.join(running_output, 'qid_to_sub2_res.json'), 'w+'))
        if not rough_estimate:
            print 'sub2 path: {}'.format(sub2_path)
            print 'running output: {}'.format(running_output)
            print 'prediction output path: {}'.format(prediction_output_path)
            test_inferencer.evaluate(sub2_path, qid_to_sub2_res, running_output, prediction_output_path)

    def inference_only(self, model, model_name, running_output, sub1_path, sub2_path, test_inferencer, top_K=5):
        qid_to_sub2_res = json.load(open(os.path.join(running_output, 'qid_to_sub2_res.json'), 'r'))
        prediction_output_path = os.path.join(running_output, model_name + '_prediction.csv')
        test_inferencer.evaluate(sub2_path, qid_to_sub2_res, running_output, prediction_output_path)



if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    # sub2_path = '/z/zxycarol/ComplexWebQuestions_Resources/0228_vanilla_test/TH0.5_OPTadam_LR0.0005_GA0.1_ATTN500_DO0.0/sub2_lookup.csv'
    # qid_to_sub2_res = json.load('/z/zxycarol/ComplexWebQuestions_Resources/0228_vanilla_test/TH0.5_OPTadam_LR0.0005_GA0.1_ATTN500_DO0.0/qid_to_sub2_res.json')
    # running_output = '/z/zxycarol/ComplexWebQuestions_Resources/0228_vanilla_test/TH0.5_OPTadam_LR0.0005_GA0.1_ATTN500_DO0.0'
    # prediction_output_path = '/z/zxycarol/ComplexWebQuestions_Resources/0228_vanilla_test/TH0.5_OPTadam_LR0.0005_GA0.1_ATTN500_DO0.0/tmp.csv'
    # #test_inferencer = Tester_Interface()
    # test_inferencer_evaluate(sub2_path, qid_to_sub2_res, running_output, prediction_output_path)


    dataset_path = '/public/ComplexWebQuestions_Resources/intermediate_data/test_sub1.json'
    vocab_prefix = '/public/ComplexWebQuestions_Resources/vocab'
    q_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'question_word_to_idx_2.pickle'), 'rb'))
    q_dep_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'question_dep_to_idx_2.pickle'), 'rb'))
    rel_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'rel_word_to_idx.pickle'), 'rb'))
    rel_id_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'rel_id_to_idx_word.pickle'), 'rb'))
    constraint_id_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_id_to_idx.pickle'), 'rb'))
    constraint_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_word_to_idx.pickle'), 'rb'))
    q_word_emb = cPickle.load(open(os.path.join(vocab_prefix, 'question_word_emb_tensor_2'), 'rb'))
    q_dep_emb = cPickle.load(open(os.path.join(vocab_prefix, 'question_dep_emb_tensor_2')))
    rel_word_emb = cPickle.load(open(os.path.join(vocab_prefix, 'rel_word_emb_word_tensor'), 'rb'))
    rel_id_emb = cPickle.load(open(os.path.join(vocab_prefix, 'rel_id_emb_word_tensor'), 'rb'))
    constraint_id_emb = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_id_emb_tensor'), 'rb'))
    constraint_word_emb = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_word_emb_tensor'), 'rb'))
    use_attn = True
    use_constraint = True
    running_output = '/public/ComplexWebQuestions_Resources/0418_nodep/test_results'
    ques_src = '/public/ComplexWebQuestions_Resources/complex_questions/data/annotated/test.json'
    sub1_cands_dir = '/public/ComplexWebQuestions_Resources/ComplexWebQuestions_preprocess/rewards/test_sub1'
    sub2_cands_dir = '/public/ComplexWebQuestions_Resources/ComplexWebQuestions_preprocess/rewards/test_sub2'
    sub1_path = os.path.join(running_output, 'sub1_lookup.csv')
    sub2_path = os.path.join(running_output, 'sub2_lookup.csv')
    use_dep = False
    data_loader = ComplexWebSP_Test_Data(
        dataset_path=dataset_path, q_word_to_idx=q_word_to_idx, q_dep_to_idx=q_dep_to_idx, rel_word_to_idx=rel_word_to_idx,
        rel_id_to_idx=rel_id_to_idx, use_constraint=True, max_constraint_word=7, max_constraint_num=3,
        constraint_word_to_idx=constraint_word_to_idx, constraint_id_to_idx=constraint_id_to_idx, use_dep=False, use_cache=False,
        cache_dir='', cache_prefix='')

    sys.stderr.write("Finish initializing data loader...\n")
    tester = Tester(dataset_loader=data_loader, q_word_to_idx=q_word_to_idx, q_dep_to_idx=q_dep_to_idx,
                    rel_word_to_idx=rel_word_to_idx, rel_id_to_idx=rel_id_to_idx, constraint_word_to_idx=constraint_word_to_idx,
                    constraint_id_to_idx=constraint_id_to_idx, q_word_emb=q_word_emb, q_dep_emb=q_dep_emb,
                    rel_word_emb=rel_word_emb, rel_id_emb=rel_id_emb, constraint_word_emb=constraint_word_emb,
                    constraint_id_emb=constraint_id_emb, use_attn=use_attn, use_constraint=use_constraint)
    tester.data_loader.write_sub1_file(sub1_path)
    tester_inferencer = Tester_Interface(ques_src=ques_src, sub1_flat_file_path=sub1_path, sub1_cands_dir=sub1_cands_dir,
                                         sub2_cands_dir=sub2_cands_dir)
    sys.stderr.write('Finish initializing tester and tester inferencer...\n')
    model_path = '/public/ComplexWebQuestions_Resources/0418_nodep/5'
    model_name = '5'
    model = cuda_wrapper(ComplexWebQSP_Model(
            q_word_emb=q_word_emb, q_dep_emb=q_dep_emb, rel_word_emb=rel_word_emb,
            rel_id_emb=rel_id_emb, use_constraint=use_constraint, use_attn=use_attn,
            constraint_id_emb=constraint_id_emb, constraint_word_emb=constraint_word_emb,
            max_seq_len=33, attn_hid_dim=500)
    )
    cp = torch.load(model_path)
    model.load_state_dict(cp['state_dict'])
    sys.stderr.write('Finish init model\n')
    tester.viz_model_prediction(model, model_name, running_output, sub1_path, sub2_path, tester_inferencer)
    sys.stderr.write("Start testing n_best f1 score for queries...\n")
    n_bests = [25, 10, 5, 1]
    for n_best in n_bests:
        print "n_best: {}".format(n_best)
        tester_inferencer.get_average_f1('/public/ComplexWebQuestions_Resources/0418_nodep/test_results/5_prediction.csv', n_best)










