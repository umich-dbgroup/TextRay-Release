import torch
import torch.nn as nn
from webqspModel.evaluator import Evaluator
from webqspModel.tester import Tester as WebQSP_Tester
from webqspModel.data_loader import WebQSP_Test_Data, WebQSP_Train_Data
from webqspModel.evaluator import Evaluator
import cPickle
import os
import subprocess
import sys
from complexq_model.nn_Modules.model import ComplexWebQSP_Model
from complexq_model.nn_Modules.cuda import cuda_wrapper


def print_model(model, f1s, max_k):
    res = ''
    for k in range(1, max_k):
        res += "top-{}: {}\n".format(k, f1s[model][k])
    return res

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    resource_prefix = '/z/zxycarol/ComplexWebQuestions_Resources'
    vocab_prefix = os.path.join(resource_prefix, 'vocab')
    checkpoint_path = '/z/zxycarol/ComplexWebQuestions_Resources/ablation_wo_prior/CONSTrue_OPTadam_LR0.0005_GA0.5_ATTNTrue500_DO0.0_PRFalse/5'
    checkpoint_prefix = '/z/zxycarol/ComplexWebQuestions_Resources/ablation_wo_prior/CONSTrue_OPTadam_LR0.0005_GA0.5_ATTNTrue500_DO0.0_PRFalse'
    device = 0
    torch.cuda.set_device(device)
    print "start loading vocab..."
    q_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'question_word_to_idx_2.pickle'), 'rb'))
    q_dep_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'question_dep_to_idx_2.pickle'), 'rb'))
    q_word_emb = cPickle.load(open(os.path.join(vocab_prefix, 'question_word_emb_tensor_2'), 'rb'))
    q_dep_emb = cPickle.load(open(os.path.join(vocab_prefix, 'question_dep_emb_tensor_2')))
    rel_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'rel_word_to_idx.pickle'), 'rb'))
    rel_id_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'rel_id_to_idx_word.pickle'), 'rb'))
    rel_word_emb = cPickle.load(open(os.path.join(vocab_prefix, 'rel_word_emb_word_tensor'), 'rb'))
    rel_id_emb = cPickle.load(open(os.path.join(vocab_prefix, 'rel_id_emb_word_tensor'), 'rb'))
    constraint_id_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_id_to_idx.pickle'), 'rb'))
    constraint_word_to_idx = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_word_to_idx.pickle'), 'rb'))
    constraint_word_emb = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_word_emb_tensor'), 'rb'))
    constraint_id_emb = cPickle.load(open(os.path.join(vocab_prefix, 'constraint_id_emb_tensor'), 'rb'))
    print "finish loading vocab..."


    print "start loading model..."
    model = cuda_wrapper(ComplexWebQSP_Model(
        q_word_emb=q_word_emb, q_dep_emb=q_dep_emb, rel_word_emb=rel_word_emb,
        rel_id_emb=rel_id_emb, use_constraint=True, use_attn=True, attn_hid_dim=500,
        constraint_id_emb=constraint_id_emb, constraint_word_emb=constraint_word_emb,
        max_seq_len=33, dropout=0
    ))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    print "finish loading model..."



    print "start constructing data loader..."
    test_dataset_path = '/z/zxycarol/WebQSP/complexwebq_modeleval/intermediate_data/processed_test.json'
    train_dataset_path = '/z/zxycarol/WebQSP/complexwebq_modeleval/intermediate_data/processed_train.json'
    test_dataset = WebQSP_Test_Data(dataset_path=test_dataset_path, q_word_to_idx=q_word_to_idx,
                                    q_dep_to_idx=q_dep_to_idx, rel_word_to_idx=rel_word_to_idx,
                                    rel_id_to_idx=rel_id_to_idx, use_constraint=True,
                                    constraint_word_to_idx=constraint_word_to_idx, constraint_id_to_idx=constraint_id_to_idx,
                                    max_constraint_word=10, max_constraint_num=4, use_entity_type=False)
    train_dataset = WebQSP_Test_Data(dataset_path=train_dataset_path, q_word_to_idx=q_word_to_idx,
                                     q_dep_to_idx=q_dep_to_idx, rel_word_to_idx=rel_word_to_idx,
                                     rel_id_to_idx=rel_id_to_idx, use_constraint=True,
                                     constraint_word_to_idx=constraint_word_to_idx, constraint_id_to_idx=constraint_id_to_idx,
                                     max_constraint_word=10, max_constraint_num=4, use_entity_type=False)
    print "finish constructing data loader..."

    print "start constructing tester"
    tester = WebQSP_Tester(dataset_loader=test_dataset, q_word_to_idx=q_word_to_idx, q_dep_to_idx=q_dep_to_idx,
                           rel_word_to_idx=rel_word_to_idx, rel_id_to_idx=rel_id_to_idx,
                           constraint_word_to_idx=constraint_word_to_idx,
                           constraint_id_to_idx=constraint_id_to_idx,
                           q_word_emb=q_word_emb, q_dep_emb=q_dep_emb,
                           rel_word_emb=rel_word_emb, rel_id_emb=rel_id_emb,
                           constraint_word_emb=constraint_word_emb,
                           constraint_id_emb=constraint_id_emb,
                           device=device, use_constraint=True, use_attn=True)
    train_tester = WebQSP_Tester(dataset_loader=train_dataset, q_word_to_idx=q_word_to_idx, q_dep_to_idx=q_dep_to_idx,
                                 rel_word_to_idx=rel_word_to_idx, rel_id_to_idx=rel_id_to_idx,
                                 constraint_word_to_idx=constraint_word_to_idx,
                                 constraint_id_to_idx=constraint_id_to_idx,
                                 q_word_emb=q_word_emb, q_dep_emb=q_dep_emb,
                                 rel_word_emb=rel_word_emb, rel_id_emb=rel_id_emb,
                                 constraint_word_emb=constraint_word_emb,
                                 constraint_id_emb=constraint_id_emb,
                                 device=device, use_constraint=True, use_attn=True)
    tester.model = model
    train_tester.model = model
    print "finish constructing tester..."

    test_save_dir = '/z/zxycarol/WebQSP/complexwebq_modeleval/wo_retrain_ablation/wo_prior'
    train_labeled_path = '/z/zxycarol/WebQSP/complexwebq_modeleval/cands_with_constraints_no_prior-scaled-train'
    test_labeled_path = '/z/zxycarol/WebQSP/complexwebq_modeleval/cands_with_constraints_no_prior-scaled-test'
    tester.test_dir(dir=checkpoint_prefix, result_prefix=test_save_dir, max_epoch=5, least_epoch=4)
    train_tester.test_dir(dir=checkpoint_prefix, result_prefix=test_save_dir, max_epoch=5, least_epoch=4)
    # evaluator = Evaluator()
    # model_names = ['E_1', 'E_2', 'E_3', 'E_4', 'E_5']
    # max_k = 10
    # test_src_path = '/z/zxycarol/WebQSP/data/WebQSP.train.json'
    # precs, f1s = evaluator.evaluate(test_save_dir, model_names, test_src_path,labeled_path,
    #                                 max_k, True)
    # print "-------------------------------------------------------------------------------"
    # print "---------------------------------Statistics------------------------------------"
    # print "-------------------------------------------------------------------------------"
    # for k in range(1, max_k):
    #     best_k = 0
    #     best_model = 0
    #     for model in model_names:
    #         if f1s[model][k] > best_k:
    #             best_k = f1s[model][k]
    #             best_model = model
    #
    #     print "Best model in top-{}: {}".format(k, best_model)
    #     print print_model(best_model, f1s, max_k)
    #     print "-------------------------------------------------------------------------------"





