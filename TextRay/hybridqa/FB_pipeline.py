import torch
import cPickle
import json
from complexq_model.nn_Modules.cuda import cuda_wrapper, obj_to_tensor
from complexq_model.nn_Modules.model import ComplexWebQSP_Model
from complexq_model.data_loader import ComplexWebSP_Test_Data, ComplexWebSP_Train_Data
from complexq_model.tester import Tester
from complexq_model.trainer import ComplexWebQSP_Trainer
import subprocess
import sys
from preprocessing.complexq.tester import Tester_Interface
from ConfigParser import SafeConfigParser
import os


class FB_pipeline:
    def __init__(self, config_path):
        '''
        hierarchy: all_output_prefix / beyond..
        In Train: train result prefix / model_config_name
        In Dev: Dev_Running_Prefix
        In Test: Test_Running_Prefix
        '''
        parser = SafeConfigParser()
        sys.stderr.write("Use parser: {}\n".format(config_path))
        parser.read(config_path)
        self.machine = parser.get('Constant', 'Machine')
        if self.machine == 'galaxy':
            torch.backends.cudnn.enabled = False
            self.resource_prefix = '/public/ComplexWebQuestions_Resources'
        elif self.machine == 'gelato':
            self.resource_prefix = '/z/zxycarol/ComplexWebQuestions_Resources'
        else:
            raise OSError('No such machine exists')

        self.cache_Dir = os.path.join(self.resource_prefix, 'cache')
        sys.stderr.write("Use Machine {}\n".format(self.machine))
        self.vocab_prefix = os.path.join(self.resource_prefix, parser.get('Constant', 'Vocab_Prefix'))
        '''runtime section'''
        self.use_cache = parser.getboolean('Runtime', 'Use_Cache')
        self.device = parser.getint('Runtime', 'Device')
        torch.cuda.set_device(self.device)
        self.do_train, self.do_dev, self.do_test = parser.getboolean('Runtime', 'Train'), \
                                                   parser.getboolean('Runtime', 'Dev'), \
                                                   parser.getboolean('Runtime', 'Test')
        sys.stderr.write('Train: {}, Dev: {}, Test: {}\n'.format(self.do_train, self.do_dev, self.do_test))
        '''train section'''
        self.train_batch_size = parser.getint('Train', 'Batch_Size')
        self.max_epoch = parser.getint('Train', 'Max_Epoch')
        self.reward_threshold = parser.getfloat('Train', 'Reward_Threshold')
        self.optimizer = parser.get('Train', 'Opitimizer').lower()
        self.learning_rate = parser.getfloat('Train', 'Learning_Rate')
        self.lr_gamma = parser.getfloat('Train', 'LR_Gamma')
        self.dropout = parser.getfloat('Train', 'Dropout')
        self.use_constraint = parser.getboolean('Train', 'Use_Constraint')
        self.use_attn = parser.getboolean('Train', 'Use_Attn')
        self.attn_dim = parser.getint('Train', 'Attn_Dim')
        self.always_pooling = parser.get('Train', 'Always_Pooling')
        self.pooling_threshold = parser.get('Train', 'Pooling_Threshold')
        self.use_entity_type = parser.getboolean('Train', 'Entity_Type')
        self.train_dataset_path = self.resource_prefix + parser.get('Train', 'Train_Dataset_Path')
        self.train_result_prefix = self.resource_prefix + parser.get('Train', 'Train_Result_Prefix')
        self.el_score = False
        self.prior_weights = False
        self.train_cache_prefix = ''

        try:
            self.el_score = parser.getboolean('Train', 'EL_Score')
        except:
            pass

        try:
            self.prior_weights = parser.getboolean('Train', 'Prior_Weights')
        except:
            pass

        try:
            self.train_cache_prefix = parser.get('Train', 'Train_Cache_Prefix')
        except:
            pass

        sys.stderr.write("Device: {}, Attn: {}, Entity Type: {}, EL Score: {}, Prior Weights: {}, Dropout: {}, LR: {}, LR_Gamma: {}\n".format(
            self.device, self.use_attn, self.use_entity_type, self.el_score, self.prior_weights, self.dropout, \
            self.learning_rate, self.lr_gamma))

        '''dev section'''
        self.dev_running_prefix = self.resource_prefix + parser.get('Dev', 'Dev_Running_Prefix')
        self.dev_dataset_path = self.resource_prefix + parser.get('Dev', 'Dev_Dataset_Path')
        self.dev_src_path = self.resource_prefix + parser.get('Dev', 'Dev_Src_Path')
        self.dev_sub1_cands_dir = self.resource_prefix + parser.get('Dev', 'Sub1_Cands_Dir')
        self.dev_sub2_cands_dir = self.resource_prefix + parser.get('Dev', 'Sub2_Cands_Dir')


        '''test section'''
        self.test_dataset_path = self.resource_prefix + parser.get('Test', 'Test_Dataset_Path')
        self.test_running_prefix = self.resource_prefix + parser.get('Test', 'Test_Running_Prefix')
        self.test_src_path = self.resource_prefix + parser.get('Test', 'Test_Src_Path')
        self.test_sub1_cands_dir = self.resource_prefix + parser.get('Test', 'Sub1_Cands_Dir')
        self.test_sub2_cands_dir = self.resource_prefix + parser.get('Test', 'Sub2_Cands_Dir')
        self.test_model_prefix = None
        self.test_rough_estimate = False

        try:
            self.test_model_prefix = parser.get('Test', 'Model_Prefix')
        except:
            sys.stderr.write('No model provided')

        try:
            self.test_rough_estimate = parser.getboolean('Test', 'Rough_Estimate')
        except:
            pass

        self.model_conf_name = "CONS{}_OPT{}_LR{}_GA{}_ATTN{}{}_DO{}_PR{}".format(self.use_constraint, self.optimizer,
                                                                         self.learning_rate, self.lr_gamma, self.use_attn,
                                                                         self.attn_dim, self.dropout, self.prior_weights)


        '''train model prefix'''
        if self.do_train:
            if not os.path.exists(self.train_result_prefix):
                if not subprocess.call('mkdir ' + self.train_result_prefix, shell=True):
                    raise OSError('cannot mkdir ' + self.train_result_prefix)
                else:
                    sys.stderr.write('Successful mkdir {}\n'.format(self.train_result_prefix))
            if not os.path.exists(os.path.join(self.train_result_prefix, self.model_conf_name)):
                if subprocess.call('mkdir ' + os.path.join(self.train_result_prefix, self.model_conf_name), shell=True):
                    raise OSError('cannot mkdir ' + os.path.join(self.train_result_prefix, self.model_conf_name))
                else:
                    sys.stderr.write('Successful mkdir {}\n'.format(os.path.join(self.train_result_prefix, self.model_conf_name)))

        '''dev and test running'''
        if self.do_dev:
            if not os.path.exists(self.dev_running_prefix):
                if subprocess.call('mkdir ' + self.dev_running_prefix, shell=True):
                    raise OSError('cannot mkdir ' + self.dev_running_prefix)
                else:
                    sys.stderr.write('Successful mkdir ' + self.dev_running_prefix)
            if not os.path.exists(os.path.join(self.dev_running_prefix, self.model_conf_name)):
                if subprocess.call('mkdir ' + os.path.join(self.dev_running_prefix, self.model_conf_name), shell=True):
                    raise OSError('cannot mkdir ' + os.path.join(self.dev_running_prefix, self.model_conf_name))
                else:
                    sys.stderr.write('Successful mkdir {}\n'.format(os.path.join(self.dev_running_prefix, self.model_conf_name)))

        if self.do_test:
            if not os.path.exists(self.test_running_prefix):
                if subprocess.call('mkdir ' + self.test_running_prefix, shell=True):
                    raise OSError('cannot mkdir ' + self.test_running_prefix)
                else:
                    sys.stderr.write('Successful mkdir {}\n'.format(self.test_running_prefix))
            else:
                sys.stderr.write("{} Exists\n".format(self.test_running_prefix))

            if not os.path.exists(os.path.join(self.test_running_prefix, self.model_conf_name)):
                if subprocess.call('mkdir ' + os.path.join(self.test_running_prefix, self.model_conf_name), shell=True):
                    raise OSError('cannot mkdir ' + os.path.join(self.test_running_prefix, self.model_conf_name))
                else:
                    sys.stderr.write('Successful mkdir {}\n'.format(os.path.join(self.test_running_prefix, self.model_conf_name)))
            else:
                sys.stderr.write("{} Exists\n".format(os.path.join(self.test_running_prefix, self.model_conf_name)))

        '''load resources'''
        self.q_word_to_idx = cPickle.load(open(os.path.join(self.vocab_prefix, 'question_word_to_idx_2.pickle'), 'rb'))
        self.q_dep_to_idx = cPickle.load(open(os.path.join(self.vocab_prefix, 'question_dep_to_idx_2.pickle'), 'rb'))
        self.q_word_emb = cPickle.load(open(os.path.join(self.vocab_prefix, 'question_word_emb_tensor_2'), 'rb'))
        self.q_dep_emb = cPickle.load(open(os.path.join(self.vocab_prefix, 'question_dep_emb_tensor_2')))
        self.rel_word_to_idx = cPickle.load(open(os.path.join(self.vocab_prefix, 'rel_word_to_idx.pickle'), 'rb'))
        self.rel_id_to_idx = cPickle.load(open(os.path.join(self.vocab_prefix, 'rel_id_to_idx_word.pickle'), 'rb'))
        self.rel_word_emb = cPickle.load(open(os.path.join(self.vocab_prefix, 'rel_word_emb_word_tensor'), 'rb'))
        self.rel_id_emb = cPickle.load(open(os.path.join(self.vocab_prefix, 'rel_id_emb_word_tensor'), 'rb'))
        self.constraint_id_to_idx, self.constraint_word_to_idx, self.constraint_word_emb, self.constraint_id_emb = None, \
        None, None, None

        if self.use_constraint:
            self.constraint_id_to_idx = cPickle.load(
                open(os.path.join(self.vocab_prefix, 'constraint_id_to_idx.pickle'), 'rb'))
            self.constraint_word_to_idx = cPickle.load(
                open(os.path.join(self.vocab_prefix, 'constraint_word_to_idx.pickle'), 'rb'))
            self.constraint_word_emb = cPickle.load(
                open(os.path.join(self.vocab_prefix, 'constraint_word_emb_tensor'), 'rb'))
            self.constraint_id_emb = cPickle.load(
                open(os.path.join(self.vocab_prefix, 'constraint_id_emb_tensor'), 'rb'))

        if self.do_train:
            self.train()

        if self.do_dev:
            self.dev()

        if self.do_test:
            self.test()

    def train(self):
        #sys.stderr.write(self.resource_prefix + '\n')
        sys.stderr.write("Train dataset path " + self.train_dataset_path + '\n')
        train_data_loader = ComplexWebSP_Train_Data(
            self.train_dataset_path, q_word_to_idx=self.q_word_to_idx, q_dep_to_idx=self.q_dep_to_idx,
            rel_word_to_idx=self.rel_word_to_idx,
            rel_id_to_idx=self.rel_id_to_idx, constraint_word_to_idx=self.constraint_word_to_idx,
            constraint_id_to_idx=self.constraint_id_to_idx, batch_size=self.train_batch_size,
            max_constraint_num=4, use_prior_weights=self.prior_weights,
            max_constraint_word=10, use_constraint=self.use_constraint, use_entity_type=self.use_entity_type,
            use_cache=self.use_cache, cache_dir=self.cache_Dir, use_el_score=self.el_score,
            cache_prefix=self.train_cache_prefix)
        sys.stderr.write('Finish Constructing Train dataset loader...\n')
        trainer = ComplexWebQSP_Trainer(
            dataset=train_data_loader, q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb,
            rel_id_emb=self.rel_id_emb, dropout=self.dropout, use_prior_weights=self.prior_weights,
            use_attn=self.use_attn, use_constraint=self.use_constraint, constraint_id_emb=self.constraint_id_emb,
            constraint_word_emb=self.constraint_word_emb, max_seq_len=train_data_loader.get_max_qw_len(),
            lr=self.learning_rate, lr_gamma=self.lr_gamma, pooling_threshold=self.pooling_threshold,
            attn_hid_dim=self.attn_dim, use_el_score=self.el_score, always_pooling=self.always_pooling,
            max_epoch=self.max_epoch)
        trainer.train(save_dir=os.path.join(self.train_result_prefix, self.model_conf_name))

    def dev(self):
        model = cuda_wrapper(ComplexWebQSP_Model(
            q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb,
            rel_id_emb=self.rel_id_emb, use_constraint=self.use_constraint, use_attn=self.use_attn,
            constraint_id_emb=self.constraint_id_emb, constraint_word_emb=self.constraint_word_emb,
            max_seq_len=33, attn_hid_dim=self.attn_dim, dropout=self.dropout
        ))
        dev_data_loader = ComplexWebSP_Test_Data(dataset_path=self.dev_dataset_path, q_word_to_idx=self.q_word_to_idx, q_dep_to_idx=self.q_dep_to_idx,
                                                 rel_word_to_idx=self.rel_word_to_idx,
                                                 rel_id_to_idx=self.rel_id_to_idx, use_constraint=True,
                                                 max_constraint_word=10, max_constraint_num=4,
                                                 constraint_word_to_idx=self.constraint_word_to_idx,
                                                 constraint_id_to_idx=self.constraint_id_to_idx,
                                                 reward_threshold=self.reward_threshold, use_entity_type=self.use_entity_type,
                                                 use_cache=True,
                                                 cache_dir=self.cache_Dir, cache_prefix='dev')
        dev_tester = Tester(dataset_loader=dev_data_loader, q_word_to_idx=self.q_word_to_idx, q_dep_to_idx=self.q_dep_to_idx,
                            rel_word_to_idx=self.rel_word_to_idx,
                            rel_id_to_idx=self.rel_id_to_idx, constraint_word_to_idx=self.constraint_word_to_idx,
                            constraint_id_to_idx=self.constraint_id_to_idx, q_word_emb=self.q_word_emb,
                            q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb, rel_id_emb=self.rel_id_emb,
                            constraint_word_emb=self.constraint_word_emb, constraint_id_emb=self.constraint_id_emb,
                            use_attn=self.use_attn, use_constraint=self.use_constraint)
        sys.stderr.write('Finish constructing dev data loader...\n')
        for root, dirs, files in os.walk(os.path.join(self.train_result_prefix, self.model_conf_name)):
            for file in files:
                epoch_num = file[0]
                if int(epoch_num) < 3:
                    continue

                if not os.path.exists(os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num)):
                    if subprocess.call('mkdir ' + os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num), shell=True):
                        raise OSError('Cannot mkdir ' + os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num))
                    else:
                        sys.stderr.write('Successful mkdir {}\n'.format(os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num)))

                if not os.path.exists(os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num, 'sub2_cands')):
                    if subprocess.call('mkdir ' + os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num, 'sub2_cands'), shell=True):
                        raise OSError('Cannot mkdir ' + os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num, 'sub2_cands'))
                    else:
                        sys.stderr.write('Successful mkdir {}\n'.format(os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num, 'sub2_cands')))
                # copy data
                model_sub2_cands_dir = os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num, 'sub2_cands')
                if subprocess.call('cp {}/*.json {}'.format(self.dev_sub2_cands_dir, model_sub2_cands_dir), shell=True):
                    raise OSError('Cannot copy data...\n')
                else:
                    sys.stderr.write('Successful copy data from dev cands2 dir\n')


                checkpoint = torch.load(os.path.join(root, file))
                model.load_state_dict(checkpoint['state_dict'])
                sys.stderr.write("Load state dict {}\n".format(os.path.join(root, file)))
                sub1_path = os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num, 'sub1_lookup.csv')
                sub2_path = os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num, 'sub2_lookup.csv')
                sys.stderr.write('sub1_path: {}, sub2_path: {}\n'.format(sub1_path, sub2_path))
                dev_tester.data_loader.write_sub1_file(sub1_path)
                dev_inferencer = Tester_Interface(ques_src=self.dev_src_path, sub1_flat_file_path=sub1_path,
                                           sub1_cands_dir=self.dev_sub1_cands_dir, sub2_cands_dir=model_sub2_cands_dir)
                dev_tester.viz_model_prediction(model=model, model_name=epoch_num, running_output=os.path.join(
                                                self.dev_running_prefix, self.model_conf_name, epoch_num), sub1_path=sub1_path, sub2_path=sub2_path,
                                                test_inferencer=dev_inferencer, top_K=5)

    def test(self):
        model = cuda_wrapper(ComplexWebQSP_Model(
            q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb,
            rel_id_emb=self.rel_id_emb, use_constraint=self.use_constraint, use_attn=self.use_attn,
            constraint_id_emb=self.constraint_id_emb, constraint_word_emb=self.constraint_word_emb,
            max_seq_len=33, attn_hid_dim=self.attn_dim, dropout=self.dropout
        ))

        sys.stderr.write("Loading from {}\n".format(self.test_model_prefix))
        valid_files = []
        for root, dirs, files in os.walk(os.path.join(self.train_result_prefix, self.model_conf_name)):
            sys.stderr.write('files: {}\n'.format(files))
            for file in files:
                if file.startswith(self.test_model_prefix):
                    valid_files.append(os.path.join(root, file))
        sys.stderr.write('Valid files: {}\n'.format(valid_files))
        if len(valid_files) == 0:
            assert False
        for idx in range(len(valid_files)):
            checkpoint = torch.load(valid_files[idx])
            if idx == 0:
                model.load_state_dict(checkpoint['state_dict'])
                sys.stderr.write("Load state dict {}\n".format(valid_files[idx]))
            else:
                break
        sub1_path = os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub1_lookup.csv')
        sub2_path = os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub2_lookup.csv')
        sys.stderr.write('sub1_path: {}, sub2_path: {}\n'.format(sub1_path, sub2_path))
        model_sub2_cands_dir = os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub2_cands')
        if not os.path.exists(os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix)):
            if subprocess.call('mkdir ' + os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix), shell=True):
                raise OSError('Cannot mkdir ' + os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix))
            else:
                sys.stderr.write('Successful mkdir {}\n'.format(os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix)))
        if not os.path.exists(os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub2_cands')):
            if subprocess.call(
                    'mkdir ' + os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub2_cands'),
                    shell=True):
                raise OSError('Cannot mkdir ' + os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix,
                                                             'sub2_cands'))
            else:
                sys.stderr.write('Successful mkdir {}\n'.format(
                    os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub2_cands')))
        if subprocess.call('cp {}/*.json {}'.format(self.test_sub2_cands_dir, model_sub2_cands_dir), shell=True):
            raise OSError('Cannot copy data...\n')
        else:
            sys.stderr.write('Successful copy data from test cands2 dir\n')

        sys.stderr.write("Test dataset Path: {}\n".format(self.test_dataset_path))
        test_data_loader = ComplexWebSP_Test_Data(
            dataset_path=self.test_dataset_path, q_word_to_idx=self.q_word_to_idx, q_dep_to_idx=self.q_dep_to_idx,
            rel_word_to_idx=self.rel_word_to_idx,
            rel_id_to_idx=self.rel_id_to_idx, use_constraint=self.use_constraint, max_constraint_word=10,
            max_constraint_num=4,
            constraint_word_to_idx=self.constraint_word_to_idx, constraint_id_to_idx=self.constraint_id_to_idx,
            reward_threshold=self.reward_threshold, use_entity_type=self.use_entity_type, use_cache=self.use_cache,
            cache_dir=self.cache_Dir, cache_prefix='test')
        sys.stderr.write('Finish Constructing test dataset loader..\n')
        tester = Tester(dataset_loader=test_data_loader, q_word_to_idx=self.q_word_to_idx, q_dep_to_idx=self.q_dep_to_idx,
                        rel_word_to_idx=self.rel_word_to_idx,
                        rel_id_to_idx=self.rel_id_to_idx, constraint_word_to_idx=self.constraint_word_to_idx,
                        constraint_id_to_idx=self.constraint_id_to_idx, q_word_emb=self.q_word_emb,
                        q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb, rel_id_emb=self.rel_id_emb,
                        constraint_word_emb=self.constraint_word_emb, constraint_id_emb=self.constraint_id_emb,
                        use_attn=self.use_attn, use_constraint=self.use_constraint, device=self.device)
        tester.data_loader.write_sub1_file(sub1_path)
        test_inferencer = Tester_Interface(ques_src=self.test_src_path, sub1_flat_file_path=sub1_path,
                                           sub1_cands_dir=self.test_sub1_cands_dir, sub2_cands_dir=model_sub2_cands_dir)


        print "Finish init model..."
        tester.viz_model_prediction(model=model, model_name=self.test_model_prefix, running_output=os.path.join(
            self.test_running_prefix, self.model_conf_name, self.test_model_prefix), sub1_path=sub1_path, sub2_path=sub2_path,
                                    test_inferencer=test_inferencer, top_K=5, rough_estimate=self.test_rough_estimate)
        if self.test_rough_estimate:
            print "Start Evaluating Sub2 quality..."
            test_inferencer.annotate_approx_sub2_labels(input_path=self.test_src_path, src_dir=os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix))
            test_inferencer.estimate_sub2_quality(ques_src=self.test_src_path, running_dir=os.path.join(self.test_running_prefix,
                                                                                                        self.model_conf_name, self.test_model_prefix))
        else:
            print "Start testing n_best f1 score for queries..."
            n_bests = [25, 10, 5, 1]
            for n_best in n_bests:
                print "n_best: {}".format(n_best)
                test_inferencer.get_average_f1(os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, self.test_model_prefix + '_prediction.csv'), n_best)


if __name__ == '__main__':
    config = None
    try:
        config = sys.argv[1]
    except:
        '''adapt to debugger'''
        config = 'FB_configs/ablation/wo_constraint_test.conf'
    pipeline_sys = FB_pipeline(config)

