import torch
import cPickle
from hybridq_model.nn_Modules.cuda import cuda_wrapper
from hybridq_model.nn_Modules.model import Hybrid_ComplexWebQSP_Model, Align_ComplexWebQSP_Model
from hybridq_model.hybrid_sep_dataloader import Hybrid_ComplexWebQ_Test_Data, Hybrid_ComplexWebq_Train_Data
from hybridq_model.align_dataloader import Align_ComplexWebq_Train_Data, Align_ComplexWebQ_Test_Data
from hybridq_model.tester import Hybrid_Tester
from hybridq_model.hybrid_trainer import Hybrid_Trainer
from hybridq_model.align_trainer import Align_Trainer
import sys
from preprocessing.complexq.hybrid_tester import Tester_Interface
from preprocessing.webqsp.hybrid_tester import WebQTestInterface
from ConfigParser import SafeConfigParser
import os
import subprocess


class Hybrid_pipeline:
    def __init__(self, config_path):
        parser = SafeConfigParser()
        sys.stderr.write("Use parser: {}\n".format(config_path))
        parser.read(config_path)

        # Constant
        self.dataset = 'complexwebq'
        try:
            self.dataset = parser.get('Constant', 'Dataset')
        except:
            pass
        assert self.dataset in ['webqsp', 'complexwebq']
        self.machine = parser.get('Constant', 'Machine')
        assert self.machine in {'galaxy', 'gelato'}
        if self.dataset == 'complexwebq':
            if self.machine == 'galaxy':
                self.resource_prefix = '/public/ComplexWebQuestions_Resources'
            else:
                self.resource_prefix = '/z/zxycarol/ComplexWebQuestions_Resources'
        else:
            if self.machine == 'galaxy':
                self.resource_prefix = '/public/WebQSP_Resources'
            else:
                self.resource_prefix = '/z/zxycarol/WebQSP_Resources'
        self.cache_dir = self.resource_prefix + parser.get('Constant', 'Cache_Dir')
        self.vocab_prefix = os.path.join(self.resource_prefix, parser.get('Constant', 'Vocab_Prefix'))
        self.eval_machine = 0
        try:
            self.eval_machine = parser.getint('Constant', 'Eval_Machine')
        except:
            pass


        # Runtime
        self.use_cache = parser.getboolean('Runtime', 'Use_Cache')
        self.save_cache = parser.getboolean('Runtime', 'Save_Cache')
        self.device = parser.getint('Runtime', 'Device')
        torch.cuda.set_device(self.device)
        self.do_train, self.do_dev, self.do_test = parser.getboolean('Runtime', 'Train'), \
                                                   parser.getboolean('Runtime', 'Dev'), \
                                                   parser.getboolean('Runtime', 'Test')
        sys.stderr.write('------------------Constant & Runtime Summary------------------\n')
        sys.stderr.write('Cache_Dir: {}\nVocab_Prefix: {}\n'.format(self.cache_dir, self.vocab_prefix))
        sys.stderr.write('Train: {}, Dev: {}, Test: {}\n'.format(self.do_train, self.do_dev, self.do_test))

        # Train
        self.train_cache_prefix = parser.get('Train', 'Train_Cache_Prefix')
        self.batch_size = parser.getint('Train', 'Batch_Size')
        self.max_epoch = parser.getint('Train', 'Max_Epoch')
        self.learning_rate = parser.getfloat('Train', 'Learning_Rate')
        self.lr_gamma = parser.getfloat('Train', 'LR_Gamma')
        self.attn_dropout = parser.getfloat('Train', 'Attn_Dropout')
        self.linear_dropout = parser.getfloat('Train', 'Linear_Dropout')
        self.use_constraint = parser.getboolean('Train', 'Use_Constraint')
        self.use_attn = parser.getboolean('Train', 'Use_Attn')
        self.use_kb = parser.getboolean('Train', 'Use_KB')
        self.use_openie = parser.getboolean('Train', 'Use_OpenIE')
        self.unified_openie = parser.getboolean('Train', 'Unified_OpenIE')
        self.always_pooling = parser.getboolean('Train', 'Always_Pooling')
        self.pooling_threshold = parser.getint('Train', 'Pooling_Threshold')
        self.train_result_prefix = self.resource_prefix + parser.get('Train', 'Train_Result_Prefix')
        self.train_dataset_path = self.resource_prefix + parser.get('Train', 'Train_Dataset_Path')
        self.mode_opt = 'hybrid'
        self.sample_strategy = 'up'
        try:
            self.model_opt = parser.get('Train', 'Model_OPT')
        except:
            pass
        assert self.model_opt in ['hybrid', 'align']
        try:
            self.sample_strategy = parser.get('Train', 'Sample_Strategy')
        except:
            pass
        assert self.sample_strategy in ['up', 'down', 'no']
        if self.model_opt == 'align':
            self.kb_openie_align = cPickle.load(open(os.path.join(self.vocab_prefix, 'qid_to_kb_to_openies.pickle'), 'rb'))
            assert self.use_kb
            assert self.cache_dir.endswith('align_cache')
        if self.sample_strategy == 'up':
            assert self.batch_size == 64
        elif self.sample_strategy == 'down':
            assert self.batch_size == 100

        self.model_conf_name = "MO{}_SA{}_CONS{}_ATTN{}_KB{}_OIE{}_UNIF{}_LR{}_LRG{}_ADO{}_LDO{}_APOOL{}_PTHR{}".format(
            self.model_opt, self.sample_strategy, self.use_constraint, self.use_attn, self.use_kb, self.use_openie,
            self.unified_openie, self.learning_rate, self.lr_gamma, self.attn_dropout, self.linear_dropout,
            self.always_pooling, self.pooling_threshold
        )


        # Dev
        self.dev_cache_prefix = parser.get('Dev', 'Dev_Cache_Prefix')
        self.dev_running_prefix = self.resource_prefix + parser.get('Dev', 'Dev_Running_Prefix')
        self.dev_dataset_path = self.resource_prefix + parser.get('Dev', 'Dev_Dataset_Path')
        self.dev_src_path = self.resource_prefix + parser.get('Dev', 'Dev_Src_Path')
        self.dev_sub1_cands_dir = self.resource_prefix + parser.get('Dev', 'Sub1_Cands_Dir')
        self.dev_sub2_cands_dir = self.resource_prefix + parser.get('Dev', 'Sub2_Cands_Dir')
        self.dev_openie_sub1_dir = self.resource_prefix + parser.get('Dev', 'Sub1_OpenIE_Dir')
        self.dev_openie_sub2_dir = self.resource_prefix + parser.get('Dev', 'Sub2_OpenIE_Dir')

        # Test
        self.test_cache_prefix = parser.get('Test', 'Test_Cache_Prefix')
        self.test_running_prefix = self.resource_prefix + parser.get('Test', 'Test_Running_Prefix')
        self.test_dataset_path = self.resource_prefix + parser.get('Test', 'Test_Dataset_Path')
        self.test_dataset_sub2_path = self.resource_prefix + parser.get('Test', 'Test_Sub2_Path')
        self.test_src_path = self.resource_prefix + parser.get('Test', 'Test_Src_Path')
        self.test_sub1_cands_dir = self.resource_prefix + parser.get('Test', 'Sub1_Cands_Dir')
        self.test_sub2_cands_dir = self.resource_prefix + parser.get('Test', 'Sub2_Cands_Dir')
        self.test_openie_sub1_dir = self.resource_prefix + parser.get('Test', 'Sub1_OpenIE_Dir')
        self.test_openie_sub2_dir = self.resource_prefix + parser.get('Test', 'Sub2_OpenIE_Dir')
        self.test_model_prefix = parser.get('Test', 'Model_Prefix')
        self.kb_prop = 1.0
        try:
            self.kb_prop = parser.getfloat('Test', 'KB_Prop')
            if self.kb_prop > 0:
                assert self.use_kb
        except:
            pass

        # make dirs
        if self.do_train:
            if not os.path.exists(self.train_result_prefix):
                os.mkdir(self.train_result_prefix)

            if not os.path.exists(os.path.join(self.train_result_prefix, self.model_conf_name)):
                os.mkdir(os.path.join(self.train_result_prefix, self.model_conf_name))

        if self.do_dev:
            if not os.path.exists(self.dev_running_prefix):
                os.mkdir(self.dev_running_prefix)

            if not os.path.exists(os.path.join(self.dev_running_prefix, self.model_conf_name)):
                os.mkdir(os.path.join(self.dev_running_prefix, self.model_conf_name))


        if self.do_test:
            if not os.path.exists(self.test_running_prefix):
                os.mkdir(self.test_running_prefix)

            if not os.path.exists(os.path.join(self.test_running_prefix, self.model_conf_name)):
                os.mkdir(os.path.join(self.test_running_prefix, self.model_conf_name))


        # load resources
        sys.stderr.write('Start loading resources...')
        self.q_word_to_idx = cPickle.load(open(os.path.join(self.vocab_prefix, 'question_word_to_idx_2.pickle'), 'rb'))
        self.q_dep_to_idx = cPickle.load(open(os.path.join(self.vocab_prefix, 'question_dep_to_idx_2.pickle'), 'rb'))
        self.q_word_emb = cPickle.load(open(os.path.join(self.vocab_prefix, 'question_word_emb_tensor_2'), 'rb'))
        self.q_dep_emb = cPickle.load(open(os.path.join(self.vocab_prefix, 'question_dep_emb_tensor_2')))
        self.rel_word_to_idx = cPickle.load(open(os.path.join(self.vocab_prefix, 'rel_word_to_idx.pickle'), 'rb'))
        self.rel_id_to_idx = cPickle.load(open(os.path.join(self.vocab_prefix, 'rel_id_to_idx_word.pickle'), 'rb'))
        self.rel_word_emb = cPickle.load(open(os.path.join(self.vocab_prefix, 'rel_word_emb_word_tensor'), 'rb'))
        self.rel_id_emb = cPickle.load(open(os.path.join(self.vocab_prefix, 'rel_id_emb_word_tensor'), 'rb'))
        if self.dataset == 'complexwebq':
            openie_suffix = '4'
        else:
            openie_suffix = '3'
        self.openie_word_to_idx = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_word_to_idx_{}.pickle'.format(openie_suffix)), 'rb'))
        self.openie_id_to_idx = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_id_to_idx.pickle'), 'rb'))
        self.openie_mapping = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_mapping.pickle'), 'rb'))
        self.openie_id_emb = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_id_emb_tensor'), 'rb'))
        self.openie_word_emb = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_word_emb_tensor_{}'.format(openie_suffix)), 'rb'))
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
        sys.stderr.write('Finished.\n')
        sys.stderr.write('------------------Vocab Size------------------\n')
        sys.stderr.write('qw: {}, qdep: {}, relw: {}, relid: {}, ow: {}, oid: {}, omap: {}\n'.format(
            len(self.q_word_to_idx), len(self.q_dep_to_idx), len(self.rel_word_to_idx), len(self.rel_id_to_idx),
            len(self.openie_word_to_idx), len(self.openie_id_to_idx), len(self.openie_mapping)
        ))


        if self.do_train:
            self.train()

        if self.do_dev:
            self.dev()

        if self.do_test:
            self.test()


    def train(self):
        openie_to_word, openie_to_id = None, None
        if os.path.exists(os.path.join(self.vocab_prefix, 'openie_to_word.pickle')) and \
                os.path.exists(os.path.join(self.vocab_prefix, 'openie_to_word.pickle')):
            sys.stderr.write('Reading OpenIE Rel dicts from cache...')
            openie_to_word = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_to_word.pickle'), 'rb'))
            openie_to_id = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_to_id.pickle'), 'rb'))
            sys.stderr.write('Finished.\n')
        sys.stderr.write("Train dataset path " + self.train_dataset_path + '\n')
        train_data_loader = None
        max_kb_align_num, max_openie_rel_word, max_kb_rel_word = 0, 0, 0
        if self.dataset == 'webqsp':
            max_kb_align_num, max_openie_rel_word, max_kb_rel_word = 7, 8, 8
        else:
            max_kb_align_num, max_openie_rel_word, max_kb_rel_word = 9, 13, 9
        if self.model_opt == 'hybrid':
            assert self.dataset == 'complexwebq'
            train_data_loader = \
                Hybrid_ComplexWebq_Train_Data(
                    dataset_path=self.train_dataset_path, q_word_to_idx=self.q_word_to_idx,
                    q_dep_to_idx=self.q_dep_to_idx, rel_word_to_idx=self.rel_word_to_idx,
                    rel_id_to_idx=self.rel_id_to_idx, openie_word_to_idx=self.openie_word_to_idx,
                    openie_id_to_idx=self.openie_id_to_idx, openie_mapping=self.openie_mapping,
                    openie_to_id=openie_to_id, openie_to_word=openie_to_word,
                    constraint_word_to_idx=self.constraint_word_to_idx, constraint_id_to_idx=self.constraint_id_to_idx,
                    use_constraint=self.use_constraint, use_openie=self.use_openie, unified_openie=self.unified_openie,
                    use_kb=self.use_kb, max_constraint_word=10, max_constraint_num=4, batch_size=self.batch_size,
                    use_cache=self.use_cache, save_cache=self.save_cache,
                    cache_prefix=self.train_cache_prefix, cache_dir=self.cache_dir, sample_strategy=self.sample_strategy)
        else:
            print "dataset {}".format(self.dataset)
            train_data_loader = Align_ComplexWebq_Train_Data(
                dataset_path=self.train_dataset_path, q_word_to_idx=self.q_word_to_idx,
                q_dep_to_idx=self.q_dep_to_idx, rel_word_to_idx=self.rel_word_to_idx,
                rel_id_to_idx=self.rel_id_to_idx, openie_mapping=self.openie_mapping,
                openie_word_to_idx=self.openie_word_to_idx, openie_id_to_idx=self.openie_id_to_idx,
                openie_to_id=openie_to_id, openie_to_word=openie_to_word,
                constraint_word_to_idx=self.constraint_word_to_idx, constraint_id_to_idx=self.constraint_id_to_idx,
                kb_openie_align=self.kb_openie_align, use_constraint=self.use_constraint, use_openie=self.use_openie,
                use_kb=self.use_kb, max_kb_rel_word=max_kb_rel_word, max_constraint_word=10, max_constraint_num=4,
                max_openie_rel_word=max_openie_rel_word, max_openie_align_num=max_kb_align_num ,batch_size=self.batch_size,
                cache_dir=self.cache_dir, use_cache=self.use_cache, unified_openie=self.unified_openie,
                save_cache=self.save_cache, cache_prefix=self.train_cache_prefix, sample_strategy=self.sample_strategy,
                dataset=self.dataset
            )

        sys.stderr.write('Finish Constructing Train dataset loader...\n')
        trainer = None
        if self.model_opt == 'hybrid':
            trainer = Hybrid_Trainer(
                dataset=train_data_loader, q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb,
                rel_word_emb=self.rel_word_emb, rel_id_emb=self.rel_id_emb, constraint_word_emb=self.constraint_word_emb,
                constraint_id_emb=self.constraint_id_emb, openie_id_emb=self.openie_id_emb,
                openie_word_emb=self.openie_word_emb, max_seq_len=train_data_loader.get_max_qw_len(),
                attn_dropout=self.attn_dropout, linear_dropout=self.linear_dropout, use_attn=self.use_attn,
                use_constraint=self.use_constraint, use_openie=self.use_openie, use_kb=self.use_kb,
                always_pooling=self.always_pooling, lr=self.learning_rate, lr_gamma=self.lr_gamma,
                pooling_threshold=self.pooling_threshold, max_epoch=self.max_epoch
            )
        else:
            trainer = Align_Trainer(
                dataset=train_data_loader, q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb,
                rel_word_emb=self.rel_word_emb, rel_id_emb=self.rel_id_emb, constraint_word_emb=self.constraint_word_emb,
                constraint_id_emb=self.constraint_id_emb, openie_word_emb=self.openie_word_emb,
                openie_id_emb=self.openie_id_emb, max_seq_len=train_data_loader.get_max_qw_len(),
                attn_dropout=self.attn_dropout, linear_dropout=self.linear_dropout, use_attn=self.use_attn,
                use_constraint=self.use_constraint, use_openie=self.use_openie, use_kb=self.use_kb,
                always_pooling=self.always_pooling, lr=self.learning_rate, lr_gamma=self.lr_gamma,
                pooling_threshold=self.pooling_threshold, max_epoch=self.max_epoch
            )
        trainer.train(save_dir=os.path.join(self.train_result_prefix, self.model_conf_name))


    def dev(self):
        if self.dataset == 'complexwebq':
            assert NotImplementedError()
        sys.stderr.write('\n------------------Start Testing------------------\n')
        model = None
        max_seq_len = 33 if self.dataset == 'complexwebq' else 14
        if self.model_opt == 'hybrid':
            model = cuda_wrapper(Hybrid_ComplexWebQSP_Model(
                q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb,
                rel_id_emb=self.rel_id_emb, constraint_word_emb=self.constraint_word_emb,
                constraint_id_emb=self.constraint_id_emb, openie_word_emb=self.openie_word_emb,
                openie_id_emb=self.openie_id_emb, use_openie=self.use_openie,
                use_attn=self.use_attn, use_constraint=self.use_constraint, use_kb=self.use_kb, max_seq_len=max_seq_len,
                attn_dropout=self.attn_dropout, linear_dropout=self.linear_dropout
            ))
        else:
            model = cuda_wrapper(Align_ComplexWebQSP_Model(
                q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb,
                rel_id_emb=self.rel_id_emb, openie_word_emb=self.openie_word_emb, openie_id_emb=self.openie_id_emb,
                use_openie=self.use_openie, use_kb=self.use_kb, use_attn=self.use_attn,
                use_constraint=self.use_constraint, constraint_word_emb=self.constraint_word_emb,
                constraint_id_emb=self.constraint_id_emb, word_emb_dim=300, id_emb_dim=300,
                lstm_hidden_dim=300, attn_q_dim=600, attn_rel_dim=600, attn_hid_dim=500, max_seq_len=max_seq_len,
                attn_dropout=self.attn_dropout, linear_dropout=self.linear_dropout, output_dim=2
            ))
        assert self.model_opt == 'align'
        # current dataset_sub2_path is dummy, because Dev is implemented for webqsp
        max_kb_align_num, max_openie_rel_word, max_kb_rel_word = 7, 8, 8
        openie_to_word, openie_to_id = None, None
        if os.path.exists(os.path.join(self.vocab_prefix, 'openie_to_word.pickle')) and \
                os.path.exists(os.path.join(self.vocab_prefix, 'openie_to_word.pickle')):
            sys.stderr.write('Reading OpenIE Rel dicts from cache...')
            openie_to_word = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_to_word.pickle'), 'rb'))
            openie_to_id = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_to_id.pickle'), 'rb'))
            sys.stderr.write('Finished.\n')
        dev_data_loader = Align_ComplexWebQ_Test_Data(
            dataset_path=self.dev_dataset_path,dataset_sub2_path=self.test_dataset_sub2_path,
            q_word_to_idx=self.q_word_to_idx, q_dep_to_idx=self.q_dep_to_idx,
            rel_word_to_idx=self.rel_word_to_idx, rel_id_to_idx=self.rel_id_to_idx, openie_mapping=self.openie_mapping,
            openie_word_to_idx=self.openie_word_to_idx, openie_id_to_idx=self.openie_id_to_idx,
            openie_to_word=openie_to_word, openie_to_id=openie_to_id, constraint_word_to_idx=self.constraint_word_to_idx,
            constraint_id_to_idx=self.constraint_id_to_idx, use_constraint=self.use_constraint,
            use_openie=self.use_openie, unified_openie=self.unified_openie, use_kb=self.use_kb, kb_prop=self.kb_prop,
            max_kb_rel_word=max_kb_rel_word, max_constraint_word=10, max_constraint_num=4, batch_size=self.batch_size,
            max_openie_rel_word=max_openie_rel_word, max_openie_align_num=max_kb_align_num,
            cache_dir=self.cache_dir, use_cache=self.use_cache, save_cache=self.save_cache,
            cache_prefix=self.dev_cache_prefix)
        sys.stderr.write('Start making dev tester..')
        dev_tester = Hybrid_Tester(
            data_loader=dev_data_loader, q_word_to_idx=self.q_word_to_idx, q_dep_to_idx=self.q_dep_to_idx,
            rel_word_to_idx=self.rel_word_to_idx, rel_id_to_idx=self.rel_id_to_idx,
            constraint_word_to_idx=self.constraint_word_to_idx, constraint_id_to_idx=self.constraint_id_to_idx,
            openie_word_to_idx=self.openie_word_to_idx, openie_id_to_idx=self.openie_id_to_idx,
            q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb,
            rel_id_emb=self.rel_id_emb, constraint_word_emb=self.constraint_word_emb,
            constraint_id_emb=self.constraint_id_emb, openie_word_emb=self.openie_word_emb,
            openie_id_emb=self.openie_id_emb, use_attn=self.use_attn, use_constraint=self.use_constraint,
            use_openie=self.use_openie, use_kb=self.use_kb, use_cache=self.use_cache, save_cache=self.save_cache,
            cache_dir=self.cache_dir, cache_prefix=self.test_cache_prefix, model_opt=self.model_opt,
            dataset=self.dataset)
        sys.stderr.write('Finished.\n')

        for root, dirs, files in os.walk(os.path.join(self.train_result_prefix, self.model_conf_name)):
            for file in files:
                epoch_num = file
                if int(epoch_num) < 3:
                    continue

                if not os.path.exists(os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num)):
                    os.mkdir(os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num))
                device = torch.device('cuda')
                checkpoint = torch.load(os.path.join(root, file), map_location=device)

                model.load_state_dict(checkpoint['state_dict'])
                sys.stderr.write("Load state dict {}\n".format(os.path.join(root, file)))
                sub1_path = os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num, 'sub1_lookup.csv')
                sub2_path = os.path.join(self.dev_running_prefix, self.model_conf_name, self.test_model_prefix,
                                         'sub2_lookup.csv')
                dev_tester.data_loader.write_sub1_file(sub1_path)
                top_K = 20 if self.dataset == 'complexwebq' else 100

                dev_tester.viz_model_prediction(
                    model=model, model_name=epoch_num, running_output=os.path.join(
                    self.dev_running_prefix, self.model_conf_name, epoch_num), sub1_path=sub1_path,
                    sub2_path=sub2_path, tester_inferencer=None, top_K=top_K)
                dev_tester_inferencer = WebQTestInterface(
                    ques_src=self.dev_src_path, lookup_path=sub1_path, kb_cands_dir=self.dev_sub1_cands_dir,
                    openie_cands_dir=self.dev_openie_sub1_dir
                )
                pred_dir = os.path.join(self.dev_running_prefix, self.model_conf_name, epoch_num)
                output_path = os.path.join(pred_dir, "final_results.csv")
                dev_tester_inferencer.evaluate(pred_dir, output_path)
                n_bests = [1, 5, 10, 25]
                print "Epoch: {}".format(epoch_num)
                for n_best in n_bests:
                    f1 = dev_tester_inferencer.get_average_f1(output_path, n_best)
                    print "{}-best: {}".format(n_best, f1)



    def test(self):
        sys.stderr.write('\n------------------Start Testing------------------\n')
        model = None
        max_seq_len = 33 if self.dataset == 'complexwebq' else 14
        if self.model_opt == 'hybrid':
            model = cuda_wrapper(Hybrid_ComplexWebQSP_Model(
                q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb,
                rel_id_emb=self.rel_id_emb, constraint_word_emb=self.constraint_word_emb,
                constraint_id_emb=self.constraint_id_emb, openie_word_emb=self.openie_word_emb,
                openie_id_emb=self.openie_id_emb, use_openie=self.use_openie,
                use_attn=self.use_attn, use_constraint=self.use_constraint, use_kb=self.use_kb, max_seq_len=max_seq_len,
                attn_dropout=self.attn_dropout, linear_dropout=self.linear_dropout
            ))
        else:
            model = cuda_wrapper(Align_ComplexWebQSP_Model(
                q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb,
                rel_id_emb=self.rel_id_emb, openie_word_emb=self.openie_word_emb, openie_id_emb=self.openie_id_emb,
                use_openie=self.use_openie, use_kb=self.use_kb, use_attn=self.use_attn,
                use_constraint=self.use_constraint, constraint_word_emb=self.constraint_word_emb,
                constraint_id_emb=self.constraint_id_emb, word_emb_dim=300, id_emb_dim=300,
                lstm_hidden_dim=300, attn_q_dim=600, attn_rel_dim=600, attn_hid_dim=500, max_seq_len=max_seq_len,
                attn_dropout=self.attn_dropout, linear_dropout=self.linear_dropout, output_dim=2
            ))
        model_loaded = False
        for root, dirs, files in os.walk(os.path.join(self.train_result_prefix, self.model_conf_name)):
            sys.stderr.write('files: {}\n'.format(files))
            for file in files:
                if file.startswith(self.test_model_prefix):
                    valid_file = os.path.join(root, file)
                    sys.stderr.write('Loading from {}...\n'.format(os.path.join(root, file)))
                    device = torch.device('cuda')
                    checkpoint = torch.load(valid_file, map_location=device)
                    model.load_state_dict(checkpoint['state_dict'])
                    sys.stderr.write('Finished.\n')
                    model_loaded = True
                    break
        assert model_loaded
        sub1_path = os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub1_lookup.csv')
        sub2_path = os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub2_lookup.csv')
        model_sub2_cands_dir = os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub2_cands')

        sys.stderr.write('Start preparing dirs and files...')
        if not os.path.exists(os.path.join(self.test_running_prefix, self.model_conf_name)):
            os.mkdir(os.path.join(self.test_running_prefix, self.model_conf_name))
        if not os.path.exists(os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix)):
            os.mkdir(os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix))
        if self.dataset == 'complexwebq':
            if not os.path.exists(os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub2_cands')):
                os.mkdir(os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix, 'sub2_cands'))

        # Copy file
        if self.dataset == 'complexwebq':
            if subprocess.call('cp {}/*.json {}'.format(self.test_sub2_cands_dir, model_sub2_cands_dir), shell=True):
                raise OSError('Cannot copy data...\n')
            sys.stderr.write('Finished.\n')

        openie_to_word, openie_to_id = None, None
        if os.path.exists(os.path.join(self.vocab_prefix, 'openie_to_word.pickle')) and \
                os.path.exists(os.path.join(self.vocab_prefix, 'openie_to_word.pickle')):
            sys.stderr.write('Reading OpenIE Rel dicts from cache...')
            openie_to_word = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_to_word.pickle'), 'rb'))
            openie_to_id = cPickle.load(open(os.path.join(self.vocab_prefix, 'openie_to_id.pickle'), 'rb'))
            sys.stderr.write('Finished.\n')

        sys.stderr.write('Start constructing test data loader from {}\n'.format(self.test_dataset_path))

        max_kb_align_num, max_openie_rel_word, max_kb_rel_word = 0, 0, 0
        if self.dataset == 'webqsp':
            max_kb_align_num, max_openie_rel_word, max_kb_rel_word = 7, 8, 8
        else:
            max_kb_align_num, max_openie_rel_word, max_kb_rel_word = 9, 13, 9

        test_data_loader = None
        if self.model_opt == 'hybrid':
            test_data_loader = Hybrid_ComplexWebQ_Test_Data(
                dataset_path=self.test_dataset_path, dataset_sub2_path=self.test_dataset_sub2_path,
                q_word_to_idx=self.q_word_to_idx, q_dep_to_idx=self.q_dep_to_idx,
                rel_word_to_idx=self.rel_word_to_idx, rel_id_to_idx=self.rel_id_to_idx, openie_mapping=self.openie_mapping,
                openie_word_to_idx=self.openie_word_to_idx, openie_id_to_idx=self.openie_id_to_idx,
                openie_to_word=openie_to_word, openie_to_id=openie_to_id, constraint_word_to_idx=self.constraint_word_to_idx,
                constraint_id_to_idx=self.constraint_id_to_idx, use_constraint=self.use_constraint,
                use_openie=self.use_openie, unified_openie=self.unified_openie, use_kb=self.use_kb,
                max_kb_rel_word=max_kb_rel_word, max_constraint_word=10, max_constraint_num=4, batch_size=self.batch_size, max_openie_rel_word=13,
                cache_dir=self.cache_dir, use_cache=self.use_cache, save_cache=self.save_cache,
                cache_prefix=self.test_cache_prefix)
        else:
            test_data_loader = Align_ComplexWebQ_Test_Data(
                dataset_path=self.test_dataset_path, dataset_sub2_path=self.test_dataset_sub2_path,
                q_word_to_idx=self.q_word_to_idx, q_dep_to_idx=self.q_dep_to_idx,
                rel_word_to_idx=self.rel_word_to_idx, rel_id_to_idx=self.rel_id_to_idx, openie_mapping=self.openie_mapping,
                openie_word_to_idx=self.openie_word_to_idx, openie_id_to_idx=self.openie_id_to_idx,
                openie_to_word=openie_to_word, openie_to_id=openie_to_id, constraint_word_to_idx=self.constraint_word_to_idx,
                constraint_id_to_idx=self.constraint_id_to_idx, use_constraint=self.use_constraint,
                use_openie=self.use_openie, unified_openie=self.unified_openie, use_kb=self.use_kb, kb_prop=self.kb_prop,
                max_kb_rel_word=max_kb_rel_word, max_constraint_word=10, max_constraint_num=4, batch_size=self.batch_size,
                max_openie_rel_word=max_openie_rel_word, max_openie_align_num=max_kb_align_num,
                cache_dir=self.cache_dir, use_cache=self.use_cache, save_cache=self.save_cache,
                cache_prefix=self.test_cache_prefix
            )

        sys.stderr.write('Finish Constructing test dataset loader..\n')

        sys.stderr.write('Start making tester...\n')
        tester = Hybrid_Tester(
            data_loader=test_data_loader, q_word_to_idx=self.q_word_to_idx, q_dep_to_idx=self.q_dep_to_idx,
            rel_word_to_idx=self.rel_word_to_idx, rel_id_to_idx=self.rel_id_to_idx,
            constraint_word_to_idx=self.constraint_word_to_idx, constraint_id_to_idx=self.constraint_id_to_idx,
            openie_word_to_idx=self.openie_word_to_idx, openie_id_to_idx=self.openie_id_to_idx,
            q_word_emb=self.q_word_emb, q_dep_emb=self.q_dep_emb, rel_word_emb=self.rel_word_emb,
            rel_id_emb=self.rel_id_emb, constraint_word_emb=self.constraint_word_emb,
            constraint_id_emb=self.constraint_id_emb, openie_word_emb=self.openie_word_emb,
            openie_id_emb=self.openie_id_emb, use_attn=self.use_attn, use_constraint=self.use_constraint,
            use_openie=self.use_openie, use_kb=self.use_kb, use_cache=self.use_cache, save_cache=self.save_cache,
            cache_dir=self.cache_dir, cache_prefix=self.test_cache_prefix, model_opt=self.model_opt,
            dataset=self.dataset
        )
        sys.stderr.write('Finish making tester...\n')
        tester.data_loader.write_sub1_file(sub1_path)
        sys.stderr.write('Start making tester inferencer...')
        test_inferencer = None
        if self.dataset == 'complexwebq':
            if self.eval_machine == 0:
                test_inferencer = Tester_Interface(
                    ques_src=self.test_src_path, sub1_flat_file_path=sub1_path, sub1_cands_dir=self.test_sub1_cands_dir,
                    sub2_cands_dir=model_sub2_cands_dir, sub1_openie_dir=self.test_openie_sub1_dir,
                    sub2_openie_dir=self.test_openie_sub2_dir)
            else:
                test_inferencer = Tester_Interface(
                    ques_src=self.test_src_path, sub1_flat_file_path=sub1_path, sub1_cands_dir=self.test_sub1_cands_dir,
                    sub2_cands_dir=model_sub2_cands_dir, sub1_openie_dir=self.test_openie_sub1_dir,
                    sub2_openie_dir=self.test_openie_sub2_dir, sparql_host='141.212.110.193', sparql_port='3095')
        else:
            test_inferencer = WebQTestInterface(
                    ques_src=self.test_src_path, lookup_path=sub1_path, kb_cands_dir=self.test_sub1_cands_dir,
                    openie_cands_dir=self.test_openie_sub1_dir
                )
            sys.stderr.write('Finished\n')
        top_K = 20 if self.dataset == 'complexwebq' else 100
        tester.viz_model_prediction(
            model=model, model_name=self.test_model_prefix,
            running_output=os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix),
            sub1_path=sub1_path, sub2_path=sub2_path, tester_inferencer=test_inferencer, top_K=top_K,
        )
        if self.dataset == 'webqsp':
            n_bests = [1, 5, 10, 25]
            pred_dir = os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix)
            output_path = os.path.join(pred_dir, "final_results.csv")
            test_inferencer.evaluate(pred_dir, output_path)
            for n_best in n_bests:
                f1 = test_inferencer.get_average_f1(output_path, n_best)
                print "{}-best: {}".format(n_best, f1)
        else:
            n_bests = [25, 10, 5, 1]
            for n_best in n_bests:
                print "n_best: {}".format(n_best)
                score = test_inferencer.get_average_f1(
                    os.path.join(self.test_running_prefix, self.model_conf_name, self.test_model_prefix,
                    self.test_model_prefix + '_prediction.csv'), n_best)
                print "f1: {}".format(score)

if __name__ == '__main__':
    config = None
    try:
        config = sys.argv[1]
    except:
        sys.stderr.write('WARNING: Use default config {}'.format(config))
        config = 'Hybrid_configs/webqsp/ablation/wo_attn.conf'

    pipeline_sys = Hybrid_pipeline(config)









