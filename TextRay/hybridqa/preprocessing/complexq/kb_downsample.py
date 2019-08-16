import json
import os
import sys
import random
sys.path.insert(0, '../../')
from kbEndPoint.utils.sparql import sparqlUtils
import pandas as pd
import re
from preprocessing import metricUtils
import nltk
import numpy as np
import codecs
from corechainGen import CoreChainGen
from queryGraphGen import QueryGraphGen
from complexqEndpoint import ComplexQuestionEndPoint
import os
import operator
import ast
from collections import OrderedDict
import time

PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess"

class DownSampleInterface(object):
    def __init__(self, ques_src, running_dir, sub1_cands_dir, sub1_openie_dir, sub2_openie_dir,
                 sparql_host="141.212.110.80", sparql_port="3093"):
        self.ques_src = ques_src
        self.running_dir = running_dir
        self.sub1_flat_file_path = os.path.join(running_dir, "sub1_lookup.csv")
        self.sub1_cands_dir = sub1_cands_dir
        self.sub2_cands_dir = os.path.join(running_dir, "sub2_cands")
        self.sub1_openie_dir = sub1_openie_dir
        self.sub2_openie_dir = sub2_openie_dir

        self.questions_dict = {}
        self.sparql = sparqlUtils(host=sparql_host, port=sparql_port)
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))

        for q in questions:
            if q["compositionality_type"] == "composition" or q["compositionality_type"] == "conjunction":
                self.questions_dict[q["ID"]] = q

        self.test_index = pd.read_csv(self.sub1_flat_file_path, sep=',')
        self.test_index['index'] = self.test_index['index'].astype(int)

    def evaluate(self, frac=0.9, n_best=1):
        records = []
        queries_dest = os.path.join(self.running_dir, "queries_" + str(frac))
        for ques_id, question in self.questions_dict.iteritems():
            comp_type = question["compositionality_type"]
            if comp_type == "composition" or comp_type == "conjunction":
                queries_path = os.path.join(queries_dest, "query_" + ques_id + ".json")
                if not os.path.exists(queries_path):
                    continue
                if os.path.exists(queries_path):
                    records_df = pd.read_csv(queries_path)
                    records += (records_df.to_dict("records"))
        preds_df = pd.DataFrame(records)
        output_path = os.path.join(self.running_dir, "9_prediction_v2_" + str(frac) + ".csv")
        preds_df.to_csv(output_path, index=False)

        metrics = self.get_average_f1(output_path, n_best)
        print metrics
        return metrics

    def get_average_f1(self, src, n_best=25):
        df = pd.read_csv(src)
        results = {}
        all_f1s = []
        all_precision = []
        all_recall = []
        prec_1 = 0
        total_prec1 = 0
        for qid, group in df.groupby("qid"):
            total_prec1 += 1
            question = self.questions_dict[qid]
            ground_answers = question["Answers"]
            group_df = group.reset_index()
            group_df['sub2_score'].fillna(0.0, inplace=True)
            group_df['agg'] = group_df['sub1_score'] + group_df['sub2_score']
            group_df['pred_entities'] = group_df['pred_entities'].apply(lambda x: ast.literal_eval(x))
            group_df = group_df.sort_values(by=["agg"], ascending=False)
            group_df_sub_records = group_df.head(min(len(group_df), n_best)).to_dict('records')
            best_f1 = 0.0
            best_recall = 0.0
            best_precision = 0.0
            is_true = False
            for record in group_df_sub_records:
                if len(ground_answers) == 0:
                    recall, precision, f1 = 0.0, 0.0, 0.0
                else:
                    recall, precision, f1 = metricUtils.compute_f1(ground_answers, record['pred_entities'])
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                if is_true:
                    continue
                for pred in record['pred_entities']:
                    if pred in ground_answers:
                        is_true = True
                        break
            if is_true:
                prec_1 += 1

            all_f1s.append(best_f1)
            all_recall.append(best_recall)
            all_precision.append(best_precision)
        macro_avg_f1 = float(np.sum(all_f1s)) / float(len(self.questions_dict))
        prec_at_1 = float(prec_1) / float(len(self.questions_dict))
        results["macro_f1"] = macro_avg_f1
        results["hit1"] = prec_at_1
        print(len(all_f1s))
        return results

    def get_path_key(self, row):
        return str(row['sub1_index']) + "_" + str(row["sub2_relation"])

    def downsample(self, frac=0.9):
        files = os.listdir(self.running_dir)
        queries_dir = os.path.join(self.running_dir, "queries")
        queries_dest = os.path.join(self.running_dir, "queries_" + str(frac))
        if not os.path.exists(queries_dest):
            os.makedirs(queries_dest)
        files = [f for f in files if f.endswith("_sub2_data.json")]
        for f in files:
            qid = f.replace("_sub2_data.json", "")
            print qid
            sub2_data= json.load(codecs.open(os.path.join(self.running_dir, f)))
            keys = set()
            for topic, paths in sub2_data.items():
                for path in paths:
                    if path['is_openie']:
                        continue
                    key = str(topic + "_" + str(path['relations']))
                    keys.add(key)
            subset_keys = random.sample(keys, k=int(frac*len(keys)))
            # print(len(keys))
            # print(len(subset_keys))
            # print subset_keys
            for topic, paths in sub2_data.items():
                for path in paths:
                    if path['is_openie']:
                        subset_keys.append(topic + "_" + str(path['relations']))
            path_keys = set()
            for topic, paths in sub2_data.items():
                for path in paths:
                    key = str(topic + "_" + str(path['relations']))
                    if key in subset_keys:
                        path_keys.add(str(path['parent_index']) + "_" + str(path['relations']))

            queries = pd.read_csv(os.path.join(queries_dir, "query_" + qid + ".json"))
            queries['path_key'] = queries.apply(lambda x: self.get_path_key(x), axis=1)
            queries_filtered = queries[queries['path_key'].isin(path_keys)]
            queries_filtered.to_csv(os.path.join(queries_dest, "query_" + qid + ".json"), index=False)
            print(len(queries))
            print(len(queries_filtered))

if __name__ == '__main__':
    frac = 0.75
    RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_downsample/kb_" + str(frac)
    INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
    SUB1_CANDIDATES_DIR = os.path.join(PREFIX, "rewards/test/sub1")
    OPENIE_DIR_1 = os.path.join(PREFIX, "stanfordie/all/normalized_cands/train_cands/test_sub1")
    OPENIE_DIR_2 = os.path.join(PREFIX, "stanfordie/all/normalized_cands/train_cands/test_sub2")
    interface = DownSampleInterface(INPUT_PATH, RUNNING_DIR, SUB1_CANDIDATES_DIR, OPENIE_DIR_1, OPENIE_DIR_2, sparql_host="141.212.110.193", sparql_port="3095")
    # interface.downsample(frac=frac)
    interface.evaluate(frac=frac, n_best=1)
