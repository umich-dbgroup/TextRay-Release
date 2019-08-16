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

#PREFIX = "/z/zxycarol/ComplexWebQuestions_Resources/ComplexWebQuestions_preprocess"
PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess"
MAX_NEGATIVES_FRAC = 5

MAX_DEGREE = 2000

# RAW_INPUT_PATH = os.path.join(PREFIX, "annotated_orig/test.json")
# INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
# RAW_EL_PATH = os.path.join(PREFIX, "el/sub1/test_el.csv")
# TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/test_topic.csv")
# SUB1_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub1/test_topic.csv")
# SUB2_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub2/test_topic.csv")
#
#
# CANDIDATE_SUB1_DEST_PATH = os.path.join(PREFIX, "cands/test/sub1")
# CANDIDATE_SUB1_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/test/sub1_with_constraints")
# CANDIDATE_SUB1_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/dev/sub1_with_constraints")
# CANDIDATE_SUB2_DEST_PATH = os.path.join(PREFIX, "cands/test/sub2")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/test/main_with_constraints")

entity_pattern = re.compile(r'ns:([a-z]\.([a-zA-Z0-9_]+)) ')
ANS_CONSTRAINT_RELATIONS = ["people.person.gender", "common.topic.notable_types", "common.topic.notable_for"]

from string import Template


class Align_Tester_Interface(object):

    def __init__(self, ques_src):
        self.ques_src = ques_src
        self.questions_dict = {}
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
        for q in questions:
            if q["compositionality_type"] == "composition" or q["compositionality_type"] == "conjunction":
                self.questions_dict[q["ID"]] = q

    def is_open(self, rels):
        if len(rels) == 0:
            return 0
        for rel in rels:
            if len(rel.split('.')) > 2:
                return 0
        return 1

    def get_kb_answer_dict(self, df):
        # print df.shape
        kb_df = df[(df['is_open1'] == 0) & (df['is_open2'] == 0)]
        # print kb_df.shape
        kb_entities_dict = {}
        for qid, group in kb_df.groupby("qid"):
            group = group.reset_index()
            answers = set()
            records= group.to_dict(orient='records')
            for r in records:
                for a in r['pred_entities']:
                    answers.add(a)
            kb_entities_dict[qid] = answers
        return kb_entities_dict

    def is_aligned(self, row, ans_dict):
        if row['is_open1'] == 1 or row['is_open2'] == 1:
            answers = ans_dict.get(row['qid'], set())
            row_preds = row['pred_entities']
            is_aligned = len(set.intersection(set(row_preds), answers)) > 0
            return is_aligned
        return True

    def filter_openie(self, src, dest):
        df = pd.read_csv(src)
        print df.shape
        df[['sub2_relation', 'sub2_constraints']] = df[['sub2_relation', 'sub2_constraints']].fillna(value='[]')
        df['pred_entities'] = df['pred_entities'].apply(lambda x: ast.literal_eval((x)))
        df['sub1_relation'] = df['sub1_relation'].apply(lambda x: ast.literal_eval((x)))
        df['sub2_relation'] = df['sub2_relation'].apply(lambda x: ast.literal_eval((x)))
        df['is_open1'] = df['sub1_relation'].apply(lambda x: self.is_open(x))
        df['is_open2'] = df['sub2_relation'].apply(lambda x: self.is_open(x))
        answers_dict = self.get_kb_answer_dict(df)
        df['is_aligned'] = df.apply(lambda x: self.is_aligned(x, ans_dict=answers_dict), axis=1)
        df = df[df['is_aligned']==True]
        df = df.drop(['is_aligned', 'is_open1', 'is_open2'], axis=1)
        df.to_csv(dest, index=False)
        # print df.shape


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
        return results

if __name__ == '__main__':
    # INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
    INPUT_PATH = os.path.join(PREFIX, "annotated/dev.json")

    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_wo_constraints"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_wo_prior"
    RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_dev"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_wo_attention"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_dev"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_only"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_downsample/kb_0.9"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_downsample/kb_0.75"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_downsample/kb_0.5"


    test_interface = Align_Tester_Interface(INPUT_PATH)

    FINAL_RESULTS_PATH = os.path.join(RUNNING_DIR, "9_prediction_v2.csv")
    FINAL_RESULTS_DEST_PATH = os.path.join(RUNNING_DIR, "9_prediction_align.csv")

    test_interface.filter_openie(FINAL_RESULTS_PATH, FINAL_RESULTS_DEST_PATH)

    print(RUNNING_DIR)
    n_bests = [1]
    for n_best in n_bests:
        print("{}: {}".format(n_best, test_interface.get_average_f1(FINAL_RESULTS_DEST_PATH, n_best=n_best)))
