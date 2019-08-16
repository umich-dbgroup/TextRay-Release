import json
import os
import sys
import pandas as pd
import re
from preprocessing import metricUtils
import numpy as np
import codecs
import os
import operator
import ast
from collections import OrderedDict




class Align_Tester_Interface(object):

    def __init__(self, ques_src):
        self.ques_src = ques_src
        self.questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))["Questions"]
        self.ground_ans_dict = {}
        for q in self.questions:
            ques_id = q["QuestionId"]
            parses = q.get("Parses", [])
            entity_ans_dict = {}
            for parse in parses:
                topic_entity = parse["TopicEntityMid"]
                answers = parse.get("Answers", [])
                entity_ans_dict[topic_entity] = [a["AnswerArgument"] for a in answers]
            self.ground_ans_dict[ques_id] = entity_ans_dict

    def is_open(self, rels):
        if len(rels) == 0:
            return 0
        for rel in rels:
            if len(rel.split('.')) > 2:
                return 0
        return 1

    def get_kb_answer_dict(self, df):
        print df.shape
        kb_df = df[(df['is_open1'] == 0)]
        print kb_df.shape
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
        if row['is_open1'] == 1:
            answers = ans_dict.get(row['qid'], set())
            row_preds = row['pred_entities']
            is_aligned = len(set.intersection(set(row_preds), answers)) > 0
            return is_aligned
        return True

    def filter_openie(self, src, dest):
        df = pd.read_csv(src)
        print df.shape
        self.get_average_f1(src, 1)
        df['pred_entities'] = df['pred_entities'].apply(lambda x: ast.literal_eval((x)))
        df['sub1_relation'] = df['sub1_relation'].apply(lambda x: ast.literal_eval((x)))
        df['is_open1'] = df['sub1_relation'].apply(lambda x: self.is_open(x))
        answers_dict = self.get_kb_answer_dict(df)
        df['is_aligned'] = df.apply(lambda x: self.is_aligned(x, ans_dict=answers_dict), axis=1)
        df = df[df['is_aligned']==True]
        df = df.drop(['is_aligned', 'is_open1'], axis=1)
        df.to_csv(dest, index=False)
        print df.shape
        print dest
        self.get_average_f1(dest, 1)


    def get_average_f1(self, final_results_path, n_best=25):
        df = pd.read_csv(final_results_path)
        all_f1s = []
        prec_1 = 0
        total_prec1 = 0
        results = {}
        for qid, group in df.groupby("qid"):
            total_prec1 += 1
            group_df = group.reset_index()
            group_df['pred_entities'] = group_df['pred_entities'].apply(lambda x: ast.literal_eval(x))
            group_df = group_df.sort_values(by=["sub1_score"], ascending=False)
            group_df_sub_records = group_df.head(min(len(group_df), n_best)).to_dict('records')
            best_f1 = 0.0
            is_true = False
            for record in group_df_sub_records:
                ground_answers = self.ground_ans_dict[qid].get(record['topic'], [])
                if len(ground_answers) == 0:
                    recall, precision, f1 = 0.0, 0.0, 0.0
                else:
                    recall, precision, f1 = metricUtils.compute_f1(ground_answers, record['pred_entities'])
                if f1 > best_f1:
                    best_f1 = f1
                if is_true:
                    continue
                for pred in record['pred_entities']:
                    if pred in ground_answers:
                        is_true = True
                        break
            if is_true:
                prec_1 += 1

            all_f1s.append(best_f1)
        macro_avg_f1 = float(sum(all_f1s)) / len(self.questions)
        prec_at_1 = float(prec_1) / len(self.questions)
        results["macro"] = (macro_avg_f1)
        results["hit1"] = prec_at_1
        print results
        return results

if __name__ == '__main__':
    split="test"
    PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final"
    working_dir = "/media/nbhutani/Data/textray_workspace/emnlp_data/webqsp"
    ques_src = os.path.join(PREFIX, "data/WebQSP." + split + ".json")
    test_interface = Align_Tester_Interface(ques_src)

    configs = {
        "train": {
            "data_dir": "full_model/train_set",
            "pred_file_src": "prediction.csv",
            "pred_file_dest": "prediction_align.csv"
        },
        "align": {
            "data_dir": "full_model/test_set",
            "pred_file_src": "prediction.csv",
            "pred_file_dest": "prediction_align.csv"
        },
        "wo_constraints": {
            "data_dir": "wo_cons",
            "pred_file_src": "prediction.csv",
            "pred_file_dest": "prediction_align.csv"
        },
        "wo_prior": {
            "data_dir": "wo_prior",
            "pred_file_src": "prediction.csv",
            "pred_file_dest": "prediction_align.csv"
        },
        "wo_attn": {
            "data_dir": "wo_attn3",
            "pred_file_src": "prediction.csv",
            "pred_file_dest": "prediction_align.csv"
        }
        # "wo_attn": {
        #     "data_dir": "wo_attn",
        #     "pred_file_src": "prediction.csv",
        #     "pred_file_dest": "prediction_align.csv"
        # }
    }

    # config = configs["align"]
    # config = configs["wo_constraints"]
    # config = configs["wo_prior"]
    config = configs["wo_attn"]

    running_dir = os.path.join(working_dir, config["data_dir"])
    print running_dir
    src = os.path.join(running_dir, config["pred_file_src"])
    dest = os.path.join(running_dir, config["pred_file_dest"])

    test_interface.filter_openie(src, dest)

