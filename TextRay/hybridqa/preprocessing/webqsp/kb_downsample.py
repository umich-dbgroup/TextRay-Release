import os
import json
from kbEndPoint.utils.sparql import sparqlUtils
from preprocessing import stringUtils
from preprocessing import metricUtils
import ast
import numpy as np

import codecs
import pandas as pd



class DownsampleInterface(object):

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

    def downsample(self, src, dest, frac=0.9):
        preds_df = pd.read_csv(src)
        preds_df['sub1_relation'] = preds_df['sub1_relation'].apply(lambda x: ast.literal_eval(x))
        preds_df['sub1_constraints'] = preds_df['sub1_constraints'].apply(lambda x: ast.literal_eval(x))
        preds_df['pred_entities'] = preds_df['pred_entities'].apply(lambda x: ast.literal_eval(x))
        preds_df['is_open'] = preds_df['sub1_relation'].apply(lambda x: self.is_open(x))
        print preds_df.shape
        self.get_average_f1(src, 1)

        kb_df = preds_df[preds_df['is_open']==0]

        openie_df = preds_df[preds_df['is_open']>0]

        kb_df = kb_df.groupby('qid').apply(lambda x: x.sample(frac=frac)).reset_index(drop=True)
        print kb_df.shape

        downsampled_df = pd.concat([openie_df, kb_df])
        downsampled_df.to_csv(dest, index=False)
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
    working_dir = "/media/nbhutani/Data/textray_workspace/emnlp_data/webqsp"
    ques_data_dir = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final"
    test_input_path = os.path.join(ques_data_dir, "data/WebQSP.test.json")

    # configs = {
    #     "kb_0.9": {
    #                   "data_dir": "kb_downsample/kb_0.9",
    #                   "pred_src": "prediction.csv",
    #                   "pred_dest": "prediction_0.9.csv",
    #                     "frac": 0.9
    #               },
    #     "kb_0.75": {
    #                    "data_dir": "kb_downsample/kb_0.75",
    #                     "pred_src": "prediction.csv",
    #                     "pred_dest": "prediction_0.75.csv",
    #                     "frac": 0.75
    #                },
    #     "kb_0.5": {
    #                   "data_dir": "kb_downsample/kb_0.5",
    #                     "pred_src": "prediction.csv",
    #                     "pred_dest": "prediction_0.5.csv",
    #                     "frac": 0.5
    #               }
    # }

    configs = {
        "kb_0.9": {
            "data_dir": "kb_downsample/kb_0.9",
            "pred_src": "prediction_align.csv",
            "pred_dest": "prediction_align_0.9.csv",
            "frac": 0.9
        },
        "kb_0.75": {
            "data_dir": "kb_downsample/kb_0.75",
            "pred_src": "prediction_align.csv",
            "pred_dest": "prediction_align_0.75.csv",
            "frac": 0.75
        },
        "kb_0.5": {
            "data_dir": "kb_downsample/kb_0.5",
            "pred_src": "prediction_align.csv",
            "pred_dest": "prediction_align_0.5.csv",
            "frac": 0.5
        }
    }

    config = configs["kb_0.5"]

    data_dir = os.path.join(working_dir, config["data_dir"])
    print data_dir

    interface = DownsampleInterface(test_input_path)
    src = os.path.join(data_dir, config["pred_src"])
    dest = os.path.join(data_dir, config["pred_dest"])
    interface.downsample(src, dest, config["frac"])
