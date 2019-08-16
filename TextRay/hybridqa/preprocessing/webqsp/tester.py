import os
import json
from kbEndPoint.utils.sparql import sparqlUtils
from preprocessing import stringUtils
from preprocessing import metricUtils
import numpy as np

import codecs
import pandas as pd



class WebQTestInterface(object):

    def __init__(self):
        self.sparql = sparqlUtils()

    def get_f1(self, ques_src, cands_dir):
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))["Questions"]
        all_f1s = []
        for q in questions:
            ques_id = q["QuestionId"]
            parses = q["Parses"]
            entity_ans_dict = {}
            for parse in parses:
                topic_entity = parse["TopicEntityMid"]
                answer_entities = [a["AnswerArgument"] for a in parse["Answers"]]
                entity_ans_dict[topic_entity] = answer_entities
            ques_path = os.path.join(cands_dir, ques_id)
            if not os.path.exists(ques_path):
                continue
            print(ques_id)
            best_f1 = 0
            best_score = 0.0
            main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
            for topic in main_entity_paths:
                ground_ans = entity_ans_dict.get(topic, [])
                for path in main_entity_paths[topic]:
                    if path["score"] > best_score:
                        best_score = path["score"]
                        if len(ground_ans) == 0:
                            best_f1 = 0
                        else:
                            predicted_ans = path["entities"]
                            best_f1 = metricUtils.compute_f1(ground_ans, predicted_ans)[2]
            all_f1s.append(best_f1)
        print(np.mean(all_f1s))

    def get_accuracy(self, ques_src, cands_dir):
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))["Questions"]
        true_pos_ct = 0
        total_ct = 0
        for q in questions:
            ques_id = q["QuestionId"]
            # print(ques_id)
            ques_path = os.path.join(cands_dir, ques_id + ".json")
            if not os.path.exists(ques_path):
                continue
            is_true = False
            total_ct += 1
            best_score = 0.0
            main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
            for topic in main_entity_paths:
                for path in main_entity_paths[topic]:
                    if path["score"] > best_score:
                        best_score = path["score"]
                        if path["true_label"] == 1:
                            is_true = True
                        else:
                            is_true = False
            if is_true:
                true_pos_ct += 1
        print(true_pos_ct)
        print(total_ct)
        print(float(true_pos_ct) * 1.0 / float(total_ct))

    def get_predictions(self, pred_dir, include_constraints=False):
        files = [f for f in os.listdir(pred_dir) if os.path.isfile(os.path.join(pred_dir, f))]
        pred_records = []  # list of tuple (ques_id, pred_sequence, constraint_seq, scores) (sorted)
        for f in files:

            pred_json = json.load(open(os.path.join(pred_dir, f), 'r'))
            pred_dict = set()
            for pred in pred_json:
                inference_chain = pred['relation']
                if len(inference_chain) == 2:
                    tuple = (unicode(inference_chain[0]), unicode(inference_chain[1]))
                else:
                    tuple = (unicode(inference_chain[0]),)
                pred_key = str(tuple)
                score = pred['score']
                constraints = []
                if include_constraints:
                    for p in pred["constraints"]:
                        if p == "<CONSTRAINT_ID_PAD>": continue
                        constraints.append(p)
                constraints_key = str(constraints)
                if not include_constraints:
                    if pred_key in pred_dict: continue
                    pred_dict.add(pred_key)
                else:
                    if pred_key + ":" + constraints_key in pred_dict: continue
                    pred_dict.add(pred_key + ":" + constraints_key)
                pred_record = (f, pred_key, constraints_key, score)
                pred_records.append(pred_record)
        labels = ['ques_id', 'pred_sequence', 'constraint_sequence', 'score']
        return pd.DataFrame.from_records(pred_records, columns=labels)

    def write_join_df(self, ques_src, pred_dir, label_path, results_path, include_constraints=False):
        predictions_df = self.get_predictions(pred_dir, include_constraints=include_constraints)
        prediction_ans_df = self.get_predictions_answer_sets(label_path)
        df = predictions_df.merge(prediction_ans_df, on=['ques_id', 'pred_sequence', 'constraint_sequence'], how='left')
        #print(df.shape)
        ground_truth_df = self.get_ground_truth(ques_src)
        #print(ground_truth_df.shape)

        if include_constraints:
            labels_df = df.merge(ground_truth_df,
                                 on=['ques_id', 'topic_entity', 'pred_sequence', 'constraint_sequence'], how='left',
                                 indicator='pred_match')
        else:
            labels_df = df.merge(ground_truth_df, on=['ques_id', 'topic_entity', 'pred_sequence'],
                                 how='left', indicator='pred_match')
        #print(labels_df.shape)
        labels_df['pred_match'] = np.where(labels_df.pred_match == 'both', 1, 0)
        labels_df.to_csv(results_path, index=None)

if __name__ == '__main__':
    PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final"
    ques_src = os.path.join(PREFIX, "data/WebQSP.test.json")
    PRED_PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/results/webqsp_wo_retrain/test_result"
    preds_dir = os.path.join(PRED_PREFIX, "E_5")


    webqtest = WebQTestInterface()

    webqtest.get_f1(ques_src, preds_dir)