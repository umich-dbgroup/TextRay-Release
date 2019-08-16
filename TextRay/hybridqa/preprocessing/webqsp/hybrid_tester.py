import os
import json
from kbEndPoint.utils.sparql import sparqlUtils
from preprocessing import stringUtils
from preprocessing import metricUtils
import ast
import numpy as np

import codecs
import pandas as pd



class WebQTestInterface(object):

    def __init__(self, ques_src, lookup_path, kb_cands_dir, openie_cands_dir):
        self.sparql = sparqlUtils()
        self.ques_src = ques_src
        self.kb_cands_dir = kb_cands_dir
        self.lookup = pd.read_csv(lookup_path, sep=',')
        self.lookup['index'] = self.lookup['index'].astype(int)
        self.openie_cands_dir = openie_cands_dir
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

    def __get_paths__(self, qid):
        path_dict = {}
        kb_path = os.path.join(self.kb_cands_dir, qid + ".json")  # read sub1 json
        if os.path.exists(kb_path):
            kb_json = json.load(codecs.open(kb_path, 'r', encoding='utf-8'))
            for topic, paths in kb_json.items():
                for path in paths:
                    path["topic"] = topic
                    path["is_openie"] = False
                    constraints = []
                    if 'constraints' in path:
                        for constraint in path['constraints']:
                            constraints.append(constraint['relation'])
                    key = topic + "_" + str(tuple([p for p in path['relations']])) + "_" + str(tuple(constraints))
                    key_paths = path_dict.get(key, [])
                    key_paths.append(path)
                    path_dict[key] = key_paths
        openie_path = os.path.join(self.openie_cands_dir, qid + ".json")  # read sub1 json
        if os.path.exists(openie_path):
            openie_json = json.load(codecs.open(openie_path, 'r', encoding='utf-8'))
            for topic, paths in openie_json.items():
                for path in paths:
                    path["topic"] = topic
                    path["is_openie"] = True
                    constraints = []
                    if 'constraints' in path:
                        for constraint in path['constraints']:
                            constraints.append(constraint['relation'])
                    key = topic + "_" + str(tuple([p for p in path['relations']])) + "_" + str(tuple(constraints))
                    key_paths = path_dict.get(key, [])
                    key_paths.append(path)
                    path_dict[key] = key_paths
        return path_dict

    def __get_paths_data__(self, pred, path_dict):
        rel_data_keys = self.lookup[self.lookup["index"] == pred["index"]].to_dict('records')
        if len(rel_data_keys) == 0:
            return None, None
        rel_data_key = rel_data_keys[0]
        topic = rel_data_key["topic"]
        if pred["is_openie"]:
            rels = tuple([pred['sub1_openie']])
            look_up_key = topic + "_" + (str(rels)) + "_" + (str(()))
        else:
            look_up_key = topic + "_" + str(tuple(pred["sub1_relation"])) + "_" + str(pred.get("sub1_constraints", ()))
        if not look_up_key in path_dict:
            print look_up_key
            print path_dict
            return topic, None
        rel_data = path_dict[look_up_key]
        return topic, rel_data

    def evaluate(self, preds_dir, output_path):
        records = []
        for qid, ans_dict in self.ground_ans_dict.items():
            # print qid
            pred_path = os.path.join(preds_dir, "sub1_pred_" + qid + ".json")
            if not os.path.exists(pred_path):
                continue
            path_dict = self.__get_paths__(qid)
            predictions = json.load(codecs.open(pred_path))
            for prediction in predictions:
                topic, rel_data = self.__get_paths_data__(prediction, path_dict)
                for rel in rel_data:
                    if "entities_score" in rel:
                        entity_scores = rel["entities_score"]
                    else:
                        entity_scores = [1.0 for e in rel['entities']]
                    record = self.to_record(qid, prediction["index"], topic, rel, prediction["sub1_score"],
                                            rel['entities'], entity_scores)

                    records.append(record)
        if len(records) == 0:
            print("evaluation records should not have not been empty")
        preds_df = pd.DataFrame(records)
        print output_path
        preds_df.to_csv(output_path, index=False)
        self.get_average_f1(output_path, 1)

    def to_record(self, qid, index, topic, rel_data, score, entities, entity_scores):
        record = {"qid": qid,
                  "sub1_index": index,
                  "topic": topic,
                  "sub1_relation": rel_data.get("relations", []),
                  "sub1_constraints": rel_data.get("constraints", []),
                  "sub1_score": score,
                  "pred_entities": entities,
                  "agg_score": score,
                  "entity_scores":  entity_scores
                  }
        return record

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

    def is_open(self, rels):
        if len(rels) == 0:
            return 0
        for rel in rels:
            if len(rel.split('.')) > 2:
                return 0
        return 1

    def downsample(self, src, tgt, frac):
        preds_df = pd.read_csv(src)
        preds_df['sub1_relation_arr'] = preds_df['sub1_relation'].apply(lambda x: ast.literal_eval(x))

        preds_df['is_openie'] = preds_df.apply(lambda x: self.is_open(x['sub1_relation_arr']), axis=1)

        kb_df = preds_df[preds_df['is_openie'] == 0]
        openie_df = preds_df[preds_df['is_openie'] > 0]
        print(openie_df.shape)
        downsampled_df = kb_df.drop_duplicates(['qid', 'sub1_relation'])
        downsampled_df = downsampled_df.groupby('qid').apply(lambda x: x.sample(frac=frac)).reset_index(drop=True)
        # final_df = pd.concat([openie_df, downsampled_df])
        # final_df.to_csv(tgt)
        # return self.get_average_f1(tgt, 1)


    def downsample_kb_only(self, src, tgt):
        preds_df = pd.read_csv(src)
        preds_df['sub1_relation_arr'] = preds_df['sub1_relation'].apply(lambda x: ast.literal_eval(x))

        preds_df['is_openie'] = preds_df.apply(lambda x: self.is_open(x['sub1_relation_arr']), axis=1)

        kb_df = preds_df[preds_df['is_openie'] == 0]
        kb_df.to_csv(tgt, index=None)
        return self.get_average_f1(tgt, 1)

    def get_key(self, row):
        return row['qid'] + "_" + row['topic'] + "_" + row['sub1_relation'] + "_" + row['sub1_constraints']

    def compare(self, pred1, pred2):
        df1 = pd.read_csv(pred1)
        df2 = pd.read_csv(pred2)

        df1['sub1_relation_arr'] = df1['sub1_relation'].apply(lambda x: ast.literal_eval(x))
        df1['is_openie'] = df1.apply(lambda x: self.is_open(x['sub1_relation_arr']), axis=1)

        df2['sub1_relation_arr'] = df2['sub1_relation'].apply(lambda x: ast.literal_eval(x))
        df2['is_openie'] = df2.apply(lambda x: self.is_open(x['sub1_relation_arr']), axis=1)

        kb_df1 = df1[df1['is_openie'] == 0]
        kb_df2 = df2[df2['is_openie'] == 0]

        kb_df1['key'] = df1.apply(lambda x: self.get_key(x), axis=1)
        kb_df2['key'] = df2.apply(lambda x: self.get_key(x), axis=1)

        keys1 = set(kb_df1['key'].values)
        keys2 = set(kb_df2['key'].values)
        print(len(keys1))
        print(len(keys2))
        for key in keys2:
            if key not in keys1:
                print key


if __name__ == '__main__':
    # PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final"
    # ques_src = os.path.join(PREFIX, "data/WebQSP.test.json")
    # pred_dir = "/media/nbhutani/Data/textray_workspace/emnlp_data/webqsp/debug"
    # output_path = os.path.join(pred_dir, "final_results.csv")
    # kb_cand_dir = os.path.join(PREFIX, "cands_with_constraints-scaled-test")
    # openie_cand_dir = os.path.join(PREFIX, "stanfordie/all_final/normalized_cands/train_cands_lemmatized/test-dedup")
    # webqtest = WebQTestInterface(ques_src, kb_cand_dir, openie_cand_dir)
    # webqtest.evaluate(pred_dir, output_path)
    # # print(webqtest.get_average_f1(output_path, 1))

    '''========'''
    split = 'test'
    PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final"
    ques_src = os.path.join(PREFIX, "data/WebQSP." + split + ".json")
    working_dir = "/media/nbhutani/Data/textray_workspace/emnlp_data/webqsp"
    kb_cand_dir = os.path.join(PREFIX, "cands_with_constraints-scaled-" + split)
    openie_cand_dir = os.path.join(PREFIX, "stanfordie/all_final/normalized_cands/train_cands_lemmatized/" + split + "-dedup")


    configs = {
        "train": {
            "data_dir": "full_model/train_set",
            "pred_file": "prediction.csv"
        },
        "kb_only": {
            "data_dir": "kbonly",
            "pred_file": "prediction.csv"
        },
        "test": {
            "data_dir": "full_model/test_set",
            "pred_file": "prediction.csv"
        },
        "wo_constraints": {
            "data_dir": "wo_cons",
            "pred_file": "prediction.csv"
        },
        "wo_prior": {
            "data_dir": "wo_prior",
            "pred_file": "prediction.csv"
        },
        "wo_attn": {
            "data_dir": "wo_attn3",
            "pred_file": "prediction.csv"
        }
        # "wo_attn": {
        #     "data_dir": "wo_attn",
        #     "pred_file": "prediction.csv"
        # }
    }

    config = configs["wo_attn"]
    pred_dir = os.path.join(working_dir, config["data_dir"])

    print pred_dir

    lookup = os.path.join(pred_dir, "sub1_lookup.csv")
    webqtest = WebQTestInterface(ques_src, lookup, kb_cand_dir, openie_cand_dir)
    webqtest.evaluate(pred_dir, os.path.join(pred_dir, config["pred_file"]))

    # df = pd.read_csv(output_path)
    # downsampled_df = df.groupby('qid').apply(lambda x: x.sample(frac=0.9)).reset_index(drop=True)
    # downsampled_df.to_csv(os.path.join(pred_dir, "final_results_0.9.csv"), index=False)
    # print(webqtest.get_average_f1(os.path.join(pred_dir, "final_results.csv"), 1))
    # print(webqtest.downsample(output_path, os.path.join(pred_dir, "final_results_0.5.csv"), 0.5))


