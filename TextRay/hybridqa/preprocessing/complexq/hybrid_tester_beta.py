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


class Tester_Interface(object):

    def __init__(self, ques_src, sub1_flat_file_path, sub1_cands_dir, sub2_cands_dir, sub1_openie_dir, sub2_openie_dir, sparql_host="141.212.110.80", sparql_port="3093"):
        self.complexqEndpoint = ComplexQuestionEndPoint()
        self.corechainGen = CoreChainGen()
        self.queryGraphGen = QueryGraphGen()

        self.ques_src = ques_src
        self.sub1_flat_file_path = sub1_flat_file_path
        self.sub1_cands_dir = sub1_cands_dir
        self.sub2_cands_dir = sub2_cands_dir
        self.sub1_openie_dir = sub1_openie_dir
        self.sub2_openie_dir = sub2_openie_dir
        self.forward_cache = {}
        self.forward_cache_fifo = []
        self.query_cache = {}
        self.query_cache_fifo = []
        self.MAX_SIZE = 10000
        self.questions_dict = {}
        self.sparql = sparqlUtils(host=sparql_host, port=sparql_port)
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))

        for q in questions:
            if q["compositionality_type"] == "composition" or q["compositionality_type"] == "conjunction":
                self.questions_dict[q["ID"]] = q

        self.template = Template('''
                                        PREFIX ns: <http://rdf.freebase.com/ns/>
                                        SELECT DISTINCT ?x
                                        WHERE {
                                            ${r}
                                            ?x ns:type.object.name ?name .
                                            ${f}   
                                        }
                                        ''')

        self.test_index = pd.read_csv(sub1_flat_file_path, sep=',')
        self.test_index['index'] = self.test_index['index'].astype(int)

    def generate_all_sub2_candidates(self, ques_src, output_dir, n_best=6):
        questions = json.load(open(ques_src))
        for q in questions:
            qid = q["ID"]
            sub1_pred_path = os.path.join(output_dir, "sub1_pred_" + qid + ".json")
            if os.path.exists(sub1_pred_path):
                print(qid)
                top_k_preds = json.load(codecs.open(sub1_pred_path, encoding='utf-8'))
                top_k_preds = top_k_preds[:min(len(top_k_preds), n_best)]
                self.generate_sub2_candidates(top_k_preds, qid, os.path.join(output_dir, qid + "_sub2_data.json"))

    def generate_sub2_candidates(self, topk_results, qid, output_path):
        if not qid in self.questions_dict:
            # print("question not found in the source file")
            return {}
        question = self.questions_dict[qid]
        # print('question_id{}'.format(qid))
        # print(qid + "\t" + question["compositionality_type"])
        all_cands = {}
        path_dict = self.__get_sub1_paths__(qid)
        if os.path.exists(output_path):
            return {}
        for r in topk_results:
            # print('top result{}'.format(r))
            cand_index= r["index"]
            # print('top result index{}'.format(r))

            topic, rel_data = self.__get_sub1_paths_data__(cand_index, path_dict)
            # print("{} has {}".format(cand_index, topic))
            if rel_data is None:
                continue
            if question["compositionality_type"] == "composition":
                # interim_ans_set = self.__get_interim_entities__(topic, rel_data)
                interim_ans_set = rel_data["entities"]
                # print interim_ans_set
                cands = self.__get_compositional_candidates_cached__(qid, cand_index, topic, interim_ans_set)
                if len(cands) == 0:
                    if len(rel_data["entities"]) > 0:
                        print("no next cands found for result {}".format(rel_data["relations"]))
                    continue
                topic_cands = all_cands.get(topic, [])
                cands = [c for c in cands if str(c['relations']) != rel_data['relations']]
                topic_cands += cands
                all_cands[topic] = topic_cands
            elif question["compositionality_type"] == "conjunction":
                # print("{} rel data".format(rel_data["relations"]))
                interim_ans_set = rel_data["entities"]
                # print(len(interim_ans_set))
                conj_cands = self.__get_conjunction_candidates__(qid, cand_index, topic, interim_ans_set)
                for topic2, cands in conj_cands.iteritems():
                    topic_cands = all_cands.get(topic2, [])
                    cands = [c for c in cands if str(c['relations']) != rel_data['relations']]
                    topic_cands += cands
                    all_cands[topic2] = topic_cands
        if len(all_cands.keys()) == 0:
            print("{} failed to find next expansion candidates for any of the predictions".format(qid))
        with open(output_path, 'w+') as fp:
            json.dump(all_cands, fp, indent=4)


    def __get_sub1_paths__(self, qid):
        # print('reading query graph 1 for ' + qid)
        path = os.path.join(self.sub1_cands_dir, qid + ".json") # read sub1 json
        path_dict = {}
        if not os.path.exists(path):
            # print('path does not exist for qid ' + qid)
            return path_dict
        sub1_paths = json.load(codecs.open(path, 'r', encoding='utf-8'))
        for topic in sub1_paths.keys():
            for path in sub1_paths[topic]:
                path["topic"] = topic
                path["is_openie"] = False
                key = self.__get_path_key__(topic, path)
                path_dict[key] = path

        '''add openie info in look up'''
        openie_path = os.path.join(self.sub1_openie_dir, qid + ".json") # read sub1 json
        if os.path.exists(openie_path):
            sub1_open_paths = json.load(codecs.open(openie_path, 'r'))
            for topic in sub1_open_paths.keys():
                for path in sub1_open_paths[topic]:
                    path["topic"] = topic
                    path["is_openie"] = True
                    key = self.__get_path_key__(topic, path)
                    path_dict[key] = path
        return path_dict

    def __get_path_key__(self, topic, path):
        if path["is_openie"]:
            # rels = tuple([p.encode('ascii') for p in path['relations']])
            rels = tuple([p.encode('ascii') for p in path['unnormalized_relations']])
        else:
            rels = tuple([p for p in path['relations']])
        # rels = tuple([p for p in path['relations']])
        constraints = []
        if 'constraints' in path:
            constraints = [constraint['relation'] for constraint in path['constraints']]
        constraints = tuple(constraints)
        key = topic + "_" + str(rels) + "_" + str(constraints)
        return key

    def __get_sub1_paths_data__(self, cand_index, path_dict):
        # print 'Cand Index: {}'.format(cand_index)
        #rel_data_keys = self.test_index[self.test_index["index"] == cand_index].to_records(index=False)
        rel_data_keys = self.test_index[self.test_index["index"] == cand_index].to_dict('records')
        if len(rel_data_keys) == 0:
            # print "Key not found"
            return None, None
        rel_data_key = rel_data_keys[0]
        topic = rel_data_key["topic"]
        if rel_data_key["is_openie"]:
            rels = tuple([rel_data_key['openie']])
            lookup_key = topic + "_" + str(rels) + "_" + str(())
        else:
            look_up_key = topic + "_" + str(rel_data_key["relations"]) + "_" + str(rel_data_key.get("constraints", ()))
        if look_up_key not in path_dict:
            print look_up_key
            print path_dict
        rel_data = path_dict.get(look_up_key, None)
        return topic, rel_data

    ''':returns list of sub2 paths'''
    def __get_compositional_candidates_cached__(self, qid, parent_index, topic, interim_ans_set):
        sub2_path = os.path.join(self.sub2_cands_dir, qid + ".json")
        cached = {} # dict of parent chains sub2...
        if os.path.exists(sub2_path):
            cached = json.load(codecs.open(sub2_path, 'r', encoding='utf-8'))
        if parent_index in cached:
            return cached[parent_index]

        all_open_path = os.path.join(self.sub2_openie_dir, qid + ".json")
        all_open_cands = {}
        if os.path.exists(all_open_path):
            all_open_cands = json.load(codecs.open(all_open_path, 'r'))

        cands = []
        visited_entities = set()
        rel_keys = set()
        if interim_ans_set is not None and 0 < len(interim_ans_set) < 800:
            for interim_entity in interim_ans_set:
                if interim_entity in visited_entities:
                    continue
                visited_entities.add(interim_entity)
                if interim_entity in self.forward_cache:
                    cand_preds = self.forward_cache[interim_entity]
                else:
                    #cand_preds = self.corechainGen.main_path_candidates_forward(interim_entity) # not considering reverse edges, perhaps we should?
                    # print parent_index
                    # print interim_entity
                    cand_preds = self.corechainGen.main_path_candidates(interim_entity)  # not considering reverse edges, perhaps we should?
                    # for cand_pred in cand_preds:
                    #      print cand_pred
                    self.forward_cache_fifo.append(interim_entity)
                    self.forward_cache[interim_entity] = cand_preds
                    if len(self.forward_cache_fifo) > self.MAX_SIZE:
                        entity_to_delete = self.forward_cache_fifo.pop(0)
                        if entity_to_delete in self.forward_cache:
                            del self.forward_cache[entity_to_delete]

                if len(cand_preds) > 1000: # assuming fan-out of answers will not be too large
                    continue
                for cand in cand_preds:
                    rels = cand["relations"]
                    if not self.queryGraphGen.__is_valid_rel_path__(rels):
                        continue
                    key = str(rels)
                    if not key in rel_keys:
                        rel_keys.add(key)
                        if len(rels) == 1:
                            ans_entities = self.sparql.eval_one_hop_expansion(interim_entity, rel1=rels[0])
                        else:
                            ans_entities = self.sparql.eval_two_hop_expansion(interim_entity, rel1=rels[0], rel2=rels[1])
                        cands.append({"relations": rels, "is_reverse": False,  "entities": ans_entities, "parent_index": parent_index, "topic_entity": topic, "is_openie": False})
                        visited_entities |= set(ans_entities)  # already visited answers don't revisit

                if topic in all_open_cands:
                    open_cands = all_open_cands[topic]
                    for cand in open_cands:
                        cands.append({"relations": cand['relations'], "unnormalized_relations": cand["unnormalized_relations"], "is_reverse": cand["is_reverse"], "entities": cand['entities'], "parent_index": parent_index, "topic_entity": topic, "is_openie": True})

        new_cands = cached.get(parent_index, [])
        new_cands += cands
        cached[parent_index] = new_cands
        with open(sub2_path, 'w+') as fp:
            json.dump(cached, fp, indent=4)
        # print('compositional candidates {}'.format(cands))
        return cands


    ''':returns a dict of topic2 to list of sub2 paths'''
    def __get_conjunction_candidates__(self, qid, parent_index, topic, interim_entities):
        cands = {}
        sub2_candidates_path = os.path.join(self.sub2_cands_dir, qid + ".json")
        if not os.path.exists(sub2_candidates_path):
            return cands
        sub2_candidates = json.load(codecs.open(sub2_candidates_path, 'r', encoding='utf-8'))
        for topic in sub2_candidates:
            for path in sub2_candidates[topic]:
                path["is_openie"] = False

        all_open_path = os.path.join(self.sub2_openie_dir, qid + ".json")
        if os.path.exists(all_open_path):
            all_open_cands = json.load(codecs.open(all_open_path, 'r'))
            for c in all_open_cands.keys():
                open_cands = all_open_cands[c]
                for open_c in open_cands:
                    open_c["is_openie"] = True
                to_update = sub2_candidates.get(c, [])
                to_update += open_cands
                sub2_candidates[c] = to_update

        sub2_reverse_map = self.complexqEndpoint.__get_entities_reverse_map__(sub2_candidates)
        join_cands = set(interim_entities).intersection(sub2_reverse_map.keys())
        for join_cand in join_cands:
            for rel2_path in sub2_reverse_map[join_cand]:
                # print(rel2_path)
                rel2_path["src"] = "sub2"
                rel2_path["parent_index"] = parent_index

                topic2 = rel2_path["topic_entity"]
                rel2_path_cands = cands.get(topic2, [])
                rel2_path_cands.append(rel2_path)
                cands[topic2] = rel2_path_cands

        return cands

    def __get_sub2_paths__(self, main_cand_dir, qid):
        path = os.path.join(main_cand_dir, qid + "_sub2_data.json")
        path_dict = {}
        if not os.path.exists(path):
            # print('path does not exist for qid ' + qid)
            return path_dict
        file_json = json.load(codecs.open(path, 'r', encoding='utf-8'))
        for topic in file_json.keys():
            path_cands = file_json[topic]
            for path in path_cands:
                path["topic"] = topic
                if path["is_openie"]:
                    # rels = tuple([p.encode('ascii') for p in path['relations']])
                    rels = tuple([p.encode('ascii') for p in path['unnormalized_relations']])
                else:
                    rels = tuple([p for p in path['relations']])
                constraints = []
                if 'constraints' in path:
                    for constraint in path['constraints']:
                        #constraints.append(unicode(constraint['relation'], 'utf-8'))
                        constraints.append(constraint['relation'])
                constraints = tuple(constraints)
                key = topic + "_" + str(rels) + "_" + str(constraints)
                key_paths = path_dict.get(key, [])
                key_paths.append(path)
                path_dict[key] = key_paths
        return path_dict

    def __get_sub2_paths_data__(self, sub2_df, index, parent_index, path_dict):
        #rel_data_keys = sub2_df[(sub2_df["index"] == index) & (sub2_df["parent_index"] == parent_index)].to_records(index=False)
        rel_data_keys = sub2_df[(sub2_df["index"] == index) & (sub2_df["parent_index"] == parent_index)].to_dict('records')
        if len(rel_data_keys) == 0:
            return None, None
        rel_data_key = rel_data_keys[0]
        topic = rel_data_key["topic"]
        if rel_data_key["is_openie"]:
            rels = tuple([rel_data_key['openie']])
            look_up_key = (topic) + "_" + (str(rels)) + "_" + (str(()))
        else:
            look_up_key = topic + "_" + str(rel_data_key["relations"]) + "_" + str(rel_data_key.get("constraints", ()))
        if not look_up_key in path_dict:
            print("path 2 not in the reward file \t" + look_up_key)
            print path_dict
            return topic, None
        rel_data = path_dict[look_up_key]
        return topic, rel_data

    def __get_sub2_paths_data_openie_cached__(self, sub2_df, index, parent_index, is_openie, path_dict):
        #rel_data_keys = sub2_df[(sub2_df["index"] == index) & (sub2_df["parent_index"] == parent_index)].to_records(index=False)
        if is_openie:
            rel_data_keys = sub2_df[(sub2_df["openie_index"] == index) & (sub2_df["parent_index"] == parent_index)].to_dict('records')
        else:
            rel_data_keys = sub2_df[(sub2_df["index"] == index) & (sub2_df["parent_index"] == parent_index)].to_dict('records')
        if len(rel_data_keys) == 0:
            print("path 2 not in look up: {} {}".format(parent_index, index))
            return None, None
        rel_data_key = rel_data_keys[0]
        topic = rel_data_key["topic"]
        if rel_data_key["is_openie"]:
            rels = tuple([rel_data_key['openie']])
            look_up_key = (topic) + "_" + (str(rels)) + "_" + (str(()))
        else:
            look_up_key = topic + "_" + str(rel_data_key["relations"]) + "_" + str(rel_data_key.get("constraints", ()))
        if not look_up_key in path_dict:
            print("path 2 not in the reward file \t" + look_up_key)
            print path_dict
            return topic, None
        rel_data = path_dict[look_up_key]
        return topic, rel_data

    def __get_ques_openie_predictions__(self, sub2_flat_file_path, sub2_open_predictions):
        sub2_df = pd.read_csv(sub2_flat_file_path, sep=',')
        sub2_df = sub2_df[sub2_df['is_openie'] == True]
        sub2_df = sub2_df.drop_duplicates(subset=['parent_index', 'openie_index', 'qid', 'topic', 'openie'])
        print(sub2_df.shape)
        sub2_df['sub2_score'] = sub2_df.apply(lambda x: sub2_open_predictions[str(float(x['openie_index']))], axis=1)
        print 'joined openie predictions to scores'
        sub2_open_qid_predictions = {}
        for qid, group in sub2_df.groupby("qid"):
            group_records = group.reset_index().to_dict('records')
            sub2_open_qid_predictions[qid] = group_records
            print("{}_{}".format(qid, len(group_records)))
        print 'generated qid to openie predictions dict'
        return sub2_open_qid_predictions


    def evaluate_cached_topk(self,sub2_flat_file_path, sub2_kb_predictions, sub2_open_predictions_path, main_data_dir, output_path, write_queries=True, topk=200):
        if os.path.exists(sub2_open_predictions_path):
            sub2_open_predictions = json.load(codecs.open(sub2_open_predictions_path))
            sub2_open_qid_predictions = self.__get_ques_openie_predictions__(sub2_flat_file_path, sub2_open_predictions)
        else:
            sub2_open_qid_predictions = {}
        output_dir = None
        if write_queries:
            output_dir = os.path.join(main_data_dir, "queries")
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        sub2_df = pd.read_csv(sub2_flat_file_path, sep=',')
        counter = 0
        records = []
        for ques_id, question in self.questions_dict.iteritems():
            comp_type = question["compositionality_type"]
            if comp_type == "composition" or comp_type == "conjunction":
                start_time = time.time()
                sub1_path = os.path.join(self.sub1_cands_dir, ques_id + ".json")  # read sub1 json
                if not os.path.exists(sub1_path):
                    continue
                counter += 1
                if counter % 100 == 0:
                    print('{} questions evaluated'.format(counter))
                ques_sub2_kb_predictions = sub2_kb_predictions.get(ques_id, [])
                # ques_sub2_open_predictions = []
                # ques_df = sub2_open_qid_predictions_df[sub2_open_qid_predictions_df["qid"] == ques_id]
                # if len(ques_df) > 0:
                #     ques_sub2_open_predictions = ques_df.to_dict('records')
                ques_sub2_open_predictions = sub2_open_qid_predictions.get(ques_id, [])
                end_time = time.time()
                print("looking up sub2 predictions took {}".format((end_time - start_time)))
                evaluated_records = self.evaluate0_cached_topk(sub2_df, ques_sub2_kb_predictions, ques_sub2_open_predictions,
                                                          main_data_dir, question, output_dir, topk=topk)
                records += evaluated_records
        if len(records) == 0:
            print("evaluation records should not have not been empty")
        preds_df = pd.DataFrame(records)
        preds_df.to_csv(output_path, index=False)

    def evaluate0_cached_topk(self, sub2_df, ques_sub2_kb_predictions, ques_sub2_open_predictions, main_data_dir, question, output_dir, topk=200):
        qid = question["ID"]
        output_path = os.path.join(output_dir, "query_" + qid + ".json")
        if os.path.exists(output_path):
            # TODO RM
            # print 'output path: {}'.format(output_path)
            # try:
            #     records_df = pd.read_csv(output_path)
            #     return records_df.to_dict("records")
            # except:
            #     return []
            print 'reading from cache'
            records_df = pd.read_csv(output_path)
            return records_df.to_dict("records")

        ans_entities = question["Answers"]
        print(qid + "\t" + question["compositionality_type"]) #+ "\t" + str(len(ques_sub2_open_predictions)))
        records = []
        path_dict1 = self.__get_sub1_paths__(qid)
        path_dict2 = self.__get_sub2_paths__(main_data_dir, qid)
        sub1_scores_dict = self.__get_sub1_scores__(main_data_dir, question)
        ques_type = question["compositionality_type"]
        parents_with_sub2 = set()
        start_time = time.time()
        top_k_pairs = {}
        for r in ques_sub2_kb_predictions:
            index = r["index"]
            parent_index = r["parent_index"]
            is_openie = False
            topic1, rel1_data = self.__get_sub1_paths_data__(parent_index, path_dict1)
            if not parent_index in sub1_scores_dict or rel1_data is None:  # could happen if not in top-k
                continue
            pred_key = "{}_{}_{}".format(index, parent_index, is_openie)
            # agg_score = sub1_scores_dict[parent_index]["sub1_score"] + 0.8 * r["sub2_score"]
            agg_score = sub1_scores_dict[parent_index]["sub1_score"] + r["sub2_score"]
            top_k_pairs[pred_key] = agg_score
        for r in ques_sub2_open_predictions:
            index = r["openie_index"]
            parent_index = r["parent_index"]
            is_openie = True
            topic1, rel1_data = self.__get_sub1_paths_data__(parent_index, path_dict1)
            if not parent_index in sub1_scores_dict or rel1_data is None:  # could happen if not in top-k
                continue
            pred_key = "{}_{}_{}".format(index, parent_index, is_openie)
            # agg_score = sub1_scores_dict[parent_index]["sub1_score"] + 0.8 * r["sub2_score"]
            agg_score = sub1_scores_dict[parent_index]["sub1_score"] + r["sub2_score"]
            top_k_pairs[pred_key] = agg_score
        top_k_sorted = sorted(top_k_pairs.items(), key=operator.itemgetter(1), reverse=True)
        top_k_sublist = OrderedDict(top_k_sorted[:min(len(top_k_sorted), topk)])
        # print top_k_sublist
        openie_list = [k for k in top_k_pairs.keys() if k.endswith("True")]
        openie_sublist = [k for k in top_k_sublist.keys() if k.endswith("True")]
        end_time = time.time()
        print("finding top k took {}".format((end_time - start_time)))
        start_time = time.time()
        # print("evaluating top {} of {} with {} of {} open cands".format(len(top_k_sublist), len(top_k_sorted), len(openie_sublist), len(openie_list)))
        for r in ques_sub2_kb_predictions:
            index = r["index"]
            parent_index = r["parent_index"]
            is_openie = False
            pred_key = "{}_{}_{}".format(index, parent_index, is_openie)
            if not pred_key in top_k_sublist:
                continue
            # print pred_key
            parents_with_sub2.add(parent_index)
            topic1, rel1_data = self.__get_sub1_paths_data__(parent_index, path_dict1)
            if not parent_index in sub1_scores_dict or rel1_data is None:  # could happen if not in top-k
                # print "{} kb sub1 not found".format(parent_index)
                continue
            topic2, rel2_data = self.__get_sub2_paths_data_openie_cached__(sub2_df, index, parent_index, False, path_dict2)
            if rel2_data is None:
                # print "{} kb sub2 not found".format(index)
                continue
            sub1_score = sub1_scores_dict[parent_index]["sub1_score"]
            sub2_score = r["sub2_score"]
            entities, query = self.get_entities_and_query(ques_type, topic1, topic2, rel1_data, rel2_data, ans_entities)
            # agg_score = sub1_score + 0.8 * sub2_score
            agg_score = sub1_score + sub2_score

            micro_prec, micro_recall = metricUtils.compute_micro_precision_recall(ans_entities, entities)
            record = self.to_record(qid, parent_index, rel1_data, sub1_score, index, rel2_data[0], sub2_score, entities, micro_prec, micro_recall, agg_score, query)
            records.append(record)
        for r in ques_sub2_open_predictions:
            index = r["openie_index"]
            parent_index = r["parent_index"]
            is_openie = True
            pred_key = "{}_{}_{}".format(index, parent_index, is_openie)
            if not pred_key in top_k_sublist:
                continue
            # print pred_key
            parents_with_sub2.add(parent_index)
            topic1, rel1_data = self.__get_sub1_paths_data__(parent_index, path_dict1)
            if not parent_index in sub1_scores_dict or rel1_data is None:  # could happen if not in top-k
                # print "{} open sub1 not found".format(parent_index)
                continue
            topic2, rel2_data = self.__get_sub2_paths_data_openie_cached__(sub2_df, index, parent_index, True, path_dict2)
            if rel2_data is None:
                # print "{} open sub2 not found".format(index)
                continue
            sub1_score = sub1_scores_dict[parent_index]["sub1_score"]
            sub2_score = r["sub2_score"]
            entities, query = self.get_entities_and_query(ques_type, topic1, topic2, rel1_data, rel2_data,ans_entities)
            # agg_score = sub1_score + 0.8 * sub2_score
            agg_score = sub1_score + sub2_score

            micro_prec, micro_recall = metricUtils.compute_micro_precision_recall(ans_entities, entities)
            record = self.to_record(qid, parent_index, rel1_data, sub1_score, index, rel2_data[0], sub2_score, entities,
                                    micro_prec, micro_recall, agg_score, query)
            records.append(record)

        all_parent_cands = self.test_index[self.test_index["qid"] == qid].to_dict('records')
        for parent_cand in all_parent_cands:
            parent_index = parent_cand["index"]
            if parent_index in parents_with_sub2 or not parent_index in sub1_scores_dict:
                continue
            topic1, rel1_data = self.__get_sub1_paths_data__(parent_index, path_dict1)
            sub1_score = sub1_scores_dict[parent_index]["sub1_score"]
            entities = rel1_data["entities"]
            micro_prec, micro_recall = metricUtils.compute_micro_precision_recall(ans_entities, entities)
            query = ""
            if rel1_data["is_openie"]:
                if rel1_data['is_reverse']:
                    query = "?x" + "\t" + str(rel1_data["relations"][0]) + "\t" + str(topic1) + "\n"
                else:
                    query = str(topic1) + "\t" + str(rel1_data["relations"][0]) + "\t" + "?x" + "\n"
            else:
                r, f = self.__get_rel_chain__(rel1_data.get("relations", []), rel1_data.get("constraints", []))
                if rel1_data["is_reverse"]:
                    core_chain = "?x " + r + " ns:" + topic1 + " ."
                else:
                    core_chain = "ns:" + topic1 + " " + r + " ?x ."
                f += "\n" + self.sparql.__get_entity_filter__("?x")
                query = self.template.substitute(r=core_chain, f=f)
            record = self.to_record(qid, parent_index, rel1_data, sub1_score, None, None, None, entities,
                                    micro_prec, micro_recall, sub1_score, query)
            records.append(record)
        if len(records) == 0:
            print("no evaluation records found in " + qid)
        else:
            preds_df = pd.DataFrame(records)
            preds_df.to_csv(output_path, index=False)
            for r in records:
                if "query" in r:
                    del r["query"]

        end_time = time.time()
        print("evaluating all queries took {}".format((end_time - start_time)))
        return records

    def get_entities_and_query(self, ques_type, topic1, topic2, rel1_data, rel2_data, ans_entities):
        if ques_type == "conjunction":
            return self.evaluate_conjunction(topic1, topic2, rel1_data, rel2_data, ans_entities)
        if ques_type == "composition":
            return self.evaluate_composition(topic1, rel1_data, rel2_data, ans_entities)
        return [], ""


    def to_record(self, qid, parent_index, rel1_data, sub1_score, index, rel2_data, sub2_score, entities, micro_prec, micro_recall, agg_score, query):
        sub2_relation_rels = None
        if rel2_data is not None:
            sub2_relation_rels = rel2_data.get("relations", [])
        sub2_relation_constraints = None
        if rel2_data is not None:
            sub2_relation_constraints = rel2_data.get("constraints", [])
        record = {"qid": qid,
                  "sub1_index": parent_index,
                  "sub1_relation": rel1_data.get("relations", []),
                  "sub1_constraints": rel1_data.get("constraints", []),
                  "sub1_score": sub1_score,
                  "sub2_index": index,
                  "sub2_relation": sub2_relation_rels,
                  "sub2_constraints": sub2_relation_constraints,
                  "sub2_score": sub2_score,
                  "pred_entities": entities,
                  "precision": micro_prec,
                  "recall": micro_recall,
                  "agg_score": agg_score,
                  "query": query
                  }
        return record

    def __get_sub1_scores__(self, main_data_dir, question):
        qid = question["ID"]
        pred_path = os.path.join(main_data_dir, "sub1_pred_" + qid + ".json")
        if not os.path.exists(pred_path):
            # print 'can not located sub1 predictions'
            return {}
        preds = json.load(codecs.open(pred_path, 'r', encoding='utf-8'))
        pred_look_up = {}
        for p in preds:
            pred_look_up[p["index"]] = p
        return pred_look_up

    def evaluate_conjunction(self, topic1, topic2, rel1_data, rel2_data_rows, ans_entities):
        if rel2_data_rows[0]['is_openie']:
            answers = set()
            query = ''
            for rel2_data in rel2_data_rows:
                ans, q = self.evaluate_conjunction0(topic1, topic2, rel1_data, rel2_data)
                for a in ans:
                    answers.add(a)
                query = q
            pred_ans = []
            for p in answers:
                if p in ans_entities:
                    # print p
                    pred_ans.append(p)
            if len(pred_ans) == 0 and len(answers) > 0:
                pred_ans = [list(answers)[0]]
            return list(set(pred_ans)), query
        else:
            return self.evaluate_conjunction0(topic1, topic2, rel1_data, rel2_data_rows[0])

    def evaluate_conjunction0(self, topic1, topic2, rel1_data, rel2_data):
        if not rel1_data['is_openie'] and not rel2_data['is_openie']:
            # print('pure kb')
            core_chain1, filter1 = self.__get_core_chain_cand__(topic1, rel1_data.get("relations", []),
                                                                rel1_data.get("constraints", []),
                                                                rel1_data["is_reverse"], 1)
            core_chain2, filter2 = self.__get_core_chain_cand__(topic2, rel2_data.get("relations", []),
                                                                rel2_data.get("constraints", []),
                                                                rel2_data["is_reverse"], 2)
            r = core_chain1 + "\n" + core_chain2
            f = filter1 + "\n" + filter2 + self.sparql.__get_entity_filter__("?x")
            query = self.template.substitute(r=r, f=f)
            # if rel1_data["approx_label"] == 1 and rel2_data["approx_label"] == 1:
            #     print(query)
            return self.__get_ans__(query), query

        elif rel1_data['is_openie'] and not rel2_data['is_openie']:
            # print('openie kb')
            core_chain2, filter2 = self.__get_core_chain_cand__(topic2, rel2_data.get("relations", []),
                                                               rel2_data.get("constraints", []),
                                                               rel2_data["is_reverse"], 2)
            r = core_chain2
            f = filter2 + self.sparql.__get_entity_filter__("?x")
            query = self.template.substitute(r=r, f=f)
            sub2_ans = self.__get_ans__(query)
            sub1_ans = rel1_data['entities']
            ans = list(set(sub1_ans).intersection(sub2_ans))
            if rel1_data['is_reverse']:
                open_query = "?x" + "\t" + str(rel1_data["relations"][0]) + "\t" + str(topic1) + "\n"
            else:
                open_query = str(topic1) + "\t" + str(rel1_data["relations"][0]) + "\t" + "?x"+ "\n"
            query = query + "\n\n" + open_query
            return ans, query

        elif not rel1_data['is_openie'] and rel2_data['is_openie']:
            # print('kb openie')
            core_chain1, filter1 = self.__get_core_chain_cand__(topic1, rel1_data.get("relations", []),
                                                                rel1_data.get("constraints", []),
                                                                rel1_data["is_reverse"], 1)
            r = core_chain1
            f = filter1 + self.sparql.__get_entity_filter__("?x")
            query = self.template.substitute(r=r, f=f)
            sub1_ans = self.__get_ans__(query)
            sub2_ans = rel2_data['entities']
            ans = list(set(sub2_ans).intersection(sub1_ans))
            if rel2_data['is_reverse']:
                open_query = "?x" + "\t" + str(rel2_data["relations"][0]) + "\t" + str(topic2) + "\n"
            else:
                open_query = str(topic2) + "\t" + str(rel2_data["relations"][0]) + "\t" + "?x" + "\n"
            query = query + "\n\n" + open_query
            # print query
            return ans, query
        elif rel1_data["is_openie"] and rel2_data["is_openie"]:
            # print('openie openie')
            sub1_ans = rel1_data['entities']
            sub2_ans = rel2_data['entities']
            ans = list(set(sub2_ans).intersection(sub1_ans))
            if rel1_data['is_reverse']:
                open_query1 = "?x" + "\t" + str(rel1_data["relations"][0]) + "\t" + str(topic1) + "\n"
            else:
                open_query1 = str(topic1) + "\t" + str(rel1_data["relations"][0]) + "\t" + "?x"+ "\n"
            if rel2_data['is_reverse']:
                open_query2 = "?x" + "\t" + str(rel2_data["relations"][0]) + "\t" + str(topic2) + "\n"
            else:
                open_query2 = str(topic2) + "\t" + str(rel2_data["relations"][0]) + "\t" + "?x" + "\n"
            query = open_query1 + "\n\n" + open_query2
            return ans, query
        return [],""

    def evaluate_composition(self, topic1, rel1_data, rel2_data_rows, ans_entities):
        if rel2_data_rows[0]['is_openie']:
            answers = set()
            query = ''
            for rel2_data in rel2_data_rows:
                ans, q = self.evaluate_composition0(topic1, rel1_data, rel2_data)
                for a in ans:
                    answers.add(a)
                query = q
            pred_ans = []
            for p in answers:
                if p in ans_entities:
                    # print p
                    pred_ans.append(p)
            if len(pred_ans) == 0 and len(answers) > 0:
                pred_ans = [list(answers)[0]]
            return list(set(pred_ans)), query
        else:
            return self.evaluate_composition0(topic1, rel1_data, rel2_data_rows[0])

    def evaluate_composition0(self, topic, rel1_data, rel2_data):
        if not rel1_data["is_openie"] and not rel2_data["is_openie"]:
            # print('pure kb')
            rel1, filter1 = self.__get_rel_chain__(rel1_data["relations"], rel1_data.get("constraints", []), 1, is_composition=True)
            if rel1_data["is_reverse"]: core_chain1 = "?ie " + rel1 + " ns:" + topic + " ."
            else: core_chain1 = "ns:" + topic + " " + rel1 + " ?ie ."

            rel2, filter2 = self.__get_rel_chain__(rel2_data["relations"], rel2_data.get("constraints", []), 2)
            if rel2_data.get("is_reverse", False): core_chain2 = "?x " + rel2 + "?ie ."
            else: core_chain2 = "?ie " + rel2 + " ?x ."
            r = core_chain1 + "\n" + core_chain2
            f = filter1 + "\n" + filter2 + "\n" + self.sparql.__get_entity_filter__("?x")
            query = self.template.substitute(r=r, f=f)
            return self.__get_ans__(query), query
        elif rel1_data['is_openie'] and not rel2_data['is_openie']:
            # print('openie kb')
            topic2 = rel1_data['entities'][0]
            core_chain2, filter2 = self.__get_core_chain_cand__("?ie", rel2_data.get("relations", []),
                                                                rel2_data.get("constraints", []),
                                                                rel2_data["is_reverse"], 2)
            r = core_chain2
            f = filter2 + self.sparql.__get_entity_filter__("?ie")
            query = self.template.substitute(r=r, f=f)
            query = query.replace("?x", "?ie")
            ans = self.__get_ans__(query)
            key = "is_reverse"
            if not key in rel1_data:
                key = "is_reverse:"
            if rel1_data[key]:
                open_query = "?ie" + "\t" + str(rel1_data["relations"][0]) + "\t" + str(topic) + "\n"
            else:
                open_query = str(topic) + "\t" + str(rel1_data["relations"][0]) + "\t" + "?ie"+ "\n"
            query = open_query + "\n\n" + query
            return ans, query
        elif not rel1_data['is_openie'] and rel2_data['is_openie']:
            # print('kb openie')
            core_chain1, filter1 = self.__get_core_chain_cand__(topic, rel1_data.get("relations", []),
                                                                rel1_data.get("constraints", []),
                                                                rel1_data["is_reverse"], 1)
            r = core_chain1
            f = filter1 + self.sparql.__get_entity_filter__("?ie")
            query = self.template.substitute(r=r, f=f)
            query = query.replace("?x", "?ie")
            ans = rel2_data['entities']
            key = "is_reverse"
            if not key in rel2_data:
                key = "is_reverse:"
            if rel2_data[key]:
                open_query = "?x" + "\t" + str(rel2_data["relations"][0]) + "\t" + str("?ie") + "\n"
            else:
                open_query = str("?ie") + "\t" + str(rel2_data["relations"][0]) + "\t" + "?x" + "\n"
            query = query + "\n\n" + open_query
            # print query
            return ans, query
        elif rel1_data["is_openie"] and rel2_data["is_openie"]:
            print('openie openie')
            sub1_ans = rel1_data['entities']
            sub2_ans = rel2_data['entities']
            ans = sub2_ans
            key = "is_reverse"
            if not key in rel1_data:
                key = "is_reverse:"
            if rel1_data[key]:
                open_query1 = "?ie" + "\t" + str(rel1_data["relations"][0]) + "\t" + str(rel1_data["topic"]) + "\n"
            else:
                open_query1 = str(rel1_data["topic"]) + "\t" + str(rel1_data["relations"][0]) + "\t" + "?ie"+ "\n"
            key = "is_reverse"
            if not key in rel2_data:
                key = "is_reverse:"
            if rel2_data[key]:
                open_query2 = "?x" + "\t" + str(rel2_data["relations"][0]) + "\t" + str("?ie") + "\n"
            else:
                open_query2 = str("?ie") + "\t" + str(rel2_data["relations"][0]) + "\t" + "?x" + "\n"
            query = open_query1 + "\n\n" + open_query2
            return ans, query
        return [],""


    def __get_core_chain_cand__(self, topic_entity, relation_path, constraints, is_reverse, chain_id=None):
        rel, filter = self.__get_rel_chain__(relation_path, constraints, chain_id)
        if is_reverse:
            return "?x " + rel + " ns:" + topic_entity + " .", filter
        else:
            return "ns:" + topic_entity + " " + rel + " ?x .", filter

    def __get_rel_chain__(self, relation_path, constraints, chain_id=None, is_composition=False):
        cvt = "?y"
        if chain_id is not None:
            cvt += str(chain_id)

        rel = "ns:" + relation_path[0]
        if len(relation_path) == 2:
            rel = "ns:" + relation_path[0] + " " + cvt + ".\n " + cvt + " ns:" + relation_path[1] + " "
        filter = ""
        if constraints is not None:
            for index, c in enumerate(constraints):
                constraint_value = "ns:" + c["mid"]
                constraint_relation = "ns:" + c["relation"]
                constraint = constraint_relation + " " + constraint_value + ". \n"

                if c["is_ans_constraint"]:
                    if is_composition:
                        filter += "?ie " + constraint
                    else:
                        filter += "?x " + constraint
                elif len(relation_path) == 2:  # make sure these constraints are added to the cvt node
                    filter += cvt + " " + constraint

        return rel, filter

    def __get_ans__(self, query):
        if query in self.query_cache:
            return self.query_cache[query]
        ans = []
        results = self.sparql.execute(query)[u'results'][u'bindings']
        if results is not None and len(results) > 0:
            ans = [self.sparql.remove_uri_prefix(r["x"]["value"]) for r in results]
        self.query_cache_fifo.append(query)
        self.query_cache[query] = ans
        if len(self.query_cache_fifo) > self.MAX_SIZE:
            query_to_delete = self.query_cache_fifo.pop(0)
            if query_to_delete in self.query_cache:
                del self.query_cache[query_to_delete]
        return ans
        # results = self.sparql.execute(query)[u'results'][u'bindings']
        # if results is not None and len(results) > 0:
        #     return [self.sparql.remove_uri_prefix(r["x"]["value"]) for r in results]
        # return []

    def estimate_sub1_model_predictions_quality(self, ques_src, running_dir):
        questions = json.load(open(ques_src))
        true_ct = 0
        total_ct = 0
        for q in questions:
            qid = q["ID"]
            sub1_pred_path = os.path.join(running_dir, "sub1_pred_" + qid + ".json")
            if os.path.exists(sub1_pred_path):
                print(qid)
                total_ct += 1
                topk_results = json.load(open(sub1_pred_path))
                path_dict = self.__get_sub1_paths__(qid)
                has_true = False
                for i, r in enumerate(topk_results):
                    if i <= 5:
                        cand_index = r["index"]
                        topic, rel_data = self.__get_sub1_paths_data__(cand_index, path_dict)
                        if rel_data is None:
                            continue
                        is_true = rel_data["approx_label"]
                        if is_true:
                            has_true= True
                            break
                    else:
                        break
                if has_true:
                    true_ct += 1

        print("{} for {} have true approx labels".format(true_ct, total_ct))

    def annotate_approx_sub2_labels(self, input_path, src_dir):
        questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                sparql_str = q["sparql"]
                ques_path = os.path.join(src_dir, ques_id + "_sub2_data.json")
                if not os.path.exists(ques_path):
                    continue
                print(ques_id)
                pos_label_paths = {}
                main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                for topic in main_entity_paths:
                    pos_paths = []
                    for path in main_entity_paths[topic]:
                        is_match = topic in sparql_str  # topic should be in sparql
                        for r in path['relations']:  # relations should be in sparql
                            if not r in sparql_str:
                                is_match = False
                                break
                        if "constraints" in path:
                            for c in path["constraints"]:
                                if not c["relation"] in sparql_str:
                                    is_match = False
                                    break
                        label = 0
                        if is_match:
                            label = 1
                            path["approx_label"] = label
                            pos_paths.append(path)
                        path["approx_label"] = label
                    pos_label_paths[topic] = pos_paths

                '''this is to propagate the labels to the reverse'''
                for topic in main_entity_paths:
                    pos_paths = pos_label_paths[topic]
                    for path in main_entity_paths[topic]:
                        path_entities = set(path["entities"])
                        label = 0
                        for pos_path in pos_paths:
                            pos_entities = set(pos_path["entities"])
                            if path_entities == pos_entities:
                                label = 1
                                break
                        path["approx_label"] = label

                with open(os.path.join(src_dir, ques_id + "_sub2_data.json"), 'w+') as fp:
                    json.dump(main_entity_paths, fp, indent=4)

    def estimate_sub2_quality(self, ques_src, running_dir):
        questions = json.load(open(ques_src))
        true_ct = 0
        total_ct = 0
        for q in questions:
            qid = q["ID"]
            sub2_path = os.path.join(running_dir, qid + "_sub2_data.json")
            if os.path.exists(sub2_path):
                print(qid)
                has_true = False
                data_json = json.load(codecs.open(sub2_path, encoding="utf-8"))
                if (len(data_json) > 0):
                    total_ct += 1
                for topic in data_json:
                    for rel2_data in data_json[topic]:
                        is_true = rel2_data["approx_label"]
                        if is_true:
                            has_true = True
                            break
                if has_true:
                    true_ct += 1

        print("{} for {} have true approx labels".format(true_ct, total_ct))

    def estimate_sub2_model_predictions_quality(self, sub2_flat_file_path, sub2_preds_path, running_dir):
        self.annotate_approx_sub2_labels(self.ques_src, running_dir)
        sub2_df = pd.read_csv(sub2_flat_file_path, sep=',')
        counter = 0
        correct = 0
        sub2_predictions = json.load(open(sub2_preds_path))
        for ques_key in sub2_predictions.keys():
            if not ques_key in self.questions_dict:
                print('ques key not found')
                continue
            counter += 1
            if counter % 100 == 0:
                print('{} questions evaluated'.format(counter))
            ques_sub2_predictions = sub2_predictions[ques_key]
            path_dict2 = self.__get_sub2_paths__(running_dir, ques_key)
            print(ques_key)
            for i, r in enumerate(ques_sub2_predictions):
                index = r["index"]
                parent_index = int(r["parent_index"])
                topic2, rel2_data = self.__get_sub2_paths_data__(sub2_df, index, parent_index, path_dict2)
                if rel2_data["approx_label"] == 1:
                    correct += 1
                    break
        print("{} correct of {}".format(correct, counter))

    def get_single_ques_queries(self):
        total = 0
        single_ans = 0
        for qid in self.questions_dict:
            question = self.questions_dict[qid]
            if question["compositionality_type"] == "composition" or question["compositionality_type"] == "conjunction":
                total += 1
                if len(question["Answers"]) <= 1:
                    single_ans += 1
        print single_ans
        print total


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

    def get_overall_query_elements_accuracy(self, main_data_dir, src, sub2_flat_file_path):
        src_dir = os.path.join(main_data_dir, "queries")
        sub2_df = pd.read_csv(sub2_flat_file_path, sep=',')
        df = pd.read_csv(src)
        MAX = 25
        correct_ct = 0
        total_ct = 0
        for qid, group in df.groupby("qid"):
            print(qid)
            path_dict1 = self.__get_sub1_paths__(qid)
            path_dict2 = self.__get_sub2_paths__(main_data_dir, qid)
            query_path = os.path.join(src_dir, "query_" + qid + ".json")
            if os.path.exists(query_path):
                group_df = pd.read_csv(query_path)
                group_df = group_df.drop_duplicates(subset='query')
                group_df['sub2_score'].fillna(0.0, inplace=True)
                group_df['agg'] = group_df['sub1_score'] + group_df['sub2_score']
            else:
                group_df = group.reset_index()
                group_df['sub2_score'].fillna(0.0, inplace=True)
                group_df['agg'] = group_df['sub1_score'] + group_df['sub2_score']
            total_ct += 1
            group_df = group_df.sort_values(by=["agg"], ascending=False)
            group_df_sub = group_df.head(min(len(group_df), MAX))
            for i, row in group_df_sub.iterrows():
                sub1_index = row["sub1_index"]
                sub2_index = row["sub2_index"]
                topic1, rel1_data = self.__get_sub1_paths_data__(sub1_index, path_dict1)
                correct1 = rel1_data["approx_label"] == 1
                correct2 = True
                if not np.isnan(sub2_index):
                    topic2, rel2_data = self.__get_sub2_paths_data__(sub2_df, sub1_index, sub2_index, path_dict2)
                    if rel2_data is not None:
                        correct2 = rel2_data.get("approx_label", 0) == 1
                if correct1 and correct2:
                    correct_ct +=1
                    break
        print("{} of {} are correct".format(correct_ct, total_ct))

    def clean_sub2_dir(self):
        for qid, ques in self.questions_dict.iteritems():
            if ques["compositionality_type"] == "composition":
                path = os.path.join(self.sub2_cands_dir, qid + ".json")
                if os.path.exists(path):
                    # print qid
                    os.remove(path)

    def deduplicate_lookup(self, sub2_lookup, dest):
        df = pd.read_csv(sub2_lookup)
        openie_df = df[df['is_openie'] == True]
        openie_df = openie_df.drop_duplicates(subset=['parent_index', 'index', 'qid', 'topic', 'openie'])
        openie_df = openie_df.drop_duplicates(subset=['parent_index', 'openie_index', 'qid', 'topic', 'openie'])
        kb_df = df[df['is_openie'] == False]
        dedup_df = pd.concat([kb_df, openie_df])
        dedup_df = dedup_df.sort_values(by=['parent_index', 'index'])
        dedup_df.to_csv(dest, index=False)


if __name__ == '__main__':
    # INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
    # SUB1_CANDIDATES_DIR = os.path.join(PREFIX, "rewards/test/sub1")
    # SUB2_CANDIDATES_DIR = os.path.join(PREFIX, "rewards/test/sub2")
    # RUNNING_DIR = os.path.join(PREFIX, "debug")
    #
    # FLAT_FILE_PATH1 = os.path.join(RUNNING_DIR, "sub1_lookup.csv")
    # FLAT_FILE_PATH2 = os.path.join(RUNNING_DIR, "sub2_lookup.csv")
    # SUB2_PREDS_PATH = os.path.join(RUNNING_DIR, "qid_to_sub2_res.json")
    # FINAL_RESULTS_PATH = os.path.join(RUNNING_DIR, "final_results.csv")
    #
    epoch_num = '5'
    INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
    SUB1_CANDIDATES_DIR = os.path.join(PREFIX, "rewards/test/sub1")

    # INPUT_PATH = os.path.join(PREFIX, "annotated/dev.json")
    # SUB1_CANDIDATES_DIR = os.path.join(PREFIX, "rewards/dev/sub1")

    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/a5"
    RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_wo_constraints"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_wo_prior"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_wo_attention"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/a5_dev"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/a5_kb_0.75"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/a5_kb_0.9"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/a5_kb_0.5"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_only"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_only2/"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_downsample/kb_0.9"
    # RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_downsample/kb_0.5"

    SUB2_CANDIDATES_DIR = os.path.join(RUNNING_DIR, "sub2_cands")
    FLAT_FILE_PATH1 = os.path.join(RUNNING_DIR, "sub1_lookup.csv")
    FLAT_FILE_PATH2 = os.path.join(RUNNING_DIR, "sub2_lookup.csv")
    SUB2_PREDS_PATH = os.path.join(RUNNING_DIR, "qid_to_sub2_res.json")
    OPENIE_DIR_1 = os.path.join(PREFIX, "stanfordie/all/normalized_cands/train_cands/test_sub1")
    OPENIE_DIR_2 = os.path.join(PREFIX, "stanfordie/all/normalized_cands/train_cands/test_sub2")
    # test_interface = Tester_Interface(INPUT_PATH, FLAT_FILE_PATH1, SUB1_CANDIDATES_DIR, SUB2_CANDIDATES_DIR,
    #                                   OPENIE_DIR_1, OPENIE_DIR_2, sparql_host="141.212.110.80", sparql_port="3093")
    test_interface = Tester_Interface(INPUT_PATH, FLAT_FILE_PATH1, SUB1_CANDIDATES_DIR, SUB2_CANDIDATES_DIR,
                                      OPENIE_DIR_1, OPENIE_DIR_2, sparql_host="141.212.110.193", sparql_port="3095")

    '''Check model quality for sub1 predictions'''
    # test_interface.estimate_sub1_model_predictions_quality(INPUT_PATH, RUNNING_DIR)

    '''Generate sub2 candidates, get their approx labels and check candidate quality'''
    # test_interface.clean_sub2_dir()
    # test_interface.generate_all_sub2_candidates(INPUT_PATH, RUNNING_DIR)
    # test_interface.annotate_approx_sub2_labels(INPUT_PATH, RUNNING_DIR)
    # test_interface.estimate_sub2_quality(INPUT_PATH, RUNNING_DIR)

    '''Check model quality for sub2 predictions'''
    # test_interface.estimate_sub2_model_predictions_quality(FLAT_FILE_PATH2, SUB2_PREDS_PATH, RUNNING_DIR)

    '''Check model quality for sub2 predictions'''
    #sub2_preds = json.load(codecs.open(SUB2_PREDS_PATH, encoding='utf-8'))
    FINAL_RESULTS_PATH = os.path.join(RUNNING_DIR, "9_prediction_v2.csv")
    # FINAL_RESULTS_PATH = os.path.join(RUNNING_DIR, "5_prediction.csv")
    # test_interface.deduplicate_lookup(FLAT_FILE_PATH2, os.path.join(RUNNING_DIR, "sub2_lookup_dedup.csv"))
    # files = os.listdir(RUNNING_DIR)
    # files = [f for f in files if f.startswith('sub2_pred_')]
    # sub2_preds = {}
    # for f in files:
    #     qid = f.replace("sub2_pred_", "").replace(".json", "")
    #     content = json.load(codecs.open(os.path.join(RUNNING_DIR, f)))
    #     sub2_preds[qid] = content


    # test_interface.evaluate_cached_topk(FLAT_FILE_PATH2, sub2_preds,
    #                                     os.path.join(RUNNING_DIR, "openie_index_to_res.json"), RUNNING_DIR,
    #                                     FINAL_RESULTS_PATH)

    print(RUNNING_DIR)
    n_bests = [1]
    for n_best in n_bests:
        print("{}: {}".format(n_best, test_interface.get_average_f1(FINAL_RESULTS_PATH, n_best=n_best)))
