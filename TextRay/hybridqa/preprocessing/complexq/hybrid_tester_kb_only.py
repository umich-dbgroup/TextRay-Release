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

# PREFIX = "/z/zxycarol/ComplexWebQuestions_Resources/ComplexWebQuestions_preprocess"
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

    def __init__(self, ques_src, sub1_flat_file_path, sub1_cands_dir, sub2_cands_dir, sparql_host="141.212.110.80", sparql_port="3093"):
        self.complexqEndpoint = ComplexQuestionEndPoint()
        self.corechainGen = CoreChainGen()
        self.queryGraphGen = QueryGraphGen()

        self.ques_src = ques_src
        self.sub1_flat_file_path = sub1_flat_file_path
        self.sub1_cands_dir = sub1_cands_dir
        self.sub2_cands_dir = sub2_cands_dir
        self.forward_cache = {}
        self.forward_cache_fifo = []
        self.MAX_SIZE = 10000
        self.questions_dict = {}
        self.sparql = sparqlUtils(host=sparql_host, port=sparql_port)
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))

        for q in questions:
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
                cands = self.__get_compositional_candidates_cached__(qid, cand_index, topic, interim_ans_set)
                if len(cands) == 0:
                    if len(rel_data["entities"]) > 0:
                        print("no next cands found for result {}".format(rel_data["relations"]))
                    continue
                for cand in cands:
                    cand["is_openie"] = False
                topic_cands = all_cands.get(topic, [])
                topic_cands += cands
                all_cands[topic] = topic_cands
            elif question["compositionality_type"] == "conjunction":
                # print("{} rel data".format(rel_data["relations"]))
                interim_ans_set = rel_data["entities"]
                print(len(interim_ans_set))
                conj_cands = self.__get_conjunction_candidates__(qid, cand_index, topic, interim_ans_set)
                for topic2, cands in conj_cands.iteritems():
                    for cand in cands:
                        cand["is_openie"] = False
                    topic_cands = all_cands.get(topic2, [])
                    topic_cands += cands
                    all_cands[topic2] = topic_cands
        if len(all_cands.keys()) == 0:
            print("failed to find next expansion candidates for any of the predictions")
        with open(output_path, 'w+') as fp:
            json.dump(all_cands, fp, indent=4)


    def __get_sub1_paths__(self, qid):
        # print('reading query graph 1 for ' + qid)
        path = os.path.join(self.sub1_cands_dir, qid + ".json") # read sub1 json
        path_dict = {}
        if not os.path.exists(path):
            print('path does not exist for qid ' + qid)
            return path_dict
        sub1_paths = json.load(codecs.open(path, 'r', encoding='utf-8'))
        for topic in sub1_paths.keys():
            for path in sub1_paths[topic]:
                path["topic"] = topic
                key = self.__get_path_key__(topic, path)
                path_dict[key] = path
        return path_dict

    def __get_path_key__(self, topic, path):
        rels = tuple([p for p in path['relations']])
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
        look_up_key = topic + "_" + str(rel_data_key["relations"]) + "_" + str(rel_data_key.get("constraints", ()))
        # if look_up_key not in path_dict:
        #     print look_up_key
        #     print path_dict
        rel_data = path_dict.get(look_up_key, None)
        return topic, rel_data

    def __get_interim_entities__(self, topic_entity, rel_data):
        relation_path = rel_data["relations"]
        constraints = rel_data.get("constraints", [])
        is_reverse = rel_data["is_reverse"]
        pre_eval_entities = rel_data["entities"]
        core_chain, filter = self.__get_core_chain_cand__(topic_entity, relation_path, constraints, is_reverse)
        filter += self.sparql.__get_entity_filter__("?x")
        ans = []
        query = self.template.substitute(r=core_chain, f=filter)
        results = self.sparql.execute(query)[u'results'][u'bindings']
        if results is not None and len(results) > 0:
            ans += [self.sparql.remove_uri_prefix(r["x"]["value"]) for r in results]
        if len(ans)==0 and len(pre_eval_entities) > 0:
            print(query)
            print('interim entities set should not be empty')
            print(topic_entity + "\t" + str(relation_path))
        return ans

    def __get_sub2_paths__(self, main_cand_dir, qid):
        path = os.path.join(main_cand_dir, qid + "_sub2_data.json")
        path_dict = {}
        if not os.path.exists(path):
            print('path does not exist for qid ' + qid)
            return path_dict
        file_json = json.load(codecs.open(path, 'r', encoding='utf-8'))
        for topic in file_json.keys():
            path_cands = file_json[topic]
            for path in path_cands:
                path["topic"] = topic
                #rels = tuple([unicode(p, 'utf-8') for p in path['relations']])
                rels = tuple([p for p in path['relations']])
                constraints = []
                if 'constraints' in path:
                    for constraint in path['constraints']:
                        #constraints.append(unicode(constraint['relation'], 'utf-8'))
                        constraints.append(constraint['relation'])
                constraints = tuple(constraints)
                key = topic + "_" + str(rels) + "_" + str(constraints)
                path_dict[key] = path
        return path_dict

    def __get_sub2_paths_data__(self, sub2_df, index, parent_index, path_dict):
        #rel_data_keys = sub2_df[(sub2_df["index"] == index) & (sub2_df["parent_index"] == parent_index)].to_records(index=False)
        rel_data_keys = sub2_df[(sub2_df["index"] == index) & (sub2_df["parent_index"] == parent_index)].to_dict('records')
        if len(rel_data_keys) == 0:
            return None, None
        rel_data_key = rel_data_keys[0]
        topic = rel_data_key["topic"]
        look_up_key = topic + "_" + str(rel_data_key["relations"]) + "_" + str(rel_data_key.get("constraints", ()))
        if not look_up_key in path_dict:
            print("path 2 not in the reward file \t" + look_up_key)
            return topic, None
        rel_data = path_dict[look_up_key]
        return topic, rel_data


    def __get_rel_chain__(self, relation_path, constraints, chain_id=None):
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
                    filter += "?x " + constraint
                elif len(relation_path) == 2:  # make sure these constraints are added to the cvt node
                    filter += cvt + " " + constraint

        return rel, filter

    def __get_core_chain_cand__(self, topic_entity, relation_path, constraints, is_reverse, chain_id=None):
        rel, filter = self.__get_rel_chain__(relation_path, constraints, chain_id)
        if is_reverse:
            return "?x " + rel + " ns:" + topic_entity + " .", filter
        else:
            return "ns:" + topic_entity + " " + rel + " ?x .", filter

    ''':returns list of sub2 paths'''
    def __get_compositional_candidates_cached__(self, qid, parent_index, topic, interim_ans_set):
        sub2_path = os.path.join(self.sub2_cands_dir, qid + ".json")
        cached = {} # dict of parent chains sub2...
        if os.path.exists(sub2_path):
            cached = json.load(codecs.open(sub2_path, 'r', encoding='utf-8'))
        if parent_index in cached:
            return cached[parent_index]

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
                    cand_preds = self.corechainGen.main_path_candidates(interim_entity)  # not considering reverse edges, perhaps we should?
                    self.forward_cache_fifo.append(interim_entity)
                    self.forward_cache[interim_entity] = cand_preds
                    if len(self.forward_cache_fifo) > self.MAX_SIZE:
                        entity_to_delete = self.forward_cache_fifo.pop(0)
                        if entity_to_delete in self.forward_cache:
                            del self.forward_cache[entity_to_delete]

                if len(cand_preds) > 200: # assuming fan-out of answers will not be too large
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
                        cands.append({"relations": rels, "is_reverse:": False,  "entities": ans_entities, "parent_index": parent_index, "topic_entity": topic})
                        visited_entities |= set(ans_entities)  # already visited answers don't revisit

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
        sub2_reverse_map = self.complexqEndpoint.__get_entities_reverse_map__(sub2_candidates)
        join_cands = set(interim_entities).intersection(sub2_reverse_map.keys())
        # print("{} has interim {}".format(topic, interim_entities))
        if len(join_cands) == 0:
            print("{} with parent id {} has no overlap with sub2".format(qid, parent_index))
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

    def evaluate(self, sub2_flat_file_path, sub2_predictions, main_data_dir, output_path, write_queries=True):
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
            #if True:
                sub1_path = os.path.join(self.sub1_cands_dir, ques_id + ".json")  # read sub1 json
                if not os.path.exists(sub1_path):
                    continue
                counter += 1
                if counter % 100 == 0:
                    print('{} questions evaluated'.format(counter))
                ques_sub2_predictions = sub2_predictions.get(ques_id, {})
                evaluated_records = self.evaluate0(sub2_df, ques_sub2_predictions, main_data_dir, question, output_dir)
                records += evaluated_records
        if len(records) == 0:
            print("evaluation records should not have not been empty")
        preds_df = pd.DataFrame(records)
        preds_df.to_csv(output_path, index=False)

    def evaluate0(self, sub2_df, ques_sub2_predictions, main_data_dir, question, output_dir):
        qid = question["ID"]
        output_path = os.path.join(output_dir, "query_" + qid + ".json")
        # if os.path.exists(output_path):
        #     try:
        #         records_df = pd.read_csv(output_path)
        #         return records_df.to_dict("records")
        #     except:
        #         return []
        #     # # TODO RM
        #     # print 'output path: {}'.format(output_path)
        #     # records_df = pd.read_csv(output_path)
        #     # return records_df.to_dict("records")

        ans_entities = question["Answers"]
        print(qid + "\t" + question["compositionality_type"])
        records = []
        path_dict1 = self.__get_sub1_paths__(qid)
        path_dict2 = self.__get_sub2_paths__(main_data_dir, qid)

        sub1_scores_dict = self.__get_sub1_scores__(main_data_dir, question)

        parents_with_sub2 = set()
        for r in ques_sub2_predictions:
            index = r["index"]
            parent_index = r["parent_index"]
            parents_with_sub2.add(parent_index)
            # print(parent_index)
            # print(index)
            topic1, rel1_data = self.__get_sub1_paths_data__(parent_index, path_dict1)
            # print('sub2 index {}'.format(index))
            # print('sub1 index {}'.format(parent_index))
            if not parent_index in sub1_scores_dict: # could happen if not in top-k
                continue
            sub1_score = sub1_scores_dict[parent_index]["sub1_score"]

            if rel1_data is None:
                continue
            topic2, rel2_data = self.__get_sub2_paths_data__(sub2_df, index, parent_index, path_dict2)
            if rel2_data is None:
                continue
            sub2_score = r["sub2_score"]
            ques_type = question["compositionality_type"]
            if ques_type == "conjunction":
                entities, query = self.evaluate_conjunction(topic1, topic2, rel1_data, rel2_data)
            elif ques_type == "composition":
                entities, query = self.evaluate_composition(topic1, rel1_data, rel2_data)
            else:
                entities, query = [], ""

            agg_score = sub1_score + 0.8 * sub2_score

            if len(ans_entities) == 0:
                f1_score = 0,0,0
            else: f1_score = metricUtils.compute_f1(ans_entities, entities)

            record = {"qid": qid,
                      "sub1_index": parent_index,
                      "sub1_relation": rel1_data.get("relations", []),
                      "sub1_constraints": rel1_data.get("constraints", []),
                      "sub1_score": sub1_score,
                      "sub2_index": index,
                      "sub2_relation": rel2_data.get("relations", []),
                      "sub2_constraints": rel2_data.get("constraints", []),
                      "sub2_score": r["sub2_score"],
                      "pred_entities": entities,
                      "precision": f1_score[0],
                      "recall": f1_score[1],
                      "f1_score": f1_score[2],
                      "agg_score": agg_score,
                      "query": query
                      }
            records.append(record)

        # evaluate sub1 cands that didn't have sub2 cands
        #all_parent_c
        # ands = self.test_index[self.test_index["qid"] == qid].to_records(index=False)
        all_parent_cands = self.test_index[self.test_index["qid"] == qid].to_dict('records')
        for parent_cand in all_parent_cands:
            parent_index = parent_cand["index"]
            if parent_index in parents_with_sub2:
                continue
            if not parent_index in sub1_scores_dict: # could happen if not in top-k
                continue
            topic1, rel1_data = self.__get_sub1_paths_data__(parent_index, path_dict1)
            sub1_score = sub1_scores_dict[parent_index]["sub1_score"]
            entities = rel1_data["entities"]
            if len(ans_entities) == 0:
                f1_score = 0,0,0
            else: f1_score = metricUtils.compute_f1(ans_entities, entities)
            r, f = self.__get_rel_chain__(rel1_data.get("relations", []), rel1_data.get("constraints", []))
            if rel1_data["is_reverse"]:
                core_chain = "?x " + r + " ns:" + topic1 + " ."
            else:
                core_chain = "ns:" + topic1 + " " + r + " ?x ."
            f += "\n" + self.sparql.__get_entity_filter__("?x")
            query = self.template.substitute(r=core_chain, f=f)
            record = {"qid": qid,
                      "sub1_index": parent_index,
                      "sub1_relation": rel1_data.get("relations", []),
                      "sub1_constraints": rel1_data.get("constraints", []),
                      "sub1_score": sub1_score,
                      "sub2_index": None,
                      "sub2_relation": None,
                      "sub2_constraints": None,
                      "sub2_score": None,
                      "pred_entities": entities,
                      "precision": f1_score[0],
                      "recall": f1_score[1],
                      "f1_score": f1_score[2],
                      "agg_score": sub1_score,
                      "query": query
                      }
            records.append(record)
        if len(records) == 0:
            print("no evaluation records found in " + qid)
        preds_df = pd.DataFrame(records)
        preds_df.to_csv(output_path, index=False)
        for r in records:
            if "query" in r:
                del r["query"]
        return records


    def evaluate_no_queries(self, sub2_flat_file_path, sub2_predictions, main_data_dir, output_path):
        sub2_df = pd.read_csv(sub2_flat_file_path, sep=',')
        records = []
        counter = 0
        for ques_id, question in self.questions_dict.iteritems():
            comp_type = question["compositionality_type"]
            if comp_type == "composition" or comp_type == "conjunction":
                sub1_path = os.path.join(self.sub1_cands_dir, ques_id + ".json")  # read sub1 json
                if not os.path.exists(sub1_path):
                    continue
                counter += 1
                if counter % 100 == 0:
                    print('{} questions evaluated')
                ques_sub2_predictions = sub2_predictions.get(ques_id, {})
                evaluated_records = self.evaluate0_no_queries(sub2_df, ques_sub2_predictions, main_data_dir, question)
                records += evaluated_records
        if len(records) == 0:
            print("evaluation records should not have not been empty")
        preds_df = pd.DataFrame(records)
        preds_df.to_csv(output_path, index=False)

    def evaluate0_no_queries(self, sub2_df, ques_sub2_predictions, main_data_dir, question):
        ans_entities = question["Answers"]
        qid = question["ID"]
        print(qid + "\t" + question["compositionality_type"])
        records = []
        path_dict1 = self.__get_sub1_paths__(qid)
        path_dict2 = self.__get_sub2_paths__(main_data_dir, qid)

        sub1_scores_dict = self.__get_sub1_scores__(main_data_dir, question)

        parents_with_sub2 = set()
        for r in ques_sub2_predictions:
            index = r["index"]
            parent_index = r["parent_index"]
            parents_with_sub2.add(parent_index)
            # print(parent_index)
            # print(index)
            topic1, rel1_data = self.__get_sub1_paths_data__(parent_index, path_dict1)
            # print('sub2 index {}'.format(index))
            # print('sub1 index {}'.format(parent_index))
            if not parent_index in sub1_scores_dict: # could happen if not in top-k
                continue
            sub1_score = sub1_scores_dict[parent_index]["sub1_score"]

            if rel1_data is None:
                continue
            topic2, rel2_data = self.__get_sub2_paths_data__(sub2_df, index, parent_index, path_dict2)
            if rel2_data is None:
                continue
            sub2_score = r["sub2_score"]
            ques_type = question["compositionality_type"]
            if ques_type == "conjunction":
                entities, query = self.evaluate_conjunction(topic1, topic2, rel1_data, rel2_data)
            elif ques_type == "composition":
                entities, query = self.evaluate_composition(topic1, rel1_data, rel2_data)
            else:
                entities = []

            agg_score = sub1_score + 0.8  * sub2_score

            if len(ans_entities) == 0:
                f1_score = 0,0,0
            else: f1_score = metricUtils.compute_f1(ans_entities, entities)

            record = {"qid": qid,
                      "sub1_index": parent_index,
                      "sub1_relation": rel1_data.get("relations", []),
                      "sub1_constraints": rel1_data.get("constraints", []),
                      "sub1_score": sub1_score,
                      "sub2_index": index,
                      "sub2_relation": rel2_data.get("relations", []),
                      "sub2_constraints": rel2_data.get("constraints", []),
                      "sub2_score": r["sub2_score"],
                      "pred_entities": entities,
                      "precision": f1_score[0],
                      "recall": f1_score[1],
                      "f1_score": f1_score[2],
                      "agg_score": agg_score
                      }
            records.append(record)

        # evaluate sub1 cands that didn't have sub2 cands
        #all_parent_cands = self.test_index[self.test_index["qid"] == qid].to_records(index=False)
        all_parent_cands = self.test_index[self.test_index["qid"] == qid].to_dict('records')
        for parent_cand in all_parent_cands:
            parent_index = parent_cand["index"]
            if parent_index in parents_with_sub2:
                continue
            if not parent_index in sub1_scores_dict: # could happen if not in top-k
                continue
            topic1, rel1_data = self.__get_sub1_paths_data__(parent_index, path_dict1)
            sub1_score = sub1_scores_dict[parent_index]["sub1_score"]
            entities = rel1_data["entities"]
            if len(ans_entities) == 0:
                f1_score = 0,0,0
            else: f1_score = metricUtils.compute_f1(ans_entities, entities)
            record = {"qid": qid,
                      "sub1_index": parent_index,
                      "sub1_relation": rel1_data.get("relations", []),
                      "sub1_constraints": rel1_data.get("constraints", []),
                      "sub1_score": sub1_score,
                      "sub2_index": None,
                      "sub2_relation": None,
                      "sub2_constraints": None,
                      "sub2_score": None,
                      "pred_entities": entities,
                      "precision": f1_score[0],
                      "recall": f1_score[1],
                      "f1_score": f1_score[2],
                      "agg_score": sub1_score
                      }
            records.append(record)
        if len(records) == 0:
            print("no evaluation records found in " + qid)
        return records

    def __get_sub1_scores__(self, main_data_dir, question):
        qid = question["ID"]
        pred_path = os.path.join(main_data_dir, "sub1_pred_" + qid + ".json")
        if not os.path.exists(pred_path):
            print 'can not located sub1 predictions'
            return {}
        preds = json.load(codecs.open(pred_path, 'r', encoding='utf-8'))
        pred_look_up = {}
        for p in preds:
            pred_look_up[p["index"]] = p
        return pred_look_up

    def evaluate_conjunction(self, topic1, topic2, rel1_data, rel2_data):
        core_chain1, filter1 = self.__get_core_chain_cand__(topic1, rel1_data.get("relations", []), rel1_data.get("constraints", []), rel1_data["is_reverse"], 1)
        core_chain2, filter2 = self.__get_core_chain_cand__(topic2, rel2_data.get("relations", []), rel2_data.get("constraints", []), rel2_data["is_reverse"], 2)
        r = core_chain1 + "\n" + core_chain2
        f = filter1 + "\n" + filter2 + self.sparql.__get_entity_filter__("?x")
        query = self.template.substitute(r=r, f=f)
        # if rel1_data["approx_label"] == 1 and rel2_data["approx_label"] == 1:
        #     print(query)
        return self.__get_ans__(query), query

    def evaluate_composition(self, topic, rel1_data, rel2_data):
        rel1, filter1 = self.__get_rel_chain__(rel1_data["relations"], rel1_data.get("constraints", []), 1)
        if rel1_data["is_reverse"]:
            core_chain1 = "?ie " + rel1 + " ns:" + topic + " ."
        else:
            core_chain1 = "ns:" + topic + " " + rel1 + " ?ie ."

        rel2, filter2 = self.__get_rel_chain__(rel2_data["relations"], rel2_data.get("constraints", []), 2)
        if rel2_data.get("is_reverse", False):
            core_chain2 = "?x " + rel2 + "?ie ."
        else:
            core_chain2 = "?ie " + rel2 + " ?x ."
        r = core_chain1 + "\n" + core_chain2
        f = filter1 + "\n" + filter2 + "\n" + self.sparql.__get_entity_filter__("?x")
        query = self.template.substitute(r=r, f=f)
        # if rel1_data["approx_label"] == 1 and rel2_data["approx_label"] == 1:
        #     print(query)
        return self.__get_ans__(query), query

    def __get_ans__(self, query):
        results = self.sparql.execute(query)[u'results'][u'bindings']
        if results is not None and len(results) > 0:
            return [self.sparql.remove_uri_prefix(r["x"]["value"]) for r in results]
        return []

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

    def get_average_f1_from_queries(self, src, running_dir, n_best=25, dest=None):
        src_dir = os.path.join(running_dir, "queries")
        df = pd.read_csv(src)
        all_f1s = []
        all_precisions = []
        all_recalls = []
        top_k_results = []
        top_k_header = ['qid', 'sub1_index', 'sub1_relation', 'sub1_constraints', 'sub1_score', \
                                 'sub2_index', 'sub2_relation', 'sub2_constraints', 'sub2_score', 'pred_entities', \
                                 'precision', 'recall', 'f1_score', 'agg_score', 'agg', 'query']
        for qid, group in df.groupby("qid"):
            # print(qid)
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
                group_df['query'] = ''

            group_df = group_df[top_k_header]
            group_df = group_df.sort_values(by=["agg"], ascending=False)
            group_df_sub = group_df.head(min(len(group_df), n_best))
            group_df_sub_records = group_df_sub.to_records(index=False)
            for r in group_df_sub_records:
                top_k_results.append(r)
            # best_f1 = group_df_sub['f1_score'].max()

            best_row = group_df_sub.ix[group_df_sub['f1_score'].idxmax()]
            best_f1 = best_row['f1_score']
            best_precision = best_row['precision']
            best_recall = best_row['recall']
            if best_f1 == 0.0 or best_recall == 0.0:
                print(best_precision)
                best_precision = 0.0
            all_precisions.append(best_precision)

            all_recalls.append(best_recall)
            all_f1s.append(best_f1)

        print(len(all_f1s))
        print("macro average f1: {}".format(np.mean(all_f1s)))
        print("macro average precision: {}".format(np.mean(all_precisions)))
        print("macro average recall: {}".format(np.mean(all_recalls)))
        if dest is not None and not os.path.exists(dest):
            dest_df = pd.DataFrame.from_records(top_k_results, columns=top_k_header)
            dest_df.to_csv(dest, index=False)


    def get_average_f1(self, src, n_best=25):
        df = pd.read_csv(src)
        # df['sub2_score'].fillna(0.0, inplace=True)
        # df['agg'] = df['sub1_score'] + df['sub2_score']
        all_f1s = []
        # counter = 0
        # for ques_id, question in self.questions_dict.iteritems():
        #     comp_type = question["compositionality_type"]
        #     if comp_type == "composition" or comp_type == "conjunction":
        #         counter += 1
        #         if counter % 100 == 0: print  counter
        #         group_df = df[df["qid"] == ques_id]
        #         if len(group_df) == 0:
        #             all_f1s.append(0.0)
        #             continue
        #         # group_df['sub2_score'].fillna(0.0, inplace=True)
        #         # group_df['agg'] = group_df['sub1_score'] + group_df['sub2_score']
        #         group_df = group_df.sort_values(by=["agg"], ascending=False)
        #         group_df_sub = group_df.head(min(len(group_df), n_best))
        #         best_f1 = group_df_sub['f1_score'].max()
        #         all_f1s.append(best_f1)
        for name, group in df.groupby("qid"):
            # print(name)
            group_df = group.reset_index()
            group_df['sub2_score'].fillna(0.0, inplace=True)
            group_df['agg'] = group_df['sub1_score'] + group_df['sub2_score']
            #group_df = group_df.sort_values(by=["agg_score"], ascending=False)
            group_df = group_df.sort_values(by=["agg"], ascending=False)
            group_df_sub = group_df.head(min(len(group_df), n_best))
            # print(group_df_sub)
            best_f1 = group_df_sub['f1_score'].max()
            all_f1s.append(best_f1)
            # if best_f1 == 0:
            #     print(name)
        macro_avg_f1 = np.mean(all_f1s)
        print(len(all_f1s))
        print(macro_avg_f1)


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
    # epoch_num = '5'
    INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
    #INPUT_PATH = '/z/zxycarol/ComplexWebQuestions_Resources/complex_questions/data/annotated/dev.json'
    SUB1_CANDIDATES_DIR = os.path.join(PREFIX, "rewards/test/sub1")
    # SUB2_CANDIDATES_DIR = os.path.join(PREFIX, "rewards/test/sub2_prior_0313train")
    SUB2_CANDIDATES_DIR = os.path.join(PREFIX, "rewards/test/sub2")
    # RUNNING_DIR = os.path.join(PREFIX, "debug_model_prior_0313train/TH0.5_OPTadam_LR0.001_GA0.1_ATTN500_DO0.0_PRTrue")
    #RUNNING_DIR = os.path.join(PREFIX, "debug")
    RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/model_predictions/0429_kbonly_full/CONSTrue_ATTNTrue_KBTrue_OIEFalse_LR0.0002_LRG0.5_ADO0.0_LDO0.0_APOOLFalse_PTHR2/5"
    FLAT_FILE_PATH1 = os.path.join(RUNNING_DIR, "sub1_lookup.csv")
    FLAT_FILE_PATH2 = os.path.join(RUNNING_DIR, "sub2_lookup.csv")
    SUB2_PREDS_PATH = os.path.join(RUNNING_DIR, "qid_to_sub2_res.json")
    #DEV_PREFIX = '/z/zxycarol/ComplexWebQuestions_Resources/lr_hyper_dev_test/TH0.5_OPTadam_LR0.001_GA0.5_ATTN500_DO0.0_PRTrue'
    test_interface = Tester_Interface(INPUT_PATH, FLAT_FILE_PATH1, SUB1_CANDIDATES_DIR, SUB2_CANDIDATES_DIR, sparql_host="141.212.110.193", sparql_port="3095")
    print(test_interface.sparql.one_hop_expansion(mid='m.0gxnnwq'))
    #test_interface = Tester_Interface(INPUT_PATH, FLAT_FILE_PATH1, SUB1_CANDIDATES_DIR, SUB2_CANDIDATES_DIR)
    # n_bests = [1, 5, 10, 25]
    # for n_best in n_bests:
    #     model_dev_prefix = os.path.join(DEV_PREFIX, epoch_num)
    #     final_path = os.path.join(model_dev_prefix, epoch_num + '_prediction.csv')
    #     print "epoch: {}".format(epoch_num)
    #     f1 = test_interface.get_average_f1(final_path, n_best)


    '''Check model quality for sub1 predictions'''
    # test_interface.estimate_sub1_model_predictions_quality(INPUT_PATH, RUNNING_DIR)

    '''Generate sub2 candidates, get their approx labels and check candidate quality'''
    # test_interface.generate_all_sub2_candidates(INPUT_PATH, RUNNING_DIR)
    # test_interface.annotate_approx_sub2_labels(INPUT_PATH, RUNNING_DIR)
    # test_interface.estimate_sub2_quality(INPUT_PATH, RUNNING_DIR)

    '''Check model quality for sub2 predictions'''
    # test_interface.estimate_sub2_model_predictions_quality(FLAT_FILE_PATH2, SUB2_PREDS_PATH, RUNNING_DIR)

    '''Check model quality for sub2 predictions'''
    FINAL_RESULTS_PATH = "/media/nbhutani/Data/textray_workspace/model_predictions/0429_kbonly_full/CONSTrue_ATTNTrue_KBTrue_OIEFalse_LR0.0002_LRG0.5_ADO0.0_LDO0.0_APOOLFalse_PTHR2/5/5_prediction_f1.csv"
    # sub2_preds = json.load(codecs.open(SUB2_PREDS_PATH, encoding='utf-8'))
    # test_interface.evaluate(FLAT_FILE_PATH2, sub2_preds, RUNNING_DIR, FINAL_RESULTS_PATH)

    '''GET F1 scores'''
    # test_interface.get_average_f1_from_queries(FINAL_RESULTS_PATH, RUNNING_DIR)
    # test_interface.get_average_f1_from_queries(FINAL_RESULTS_PATH, RUNNING_DIR, n_best=25, dest=os.path.join(RUNNING_DIR, "final_top_25_results.csv"))

    '''GET query accuracy'''
    # test_interface.get_overall_query_elements_accuracy(RUNNING_DIR, FINAL_RESULTS_PATH, FLAT_FILE_PATH2)

    # test_interface.evaluate_no_queries(FLAT_FILE_PATH2, sub2_preds, RUNNING_DIR, FINAL_RESULTS_PATH)
    # test_interface.get_average_f1(FINAL_RESULTS_PATH, n_best=10)
