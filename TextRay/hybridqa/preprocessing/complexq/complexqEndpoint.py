import json
import os
import sys
import random

from kbEndPoint.utils.sparql import sparqlUtils
from preprocessing.complexq.corechainGen import CoreChainGen
from preprocessing.complexq.queryGraphGen import Constraint
from preprocessing.complexq.queryGraphGen import QueryGraphGen
from preprocessing.complexq.valueNodesUtils import ValueNodeIdentifier
from preprocessing.complexq import entityUtils
import pandas as pd
import re
from preprocessing import metricUtils
from preprocessing import stringUtils
import nltk
import codecs
import numpy as np

PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess"
#PREFIX = "/Users/funke/Documents/ComplexWebQuestions_preprocess"
MAX_NEGATIVES_FRAC = 5
MAX_DEGREE = 2000
ANS_CONSTRAINT_RELATIONS = ["people.person.gender", "common.topic.notable_types", "common.topic.notable_for"]


# RAW_INPUT_PATH = os.path.join(PREFIX, "annotated_orig/train.json")
# INPUT_PATH = os.path.join(PREFIX, "annotated/train.json")
# RAW_EL_PATH = os.path.join(PREFIX, "el/main/train_el.csv")
# NER_PATH = os.path.join(PREFIX, "named_entities/main/train_ner.csv")
# TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/train_topic.csv")
# SUB1_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub1/train_topic.csv")
# SUB2_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub2/train_topic.csv")
#
#
# CANDIDATE_SUB1_DEST_PATH = os.path.join(PREFIX, "cands/train/sub1")
# CANDIDATE_SUB1_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/train/sub1_with_constraints")
# CANDIDATE_SUB2_DEST_PATH = os.path.join(PREFIX, "cands/train/sub2")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/train/main_with_constraints")


# RAW_INPUT_PATH = os.path.join(PREFIX, "annotated_orig/dev.json")
# INPUT_PATH = os.path.join(PREFIX, "annotated/dev.json")
# RAW_EL_PATH = os.path.join(PREFIX, "el/main/dev_el.csv")
# NER_PATH = os.path.join(PREFIX, "named_entities/main/dev_ner.csv")
# TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/dev_topic.csv")
# SUB1_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub1/dev_topic.csv")
# SUB2_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub2/dev_topic.csv")
#
#
# CANDIDATE_SUB1_DEST_PATH = os.path.join(PREFIX, "cands/dev/sub1")
# CANDIDATE_SUB1_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/dev/sub1_with_constraints")
# CANDIDATE_SUB2_DEST_PATH = os.path.join(PREFIX, "cands/dev/sub2")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/dev/main_with_constraints")


RAW_INPUT_PATH = os.path.join(PREFIX, "annotated_orig/test.json")
INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
RAW_EL_PATH = os.path.join(PREFIX, "el/main/test_el.csv")
NER_PATH = os.path.join(PREFIX, "named_entities/main/test_ner.csv")
TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/test_topic.csv")
SUB1_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub1/test_topic.csv")
SUB2_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub2/test_topic.csv")

TOPIC_PATH_COMQA_TOP3 = os.path.join(PREFIX, "topic_entities/main/test_topic_compqa_3.csv")
TOPIC_PATH_COMQA_TOP5 = os.path.join(PREFIX, "topic_entities/main/test_topic_compqa_5.csv")

CANDIDATE_SUB1_DEST_PATH = os.path.join(PREFIX, "cands/test/sub1")
CANDIDATE_SUB1_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/test/sub1_with_constraints")
CANDIDATE_SUB2_DEST_PATH = os.path.join(PREFIX, "cands/test/sub2")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/test/main_with_constraints")

entity_pattern = re.compile(r'ns:([a-z]\.([a-zA-Z0-9_]+)) ')
ANS_CONSTRAINT_RELATIONS = ["people.person.gender", "common.topic.notable_types", "common.topic.notable_for"]

class ComplexQuestionEndPoint(object):
    def __init__(self):
        self.sparql = sparqlUtils()
        self.cache_maxsize = 10000

        self.corechainGen = CoreChainGen()
        self.queryGraphGen = QueryGraphGen()

        self.connecting_path_entity_cache = {}
        self.path_entity_cache_elements_fifo = []

        self.cache_maxsize = 10000
        self.cvt_constraints_cache = {}
        self.cvt_constraints_cache_elements_fifo = []

        self.type_dict = {}
        self.type_name_dict = {}

    def generate_query_graph_cands1(self,raw_input_path, sub1_topic_src, core_chain_path, dest_dir):
        self.queryGraphGen.generate_query_graph_cands(raw_input_path=raw_input_path, topic_src=sub1_topic_src,
                                                      core_chain_path=core_chain_path,dest_dir=dest_dir)

    def generate_core_chain1(self, raw_input_path, sub1_topic_src, dest_dir):
        self.corechainGen.generate_core_chain(raw_input_path=raw_input_path, topic_src=sub1_topic_src,
                                              dest_dir=dest_dir, is_subsequent=False)

    def generate_core_chain2(self, raw_input_path, sub2_topic_src, dest_dir):
        self.corechainGen.generate_core_chain(raw_input_path=raw_input_path, topic_src=sub2_topic_src,
                                              dest_dir=dest_dir, is_subsequent=True)

    def generate_main_cands(self, raw_input_path, sub2_topic_src, sub1_candidates_dir, dest_dir, sub2_candidates_dir, named_entites_src):
        questions = json.load(codecs.open(raw_input_path, 'r', encoding='utf-8'))
        df = pd.read_csv(sub2_topic_src)
        named_entities_df = pd.read_csv(named_entites_src, names=['ques_id', 'span', 'begin_index', 'end_index', 'node_name', 'ner'], sep='\t')
        no_cands_count = 0
        compositions_count = 0
        conjunction_count = 0
        no_ans_found = 0
        unsupported_type = 0
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for question in questions:
            question_id = question["ID"]
            print(question_id)
            answers = question["Answers"]
            sub1_candidates_path = os.path.join(sub1_candidates_dir, question_id + ".json")
            if not os.path.exists(sub1_candidates_path):
                print("no candidates for sub ques_1 " + question_id)
                no_cands_count += 1
                continue
            if len(answers) == 0:
                print("no answers found")
                no_ans_found += 1
                continue
            dest_path = os.path.join(dest_dir, question_id + ".json")
            if os.path.exists(dest_path):
                continue
            try:
                sub1_candidates = json.load(codecs.open(sub1_candidates_path))
            except:
                no_cands_count += 1
                continue
            compositionality_type = question["compositionality_type"]

            if compositionality_type == "composition":
                print("composition")
                compositions_count += 1
                self.compose(sub1_candidates, answers, dest_path)
            elif compositionality_type == "conjunction":
                print("conjunction")
                conjunction_count += 1
                sub2_named_entities = named_entities_df[(named_entities_df["ques_id"] == question_id) & (named_entities_df["ner"] == "DATE")]["node_name"].values
                sub2_candidates_path = os.path.join(sub2_candidates_dir, question_id + ".json")
                sub2_candidates = None
                if os.path.exists(sub2_candidates_path):
                    sub2_candidates = json.load(codecs.open(sub2_candidates_path))
                if (sub2_candidates is None or len(sub2_candidates) == 0) and (sub2_named_entities is None or len(sub2_named_entities == 0)):
                    no_cands_count += 1
                    print('no second path for: ' + question_id)
                    self.merge_with_empty(sub1_candidates, answers, dest_path)
                    continue
                topic_merged_cands = {}
                named_merged_cands = {}
                # print(len(sub2_named_entities))
                if sub2_candidates is not None and len(sub2_candidates) > 0:
                    topic_merged_cands = self.merge(sub1_candidates, sub2_candidates, answers, dest_path=None)
                if sub2_named_entities is not None and len(sub2_named_entities) > 0:
                    named_merged_cands = self.merge_with_named(sub1_candidates, sub2_named_entities, answers, dest_path=None)
                merged_cands = self.unify(topic_merged_cands, named_merged_cands)
                with open(os.path.join(dest_path), 'w+') as fp:
                    json.dump(merged_cands, fp, indent=4)
                # sub2_topic_entities = df[df["ques_id"] == question_id]["mid"].values
                # if sub2_topic_entities is None or len(sub2_topic_entities) == 0:
                #     no_cands_count += 1
                #     print('no topic entity: ' + question_id)
                #     continue
                # else:
                #     sub2_candidates_path = os.path.join(sub2_candidates_dir, question_id + ".json")
                #     sub2_candidates = None
                #     if os.path.exists(sub2_candidates_path):
                #         sub2_candidates = json.load(codecs.open(sub2_candidates_path))
                #     #
                #     # if sub2_candidates is None or len(sub2_candidates) == 0:
                #     #     no_cands_count += 1
                #     #     print('no sub2 cands: ' + question_id)
                #     #     continue
                #     # self.merge(sub1_candidates, sub2_candidates, answers, dest_path)
            else:
                print("unsupported question type")
                unsupported_type += 1

        print "compositions: " + str(compositions_count)
        print "conjunctions: " + str(conjunction_count)
        print "no_cands_count: " + str(no_cands_count)
        print "no_ans_count: " + str(no_ans_found)
        print "unsupported type: " + str(unsupported_type)

    def compose(self, sub1_candidates, ground_answers, dest_path):
        candidates = {}
        # print("sub1 candidates: " + str(len(sub1_candidates)))
        for sub1_topic_entity in sub1_candidates.keys():
            for relation_path1 in sub1_candidates[sub1_topic_entity]:
                relation_path1['src'] = 'sub1'
                if not self.queryGraphGen.__is_valid_rel_path__(relation_path1['relations']):
                    continue
                if not "is_reverse" in relation_path1:
                    relation_path1["is_reverse"] = False
                rel_counts = self.sparql.get_interim_size(sub1_topic_entity,  relation_path1['relations'], relation_path1["is_reverse"])
                if int(rel_counts) > 2000 or int(rel_counts) == 0:
                    continue
                max_reward = 0, 0, 0
                positive_cands = []
                negative_cands = []
                cands = self.__connecting_paths__(sub1_topic_entity, relation_path1, ground_answers)
                # print("{} with path {} has len {}".format(sub1_topic_entity, relation_path1['relations'], len(cands)))
                for cand in cands:
                    if len(ground_answers) == 0:
                        reward = 0,0,0
                    else : reward = metricUtils.compute_f1(ground_answers, cand["entities"])
                    cand["reward"] = reward
                    cand["src"] = "sub2"
                    if reward[2] == 0: negative_cands.append(cand)
                    else:
                        positive_cands.append(cand)
                        #print(cand)
                    if reward[2] > max_reward[2]:
                        max_reward = reward

                relation_path1["reward"] = max_reward
                # if max_reward[2] > 0:
                #     print("{} with {} has max reward {}".format(sub1_topic_entity, relation_path1['relations'], max_reward))
                # else:
                #     print("{} with {} has no max reward {}".format(sub1_topic_entity, relation_path1['relations'], max_reward))
                # sample the negatives
                random.shuffle(negative_cands)
                negative_cands_sample = negative_cands[0:min(len(positive_cands) * MAX_NEGATIVES_FRAC, len(negative_cands))]

                # we want to use topic entity as the key for the second main path, and not the interim entity
                entity_cands = []
                if sub1_topic_entity in candidates: entity_cands = candidates[sub1_topic_entity]
                entity_cands += [relation_path1] + positive_cands + negative_cands_sample
                candidates[sub1_topic_entity] = entity_cands
        # print candidates
        with open(os.path.join(dest_path), 'w+') as fp:
            json.dump(candidates, fp, indent=4)

    def __connecting_paths__(self, topic_entity, relation_path, answers):
        visited_answers = set() # do not re-evalute answers that share the same path
        cands = []
        rel_keys = set()
        for answer in answers:
            # print str(relation_path['relations']) + "\t"  + answer + "\t" + str(len(answers))
            if answer in visited_answers:
                continue
            visited_answers.add(answer)
            main_relation = relation_path['relations']
            key = topic_entity + "_" + str(main_relation) + "_" + answer
            if key in self.connecting_path_entity_cache:
                paths = self.connecting_path_entity_cache[key]
            else:
                paths = self.sparql.get_connecting_path(topic_entity, relation_path, answer)
                self.__add_result_to_cache__(key, paths)
            if len(paths) == 0: # if a path does not exist to an ans entity, invalidate the ans set
                return cands
            # print len(paths)
            for path in paths:
                rel_path = path["relations"]
                if not self.queryGraphGen.__is_valid_rel_path__(rel_path):
                    continue
                rel_key = str(main_relation + rel_path)
                if rel_key in rel_keys:
                    continue
                rel_keys.add(rel_key)
                path_ans = self.sparql.evaluate_connecting_path(topic_entity, relation_path, path)
                # if len(path_ans) == 0:
                #     print key
                cands.append({"relations": path["relations"], \
                              "entities": path_ans,
                              "constraints": [],
                              "is_reverse": path["is_reverse"]
                              })
                visited_answers |= set(path_ans)  # already visited answers don't revisit
                for path_ans_entity in path_ans: # add the path answers to the cache
                    path_key = topic_entity + "_" + str(main_relation) + "_" + path_ans_entity
                    self.__add_result_to_cache__(path_key, paths)


        # print(str(relation_path['relations']) + "\t" + str(len(cands)))
        return cands

    def __add_result_to_cache__(self, key, paths):
        self.path_entity_cache_elements_fifo.append(key)
        self.connecting_path_entity_cache[key] = paths
        if len(self.path_entity_cache_elements_fifo) > self.cache_maxsize:
            to_delete = self.path_entity_cache_elements_fifo.pop(0)
            if to_delete in self.connecting_path_entity_cache:
                del self.connecting_path_entity_cache[to_delete]

    def merge_with_empty(self, sub1_candidates, ground_answers, dest_path):
        for topic in sub1_candidates.keys():
            for rel in sub1_candidates[topic]:
                if len(ground_answers) == 0:
                    reward = 0,0,0
                else: reward = metricUtils.compute_f1(ground_answers, rel["entities"])
                rel["reward"] = reward
                rel["src"] = "sub1"
                # print("{} with {} has reward {} but not future rewards".format(topic, rel['relations'], reward))
        if dest_path:
            with open(os.path.join(dest_path), 'w+') as fp:
                json.dump(sub1_candidates, fp, indent=4)
        # print cands
        return sub1_candidates

    def merge(self, sub1_candidates, sub2_candidates, ground_answers, dest_path=None):
        # print("sub1 candidates: " + str(len(sub1_candidates)))
        # print("sub2 candidates: " + str(len(sub2_candidates)))

        sub1_reverse_map = self.__get_entities_reverse_map__(sub1_candidates)
        sub2_reverse_map = self.__get_entities_reverse_map__(sub2_candidates)

        candidates = {}
        for topic in sub1_candidates.keys():
            for rel in sub1_candidates[topic]:
                if len(ground_answers) == 0:
                    reward = 0,0,0
                else: reward = metricUtils.compute_f1(ground_answers, rel["entities"])
                rel["reward"] = reward
                rel["src"] = "sub1"
                to_update = candidates.get(topic, {})
                rel_key = topic + "_" + str(rel["relations"]) + ":" + str(rel["constraints"])
                to_update[rel_key] = rel
                candidates[topic] = to_update

        for topic in sub2_candidates.keys():
            for rel in sub2_candidates[topic]:
                # reward = metricUtils.compute_f1(ground_answers, rel["entities"])
                if len(ground_answers) == 0:
                    reward = 0,0,0
                else: reward = metricUtils.compute_f1(ground_answers, rel["entities"])
                rel["reward"] = reward
                rel["src"] = "sub2"
                if not "constraints" in rel: rel["constraints"] = []
                to_update = candidates.get(topic, {})
                rel_key = topic + "_" + str(rel["relations"])
                to_update[rel_key] = rel
                candidates[topic] = to_update

        join_cands = set(sub1_reverse_map.keys()).intersection(sub2_reverse_map.keys())

        rel_cache = set()
        for join_cand in join_cands:
            #print('has join cands')
            rel1_paths = sub1_reverse_map[join_cand]
            rel2_paths = sub2_reverse_map[join_cand]
            for rel1_path in rel1_paths:
                for rel2_path in rel2_paths:
                    rel_1 = rel1_path["relations"]
                    rel_2 = rel2_path["relations"]
                    if str(rel_1) == str(rel_2):  # can't have the same relation in conjunctions?
                        continue
                    if not self.queryGraphGen.__is_valid_rel_path__(rel_1) or not self.queryGraphGen.__is_valid_rel_path__(rel_2):
                        continue
                    topic1 = rel1_path["topic_entity"]
                    topic2 = rel2_path["topic_entity"]
                    if topic1 == topic2:
                        continue
                    constraints_1 = rel1_path["constraints"]
                    constraints_2 = []
                    rel1_path["is_reverse"] = rel1_path.get("is_reverse", False)
                    rel2_path["is_reverse"] = rel2_path.get("is_reverse", False)
                    key = topic1 + "_" + str(rel_1) + "_" + str(constraints_1) + ":" + topic2 + "_" + str(rel_2) + "_" + str(constraints_2)
                    if not key in rel_cache:
                        rel_cache.add(key)
                        ans_set = list(set(rel1_path["entities"]).intersection(rel2_path["entities"]))
                        if ans_set is not None and len(ground_answers) > 0:
                            reward = metricUtils.compute_f1(ground_answers, ans_set)
                            # update rewards for rels in the set
                            rel_key = topic1 + "_" + str(rel_1) + ":" + str(constraints_1)
                            prev_reward = candidates[topic1][rel_key]["reward"]
                            if prev_reward[2] < reward[2]:
                                candidates[topic1][rel_key]["reward"] = reward
                                candidates[topic1][rel_key]["src"] = "sub1"
                                # print("sub1 {} with {} has reward {}".format(topic1, rel_key, reward))

                            # update rewards for rels in the set
                            rel_key = topic2 + "_" + str(rel_2)
                            prev_reward = candidates[topic2][rel_key]["reward"]
                            if prev_reward[2] < reward[2]:
                                candidates[topic2][rel_key]["reward"] = reward
                                candidates[topic2][rel_key]["src"] = "sub2"
                                # print("sub2 {} with {} has reward {}".format(topic2, rel_key, reward))
        pos_count = 0
        cands = {}
        for topic in candidates.keys():
            rel_dict = candidates[topic]
            rels = []
            for rel_key in rel_dict.keys():
                rel_val = rel_dict[rel_key]
                if rel_val["reward"][2] > 0:
                    # print("final {} with {} has reward {}".format(topic, rel_val['relations'], rel_val["reward"]))
                    pos_count += 1
                rels.append(rel_val)
            cands[topic] = rels

        # print cands
        print "conjunction positive count " + str(pos_count)

        if dest_path:
            with open(os.path.join(dest_path), 'w+') as fp:
                json.dump(cands, fp, indent=4)
        # print cands
        return cands


    #TODO:: test before evluating main
    def merge_with_named(self, sub1_candidates, sub2_named_entities, ground_answers, dest_path=None):
        # print("sub1 candidates: " + str(len(sub1_candidates)))
        # print("sub2 candidates: " + str(len(sub2_candidates)))
        candidates = {}
        for topic in sub1_candidates.keys():
            for rel in sub1_candidates[topic]:
                # print('{} topic {}'.format(topic, rel))
                if not self.queryGraphGen.__is_valid_rel_path__(rel['relations']):
                    continue
                if len(ground_answers) == 0:
                    reward = 0,0,0
                else: reward = metricUtils.compute_f1(ground_answers, rel["entities"])
                rel["reward"] = reward
                rel["src"] = "sub1"
                to_update = candidates.get(topic, {})
                rel_key = topic + "_" + str(rel["relations"]) + ":" + str(rel["constraints"])
                to_update[rel_key] = rel
                candidates[topic] = to_update
                # print(rel)
                for sub2_named_entity in sub2_named_entities:
                    # print sub2_named_entity
                    rel2_paths = self.sparql.get_conjunction_path(topic, rel, sub2_named_entity)
                    if len(rel2_paths) > 0:
                        # print(rel2_paths)
                        for rel2 in rel2_paths:
                            conjunction_ans_set = self.sparql.evaluate_conjunction_path(topic, rel, rel2, sub2_named_entity)
                            print(len(conjunction_ans_set))
                            if conjunction_ans_set is not None and len(ground_answers) > 0:
                                reward = metricUtils.compute_f1(ground_answers, conjunction_ans_set)
                                prev_reward = candidates[topic][rel_key]["reward"] # update rewards for rels in the set
                                if prev_reward[2] < reward[2]:
                                    candidates[topic][rel_key]["reward"] = reward
                                    candidates[topic][rel_key]["src"] = "sub1"
        cands = {}
        for topic in candidates.keys():
            rel_dict = candidates[topic]
            rels = []
            for rel_key in rel_dict.keys():
                rel_val = rel_dict[rel_key]
                rels.append(rel_val)
            cands[topic] = rels

        #print cands
        if dest_path:
            with open(os.path.join(dest_path), 'w+') as fp:
                json.dump(cands, fp, indent=4)
        # print cands
        return cands

    #TODO: debug when evaluating main paths
    def unify(self, merged_with_topics, merged_with_named):
        if merged_with_topics is None or len(merged_with_topics) == 0:
            return merged_with_named
        if merged_with_named is None or len(merged_with_named) == 0:
            return merged_with_topics
        candidates = {}
        for topic in merged_with_topics.keys():
            for rel in merged_with_topics[topic]:
                to_update = candidates.get(topic, {})
                rel_key = topic + "_" + str(rel["relations"]) + ":" + str(rel["constraints"])
                to_update[rel_key] = rel
                candidates[topic] = to_update

        for topic in merged_with_named.keys():
            for rel in merged_with_named[topic]:
                curr_reward = rel["reward"]
                rel_key = topic + "_" + str(rel["relations"]) + ":" + str(rel["constraints"])
                if not rel_key in candidates.get(topic, {}):
                    candidates[topic][rel_key] = rel
                else:
                    prev_reward = candidates[topic][rel_key]["reward"]  # update rewards for rels in the set
                    if prev_reward[2] < curr_reward[2]:
                        candidates[topic][rel_key]["reward"] = curr_reward
                        candidates[topic][rel_key]["src"] = "sub1"
        cands = {}
        for topic in candidates.keys():
            rel_dict = candidates[topic]
            rels = []
            for rel_key in rel_dict.keys():
                rel_val = rel_dict[rel_key]
                rels.append(rel_val)
            cands[topic] = rels
        return cands


    def __rel_as_key__(self, topic, rel):
        return topic + "_" + str(rel["relations"]) + ":" + str(rel["constraints"])

    def __get_entities_reverse_map__(self, candidates):
        entities_reverse_map = {}
        for topic_entity in candidates.keys():
            for entry in candidates[topic_entity]:
                ans_entities = entry["entities"]
                for ans in ans_entities:
                    to_update = entities_reverse_map.get(ans, [])
                    entry["topic_entity"] = topic_entity
                    to_update.append(entry)
                    entities_reverse_map[ans] = to_update
        return entities_reverse_map

    def evaluate_main_cands(self, raw_input_path, dest_dir):
        questions = json.load(codecs.open(raw_input_path, 'r', encoding='utf-8'))
        conjunctions = 0
        composition = 0
        total = 0
        total_conjunction = 0
        total_composition = 0
        for q in questions:
            ques_id = q["ID"]
            print ques_id
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                if not os.path.exists(os.path.join(dest_dir, ques_id + ".json")):
                    continue
                total += 1
                if type == "conjunction":
                    total_conjunction += 1
                if type == "composition":
                    total_composition += 1
                #print(ques_id)
                entity_paths = json.load(codecs.open(os.path.join(dest_dir, ques_id + ".json"), 'r', encoding='utf-8'))
                has_positive_example = False
                for entity_key in entity_paths.keys():
                    paths = entity_paths[entity_key]
                    has_pos = False
                    for p in paths:
                        if p["reward"][2] > 0:
                            has_pos = True
                            break
                    if has_pos:
                        has_positive_example = True
                        break
                if has_positive_example:
                    if type == "conjunction":
                        conjunctions += 1
                    elif type == "composition":
                        composition += 1
                else:
                    if type == "conjunction":
                        print(ques_id)
        print "conjunctions with rewards " + str(conjunctions)
        print "conjunctions total " + str(total_conjunction)
        print "composition with rewards " + str(composition)
        print "composition total " + str(total_composition)
        print total

    def estimate_cand_quality(self,raw_input_path, dest_dir):
        questions = json.load(codecs.open(raw_input_path, 'r', encoding='utf-8'))
        true_pos_count = 0
        total_count = 0
        for q in questions:
            ques_id = q["ID"]
            sparql_str = q["sparql"]
            print ques_id
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                total_count += 1
                if not os.path.exists(os.path.join(dest_dir, ques_id + ".json")):
                    continue
                entity_paths = json.load(codecs.open(os.path.join(dest_dir, ques_id + ".json"), 'r', encoding='utf-8'))
                positive_paths = []
                for entity_key in entity_paths.keys():
                    paths = entity_paths[entity_key]
                    for p in paths:
                        if p["reward"][2] > 0:
                            p["topic"] = entity_key
                            positive_paths.append(p)
                true_positives = []
                for p in positive_paths:
                    is_true_positive = ComplexQuestionEndPoint.is_positive_path(sparql_str, p)
                    if is_true_positive:
                        true_positives.append(p)
                print(len(true_positives))
                if len(true_positives) >= 1:
                    true_pos_count += 1
        print true_pos_count
        print total_count

    @staticmethod
    def is_positive_path(sparql_str, p):
        entity = p["topic"]
        rels = p["relations"]
        cons = p["constraints"]
        if not entity in sparql_str:
            return False
        for rel in rels:
            if not rel in sparql_str:
                return False
        for c in cons:
            c_rel = c["relation"]
            if not c_rel in sparql_str:
                return False
        return True

if __name__ == '__main__':
    candGen = ComplexQuestionEndPoint()

    #valueNodeIdentifier = ValueNodeIdentifier()
    #valueNodeIdentifier.find_named_entities(INPUT_PATH, NER_PATH)

    # Generate topic entities for main question and sub questions
    # entityUtils.write_topic_entity_candidates(INPUT_PATH, RAW_EL_PATH, TOPIC_PATH)
    # entityUtils.write_sub_topic_entity_candidates(INPUT_PATH, "split_part1", TOPIC_PATH, SUB1_TOPIC_PATH)
    # entityUtils.write_sub_topic_entity_candidates(INPUT_PATH, "split_part2", TOPIC_PATH, SUB2_TOPIC_PATH)
    # entityUtils.eval_topic_entity_candidates(INPUT_PATH, TOPIC_PATH)

    # Generate path candidates for sub question_1
    # candGen.generate_core_chain1(RAW_INPUT_PATH, SUB1_TOPIC_PATH, CANDIDATE_SUB1_DEST_PATH)
    # candGen.corechainGen.evaluate_core_chain(RAW_INPUT_PATH, CANDIDATE_SUB1_DEST_PATH)
    # candGen.generate_query_graph_cands1(RAW_INPUT_PATH, SUB1_TOPIC_PATH, CANDIDATE_SUB1_DEST_PATH, CANDIDATE_SUB1_WITH_CONSTRAINTS_DEST_PATH)

    # Generate path candidates for sub question_2
    # candGen.generate_core_chain2(RAW_INPUT_PATH, SUB2_TOPIC_PATH, CANDIDATE_SUB2_DEST_PATH)
    # candGen.corechainGen.evaluate_core_inference_candidates(RAW_INPUT_PATH, CANDIDATE_SUB2_DEST_PATH)

    # Generate main candidates
    candGen.generate_main_cands(INPUT_PATH, SUB2_TOPIC_PATH, CANDIDATE_SUB1_WITH_CONSTRAINTS_DEST_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH, CANDIDATE_SUB2_DEST_PATH, NER_PATH)
    # candGen.evaluate_main_cands(RAW_INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH)

    # candGen.estimate_cand_quality(RAW_INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH)