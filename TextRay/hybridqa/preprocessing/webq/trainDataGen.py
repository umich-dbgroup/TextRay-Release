import os
import json
from kbEndPoint.utils.sparql import sparqlUtils
from preprocessing import stringUtils
from preprocessing import metricUtils
import numpy as np
import nltk

import codecs
import pandas as pd

PREFIX = "/Users/funke/webq"

#
# RAW_QUESTION_PATH = os.path.join(PREFIX, "data/webquestions.examples.train.json")
# QUESTION_PATH = os.path.join(PREFIX, "data/train.json")
# SMART_TOPIC_PATH = os.path.join(PREFIX, "SMART/webquestions.examples.train.e2e.top10.filter.tsv")
# ALL_TOPIC_PATH = os.path.join(PREFIX, "topics/train.csv")
# CANDS_DIR = os.path.join(PREFIX, "cands-train")
# CANDS_WTIH_CONSTRAINTS_DIR = os.path.join(PREFIX, "cands_with_constraints-train")
# CANDS_WTIH_CONSTRAINTS_DIR_DEDUP = os.path.join(PREFIX, "cands_with_constraints-train")
# CANDS_WTIH_CONSTRAINTS_RESCALED_DIR = os.path.join(PREFIX, "cands_with_constraints_rescaled-train")

RAW_QUESTION_PATH = os.path.join(PREFIX, "data/webquestions.examples.test.json")
QUESTION_PATH = os.path.join(PREFIX, "data/test.json")
CANDS_DIR = os.path.join(PREFIX, "cands-test")
CANDS_WTIH_CONSTRAINTS_DIR = os.path.join(PREFIX, "cands_with_constraints-test")
CANDS_WTIH_CONSTRAINTS_DIR_DEDUP = os.path.join(PREFIX, "cands_with_constraints-test")
SMART_TOPIC_PATH = os.path.join(PREFIX, "SMART/webquestions.examples.test.e2e.top10.filter.tsv")
ALL_TOPIC_PATH = os.path.join(PREFIX, "topics/test.csv")
CANDS_WTIH_CONSTRAINTS_RESCALED_DIR = os.path.join(PREFIX, "cands_with_constraints_rescaled-test")

ANS_CONSTRAINT_RELATIONS = ["people.person.gender", "common.topic.notable_types", "common.topic.notable_for"]

class Constraint(object):

    def __init__(self, mid, name, relation, is_ans_constraint, surface_form, st_pos, length):
        self.mid = mid
        self.name =name
        self.relation = relation
        self.is_ans_constraint = is_ans_constraint
        self.surface_form = surface_form
        self.st_pos = st_pos
        self.length = length

    def __str__(self):
        return str(self.mid) + " " + str(self.name) + " " + str(self.relation) + " " + str(self.is_ans_constraint)

class Smart_Entity(object):

    def __init__(self, line):
        split_line = line.strip().split('\t')
        self.q_id = split_line[0]
        self.surface_form = split_line[1]
        self.st_pos = int(split_line[2])
        self.length = int(split_line[3])
        mid = split_line[4]
        if mid.startswith('/'):
            mid = mid[1:].replace('/', '.')
        self.mid = mid
        self.e_name = split_line[5]
        self.score = float(split_line[6])

    def __str__(self):
        return str(self.surface_form) + " (" + str(self.mid) + "," + str(self.e_name) + ")"

    class WebQuestionsEndPoint(object):
        def __init__(self):
            self.sparql = sparqlUtils()
            self.topic_entity_dict = {}
            self.cache_maxsize = 10000
            self.cvt_constraints_cache = {}
            self.cvt_constraints_cache_elements_fifo = []
            self.topic_entity_dict = {}
            self.type_dict = {}
            self.type_name_dict = {}
            self.all_path_entity_cache = {}
            self.entity_name_cache={}

        def write_top_entities(self, entity_linking_path, ques_src, dest_topic_path):
            names = ['ques_id', 'mention', 'begin_index', 'length', 'mid', 'name', 'score']
            df = pd.read_csv(entity_linking_path, delimiter='\t', names=names)
            df = df.dropna()
            df['mid'] = df['mid'].apply(lambda mid: mid[1:].replace('/', '.'))
            df = df.sort_values(['ques_id', 'score'], ascending=[True, False])
            df = df.drop_duplicates(subset=['ques_id', 'mid'])
            # df = df.groupby('ques_id').reset_index(drop=True)
            df.to_csv(dest_topic_path, index=False, encoding='utf-8')

        def get_cands(self, ques_src, topic_src, dest_dir):
            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)
            topics_df = pd.read_csv(topic_src)
            file_json = json.load(open(ques_src, 'r'))
            questions = file_json

            for question in questions:
                questionId = question["QuestionId"]
                # if questionId != "WebQTrn-158":
                #     continue
                print questionId
                dest_path = os.path.join(dest_dir, questionId + ".json")
                if os.path.exists(dest_path):
                    continue
                topic_entities = topics_df[topics_df["ques_id"] == questionId].to_dict(orient='records')
                candidates = {}
                for e in topic_entities:
                    topic_entity = e['mid']
                    if topic_entity in self.all_path_entity_cache:
                        cands = self.all_path_entity_cache[topic_entity]
                        print ("found")
                    else:
                        # print(topic_entity)
                        cands = []
                        one_step = self.sparql.one_hop_expansion(topic_entity)
                        for cand in one_step:
                            relations = [cand[0]]
                            cands.append({"relations": relations, "counts": cand[1],
                                          "entities": self.sparql.eval_one_hop_expansion(topic_entity, rel1=cand[0])})
                        two_step = self.sparql.two_hop_expansion(topic_entity)
                        for cand in two_step:
                            relations = [cand[0], cand[1]]
                            cands.append({"relations": relations, "counts": cand[2],
                                          "entities": self.sparql.eval_two_hop_expansion(topic_entity, rel1=cand[0], rel2=cand[1])})
                    candidates[topic_entity] = cands
                    self.all_path_entity_cache[topic_entity] = cands

                with open(dest_path, 'w+') as fp:
                    json.dump(candidates, fp, indent=4)

        '''Add core constraints'''
        def generate_query_graph_cands(self, ques_src, topic_src, core_chain_path, dest_dir):
            topics_df = pd.read_csv(topic_src)
            questions = json.load(open(ques_src, 'r'))

            ans_dict = {}
            ques_str_dict = {}
            for question in questions:
                qid = question["QuestionId"]
                ques_str_dict[qid] = question["ProcessedQuestion"]
                ans_dict[qid] = question['Answers']

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            files = [f for f in os.listdir(core_chain_path) if os.path.isfile(os.path.join(core_chain_path, f))]
            for f in files:
                if ".DS_Store" in f:
                    continue
                q_id = f.replace(".json", "")
                ques_string = ques_str_dict[q_id]
                if os.path.exists(os.path.join(dest_dir, q_id + ".json")):
                    print("exists " + str(q_id))
                    continue
                ques_query_graph_cands = {}
                try:
                    file_json = json.load(open(os.path.join(core_chain_path, f), 'r'))
                except:
                    print(f)
                    continue
                links_df = topics_df[topics_df["ques_id"] == q_id]
                links = links_df.to_dict(orient='records')
                print("Question " + q_id)
                for mid in file_json.keys():
                    topic_entity_names = links_df[links_df['mid'] == mid]['mid'].values
                    if len(topic_entity_names) == 0:
                        print('should have a topic entity name in topics path {}'.format(mid))
                        continue
                    print(mid)
                    topic_entity_name = topic_entity_names[0]
                    answers = ans_dict[q_id]
                    paths = file_json[mid]
                    entity_query_graph_cands = []
                    for path in paths:
                        main_relation = path["relations"]
                        print main_relation
                        constraints = self.__get_constraint_candidates__(ques_string, mid, topic_entity_name, main_relation, links)
                        cands = self.__get_query_graph_cands__(mid, main_relation, constraints, answers)
                        entity_query_graph_cands.extend(cands)
                    ques_query_graph_cands[mid] = entity_query_graph_cands
                    print("topic {} candidates size {}".format(mid, len(entity_query_graph_cands)))
                with open(os.path.join(dest_dir, q_id + ".json"), 'w+') as fp:
                    json.dump(ques_query_graph_cands, fp, indent=4)

        def _add_cvt_to_cache(self, cvt_key, cvt_paths):
            self.cvt_constraints_cache_elements_fifo.append(cvt_key)
            self.cvt_constraints_cache[cvt_key] = cvt_paths
            if len(self.cvt_constraints_cache_elements_fifo) > self.cache_maxsize:
                to_delete = self.cvt_constraints_cache_elements_fifo.pop(0)
                del self.cvt_constraints_cache[to_delete]

        def __get_constraint_candidates__(self, ques_str, topic_entity, topic_entity_name, relation_path, links):
            candidates = []
            for link in links:
                if metricUtils.jaccard_ch(topic_entity_name.lower(), link["mention"].lower()) > 0.4: continue
                if link["mid"] == topic_entity: continue
                if len(relation_path) == 2:
                    rel_key = str(relation_path)
                    if rel_key in self.cvt_constraints_cache:
                        cvt_constraints = self.cvt_constraints_cache[rel_key]
                    else:
                        cvt_constraints = self.sparql.get_all_cvt_constraints(topic_entity, relation_path, False, link["mid"])
                        self._add_cvt_to_cache(rel_key, cvt_constraints)
                    for rel in cvt_constraints:
                        candidates.append(Constraint(link["mid"], link["name"], rel, False, link["mention"], link["begin_index"], link["length"]))
            relation_id = str(relation_path)
            if relation_id in self.type_dict:
                type_mids_rels = self.type_dict[relation_id]
            else:
                type_mids_rels = self.sparql.get_ans_constraint_candidates(topic_entity, relation_path, ANS_CONSTRAINT_RELATIONS, False)
                self.type_dict[relation_id] = type_mids_rels
            for mid in type_mids_rels.keys():
                if mid in self.type_name_dict:
                    names = self.type_name_dict[mid]
                else:
                    names = self.sparql.get_names(mid)
                    self.type_name_dict[mid] = names
                if names is None or len(names) == 0:
                    continue
                match = stringUtils.match_names_to_mention(ques_str, names.split("/"))
                if match is None:
                    continue
                candidates.append(Constraint(mid, names, type_mids_rels[mid], True, match[0], match[1], match[1] + match[2]))
            return candidates

        def __get_query_graph_cands__(self, topic_entity, main_relation, constraints, ans_entities):
            constraint_combinations = self.__get_constraint_combinations__(constraints)
            answer_entities = set(ans_entities)
            cands = []
            for combination in constraint_combinations:
                entity_names = set(self.sparql.eval_all_constraints_named(topic_entity, main_relation, combination, False))
                # entity_names = set()
                # for e in entities:
                #     if e in self.entity_name_cache:
                #         entity_names.add(self.entity_name_cache[e])
                #     else:
                #         entity_name = self.sparql.get_names(e)
                #         self.entity_name_cache[e] = entity_name
                #         entity_names.add(entity_name)


                # common = entities.intersection(answer_entities)
                # reward = float(len(common)) / max(1.0, (len(entities) + len(answer_entities) - len(common)))
                if len(answer_entities)  == 0:
                    reward = 0,0,0
                else:
                    reward = metricUtils.compute_f1(answer_entities, entity_names)
                cand = {"relations": main_relation,
                        "entities": list(entity_names),
                        "constraints": [ob.__dict__ for ob in combination],
                        "reward": reward}
                cands.append(cand)
            return cands

        def __get_constraint_combinations__(self, constraint_candidates):
            if len(constraint_candidates) == 0:
                return [[]]
            elif len(constraint_candidates) == 1:
                return [[], [constraint_candidates[0]]]
            conflicts = self.__get_conflicts__(constraint_candidates)
            constraint_combinations = self.__dfs_search_combinations__(conflicts)
            cand_lists = []
            cand_lists.append([])
            for constraint_combination in constraint_combinations:
                cand_list = [constraint_candidates[i] for i in constraint_combination]
                cand_lists.append(cand_list)
            return cand_lists

        def __get_conflicts__(self, constraint_candidates):
            cand_size = len(constraint_candidates)
            conflict_matrix = []
            # conflict matrix (adjacent format)
            for i in range(cand_size):
                vec = [i]
                for j in range(i + 1, cand_size):
                    cand_1 = constraint_candidates[i]
                    cand_2 = constraint_candidates[j]
                    conflict = cand_1.st_pos <= cand_2.st_pos + cand_2.length \
                               and cand_2.st_pos <= cand_1.st_pos + cand_1.length
                    if conflict: vec.append(j)
                conflict_matrix.append(vec)
            return conflict_matrix

        def __dfs_search_combinations__(self, mat):
            ret_comb_list = []
            n = len(mat)
            status = np.ones((n,), dtype='int32')
            stack = []
            ptr = -1
            while True:
                ptr = self.__nextPick__(ptr, status)
                if ptr == -1:  # backtrace: restore status array
                    if len(stack) == 0: break  # indicating the end of searching
                    pop_idx = stack.pop()
                    for item in mat[pop_idx]: status[item] += 1
                    ptr = pop_idx
                else:
                    stack.append(ptr)
                    for item in mat[ptr]: status[item] -= 1
                    comb = list(stack)
                    ret_comb_list.append(comb)
            return ret_comb_list

        def __nextPick__(self, ptr, status):
            n = len(status)
            for new_ptr in range(ptr + 1, n):
                if status[new_ptr] == 1:
                    return new_ptr
            return -1

        def get_lookup_key(self, topic, rel_data):
            if "constraints" in rel_data:
                look_up_key = topic + "_" + str(rel_data["relations"]) + "_" + str(rel_data["constraints"])
            else:
                look_up_key = topic + "_" + str(rel_data["relations"])
            return look_up_key

        def deduplicate(self, input_path, src_dir, dest_dir):
            questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            for q in questions:
                ques_id = q["QuestionId"]
                ques_path = os.path.join(src_dir, ques_id + ".json")
                if not os.path.exists(ques_path):
                    continue
                print(ques_id)
                main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                look_up_keys = set()
                main_entity_paths_dedup = {}
                for topic in main_entity_paths:
                    paths = []
                    for path in main_entity_paths[topic]:
                        look_up_key = self.get_lookup_key(topic, path)
                        if look_up_key in look_up_keys:
                            continue
                        look_up_keys.add(look_up_key)
                        paths.append(path)
                    print("{} deduplicated to {}".format(len(main_entity_paths[topic]), len(paths)))
                    if len(paths) > 0:
                        main_entity_paths_dedup[topic] = paths
                with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as fp:
                    json.dump(main_entity_paths_dedup, fp, indent=4)

        def add_ids(self, src, dest):
            questions = json.load(codecs.open(src, 'r', encoding='utf-8'))
            to_write_json = []
            for i, ques in enumerate(questions):
                ques_id = "WebQTest-{}".format(i)
                ques["QuestionId"] = ques_id
                ques["ProcessedQuestion"] = ques["utterance"]

                answer_set = set([])
                target_value = ques['targetValue']
                target_value = target_value[6: -1]
                target_value = target_value.replace(') (', ')###(')
                spt = target_value.split('###')
                for item in spt:
                    ans_str = item[13: -1]
                    if ans_str.startswith('"') and ans_str.endswith('"'):
                        ans_str = ans_str[1: -1]
                    if isinstance(ans_str, unicode):
                        ans_str = ans_str.encode('utf-8')
                    answer_set.add(ans_str)

                ques["Answers"] = list(answer_set)
                to_write_json.append(ques)
            with open(dest, 'w+') as fp:
                json.dump(to_write_json, fp, indent=4)

        def reward_with_max_f1(self, main_entity_paths):
            max_reward = 0, 0, 0
            for topic in main_entity_paths:
                for path in main_entity_paths[topic]:
                    if path["reward"][2] > max_reward[2]:
                        max_reward = path["reward"]
            return max_reward

        def rescale_rewards_max(self, src_dir, dest_dir):
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            files = [f for f in os.listdir(src_dir)]
            for f in files:
                if ".DS_Store" in f:
                    continue

                ques_id = f.replace(".json", "")
                #print(ques_id)
                ques_path = os.path.join(src_dir, f)

                main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                max_ques_reward = self.reward_with_max_f1(main_entity_paths)
                for topic in main_entity_paths:
                    for path in main_entity_paths[topic]:
                        path["rescaled_reward"] = [path["reward"][0],  path["reward"][1], path["reward"][2]]
                        if max_ques_reward[2] > 0:
                            reward = path["rescaled_reward"]
                            reward[2] = float(reward[2]) * 1.0 / float(max_ques_reward[2])
                            if max_ques_reward[0] > 0:
                                reward[0] = min(1.0, float(reward[0]) * 1.0 / float(
                                    max_ques_reward[0]))  # hacky way of clipping
                            if max_ques_reward[1] > 0:
                                reward[1] = min(1.0, float(reward[1]) * 1.0 / float(
                                    max_ques_reward[1]))  # hacky way of clipping



                with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as fp:
                    json.dump(main_entity_paths, fp, indent=4)


    if __name__ == '__main__':
        endPoint = WebQuestionsEndPoint()
        # endPoint.add_ids(RAW_QUESTION_PATH, QUESTION_PATH)
        # endPoint.write_top_entities(SMART_TOPIC_PATH, QUESTION_PATH, ALL_TOPIC_PATH)
        # endPoint.get_cands(QUESTION_PATH, ALL_TOPIC_PATH, CANDS_DIR)
        # endPoint.generate_query_graph_cands(QUESTION_PATH, ALL_TOPIC_PATH, CANDS_DIR, CANDS_WTIH_CONSTRAINTS_DIR)
        # endPoint.deduplicate(QUESTION_PATH, CANDS_WTIH_CONSTRAINTS_DIR, CANDS_WTIH_CONSTRAINTS_DIR_DEDUP)
        endPoint.rescale_rewards_max(CANDS_WTIH_CONSTRAINTS_DIR_DEDUP, CANDS_WTIH_CONSTRAINTS_RESCALED_DIR)
