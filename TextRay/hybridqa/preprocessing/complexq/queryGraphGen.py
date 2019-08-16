import json
import pandas as pd
import codecs
import os
import numpy as np
from preprocessing import metricUtils
from preprocessing import stringUtils
from kbEndPoint.utils.sparql import sparqlUtils
ANS_CONSTRAINT_RELATIONS = ["people.person.gender", "common.topic.notable_types", "common.topic.notable_for"]

class Constraint(object):

    def __init__(self, mid, name, relation, is_ans_constraint, surface_form, start_index, end_index):
        self.mid = mid
        self.name =name
        self.relation = relation
        self.is_ans_constraint = is_ans_constraint
        self.surface_form = surface_form
        self.start_index = start_index
        self.end_index = end_index

    def __str__(self):
        return str(self.mid) + " " + str(self.name) + " " + str(self.relation) + " " + str(self.is_ans_constraint)

class QueryGraphGen(object):
    def __init__(self):
        self.sparql = sparqlUtils()
        self.cache_maxsize = 10000
        self.cvt_constraints_cache = {}
        self.cvt_constraints_cache_elements_fifo = []
        self.topic_entity_dict = {}
        self.type_dict = {}
        self.type_name_dict = {}

    def generate_query_graph_cands(self, raw_input_path, topic_src, core_chain_path, dest_dir):
        questions = json.load(codecs.open(raw_input_path, 'r', encoding='utf-8'))
        links_df = pd.read_csv(topic_src)
        ques_str_dict = {}

        for question in questions:
            questionId = question["ID"]
            ques_str_dict[questionId] = question["question"]

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        files = [f for f in os.listdir(core_chain_path) if os.path.isfile(os.path.join(core_chain_path, f))]
        for f in files:
            q_id = f.replace(".json", "")
            print(q_id)
            if os.path.exists(os.path.join(dest_dir, q_id + ".json")):
                print("exists " + str(q_id))
                continue
            try:
                file_json = json.load(open(os.path.join(core_chain_path, f), 'r'))
            except:
                print(f)
                continue
            links = links_df[links_df["ques_id"] == q_id].to_records()
            ques_query_graph_cands = self.__get_graph_cand__(file_json, ques_str_dict[q_id], links)
            with open(os.path.join(dest_dir, q_id + ".json"), 'w+') as fp:
                json.dump(ques_query_graph_cands, fp)

    def __get_graph_cand__(self, core_chain_json, ques_str, entity_links):
        ques_query_graph_cands = {}
        for topic_entity in core_chain_json.keys():
            if topic_entity in self.topic_entity_dict:
                topic_entity_name = self.topic_entity_dict[topic_entity]
            else:
                topic_entity_name = self.sparql.get_names(topic_entity)
                self.topic_entity_dict[topic_entity] = topic_entity_name
            paths = core_chain_json[topic_entity]

            entity_query_graph_cands = []
            for path in paths:
                main_relation = path["relations"]
                is_reverse = path["is_reverse"]
                if not self.__is_valid_rel_path__(main_relation):
                    continue
                constraints = self.__get_constraint_candidates__(ques_str, topic_entity,
                                                                 topic_entity_name,
                                                                 main_relation, entity_links, is_reverse)
                cands = self.__get_query_graph_cands__(topic_entity, main_relation, constraints, is_reverse)
                entity_query_graph_cands.extend(cands)
            ques_query_graph_cands[topic_entity] = entity_query_graph_cands
        return ques_query_graph_cands

    def __get_constraint_candidates__(self, ques_str, topic_entity, topic_entity_name, relation_path, links, is_reverse):
        candidates = []
        for link in links:
            surface_form = link['mention']
            link_mid = link['mid']
            link_name = link['name']
            if surface_form is None or topic_entity_name is None: continue
            if metricUtils.jaccard_ch(topic_entity_name.lower(), surface_form.lower()) > 0.2: continue
            if link_mid == topic_entity: continue
            if len(relation_path) == 2:
                rel_key = str(relation_path)
                if rel_key in self.cvt_constraints_cache:
                    cvt_constraints = self.cvt_constraints_cache[rel_key]
                else:
                    cvt_constraints = self.sparql.get_all_cvt_constraints(topic_entity, relation_path, is_reverse, link_mid)
                    self._add_cvt_to_cache(rel_key, cvt_constraints)
                # print("{} with {} found cvt constraints {}".format(topic_entity, link, len(cvt_constraints)))
                for rel in cvt_constraints:
                    candidates.append(Constraint(link_mid, link_name, rel, False, surface_form, link["begin_index"], link["end_index"]))
            '''March 5 - Nikita
               Disabling this for speed up. Mostly results in 0 hits, and should be covered by later implicit ans entity checks'''
            # ans_constraints = self.sparql.get_ans_constraint_rel(topic_entity, relation_path, link_mid, constraint_relations_to_filter=ANS_CONSTRAINT_RELATIONS, is_reverse=is_reverse)
            # print("{} with {} found ans constraints {}".format(topic_entity, link, len(ans_constraints)))
            # for rel in ans_constraints:
            #     candidates.append(Constraint(link_mid, link_name, rel, True, surface_form, link["begin_index"], link["end_index"]))

        relation_id = str(relation_path)
        if relation_id in self.type_dict:
            type_mids_rels = self.type_dict[relation_id]
        else:
            type_mids_rels = self.sparql.get_ans_constraint_candidates(topic_entity, relation_path, ANS_CONSTRAINT_RELATIONS, is_reverse)
            # print("{} finding ans type constraints {}".format(relation_id, (type_mids_rels)))
            self.type_dict[relation_id] = type_mids_rels
        for mid in type_mids_rels.keys():
            names = self.type_name_dict.get(mid, self.sparql.get_names(mid))
            self.type_name_dict[mid] = names
            if names is None or len(names) == 0:
                continue
            match = stringUtils.match_names_to_mention(ques_str, names.split("/"))
            if match is None:
                continue
            # print("{} with {} matched implicit ans type constraints".format(topic_entity, mid))
            candidates.append(Constraint(mid, names, type_mids_rels[mid], True, match[0], match[1], match[1] + match[2]))
        return candidates

    def _add_cvt_to_cache(self, cvt_key, cvt_paths):
        self.cvt_constraints_cache_elements_fifo.append(cvt_key)
        self.cvt_constraints_cache[cvt_key] = cvt_paths
        if len(self.cvt_constraints_cache_elements_fifo) > self.cache_maxsize:
            to_delete = self.cvt_constraints_cache_elements_fifo.pop(0)
            del self.cvt_constraints_cache[to_delete]


    def __get_query_graph_cands__(self, topic_entity, main_relation, constraints, is_reverse):
        # print("{} with {} constraints".format(topic_entity, len(constraints)))
        constraint_combinations = self.__get_constraint_combinations__(constraints)
        cands = []
        # print("{} with {} combinations".format(topic_entity, len(constraint_combinations)))
        for combination in constraint_combinations:
            cand = {"relations": main_relation,
                    "entities": list(set(self.sparql.eval_all_constraints(topic_entity, main_relation, combination, is_reverse))),
                    "is_reverse": is_reverse,
                    "constraints": [ob.__dict__ for ob in combination]}
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
                conflict = cand_1.start_index <= cand_2.end_index and cand_2.start_index <= cand_1.end_index
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

    def __is_valid_rel_path__(self, relation_path):
        if len(relation_path)== 2:
            return self.__pred_filter__(relation_path[0]) and self.__pred_filter__(relation_path[1])
        else:
            return self.__pred_filter__(relation_path[0])

    def __pred_filter__(self, rel):
        if rel.startswith('base.descriptive_names'): return False
        if rel.startswith('freebase.'):return False
        if rel.startswith('base.ontologies'): return False
        if rel.startswith('media.'): return False
        if rel.startswith('base.skosbase.'): return False
        return True