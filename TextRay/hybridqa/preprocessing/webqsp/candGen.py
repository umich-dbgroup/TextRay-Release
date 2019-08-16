
import os
import json
import numpy as np
from kbEndPoint.utils.sparql import sparqlUtils
from preprocessing import metricUtils
from preprocessing import stringUtils

SMART_DATA_DIR = "/Users/funke/WebQSP/SMART"
# SPLIT = "train"
# QUESTION_PATH = "/Users/funke/WebQSP/data/WebQSP.train.json"
# SRC_PATH = "/Users/funke/WebQSP/cand_gen_train_unbounded_no_entites"
# DEST_PATH = "/Users/funke/WebQSP/query_cand_gen_train_v2"
# DEST_F1_PATH = "/Users/funke/WebQSP/query_cand_gen_train_f1_v2"
# DEST_LABEL_PATH= "/Users/funke/WebQSP/query_cand_gen_train_labels_f1_v2"

WORK_DIR="/Users/funke/WebQSP"


SPLIT = "test"
QUESTION_PATH = "/Users/funke/WebQSP/data/WebQSP.test.json"
SRC_PATH = "/Users/funke/WebQSP/cand_gen_test_unbounded_no_entites"
DEST_PATH = "/Users/funke/WebQSP/query_cand_gen_test_v2"
DEST_F1_PATH = "/Users/funke/WebQSP/query_cand_gen_test_f1_v2"
DEST_LABEL_PATH= "/Users/funke/WebQSP/query_cand_gen_test_labels_f1_v2"


# DEST_PATH = "/Users/funke/WebQSP/query_cand_gen_test"
# DEST_F1_PATH = "/Users/funke/WebQSP/query_cand_gen_test_f1"
# DEST_LABEL_PATH= "/Users/funke/WebQSP/query_cand_gen_test_labels_f1"
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

class CandGen(object):
    def __init__(self):
        self.entity_linking_path = os.path.join(SMART_DATA_DIR, 'webquestions.examples.%s.e2e.top10.filter.tsv' % (SPLIT))
        self.q_links_dict = self.load_webq_linking_data()
        self.sparql = sparqlUtils()
        self.type_dict = {}
        self.type_name_dict = {}

# Load S-MART entity linking result for WebQuestions
    def load_webq_linking_data(self):
        q_links_dict = {}
        with open(self.entity_linking_path, 'r') as f:
            for line in f.readlines():
                smart_item = Smart_Entity(line)
                q_id = smart_item.q_id
                if q_id not in q_links_dict:
                    q_links_dict[q_id] = []
                q_links_dict[q_id].append(smart_item)
        return q_links_dict

    # mat: conflict matrix (if j in mat[i], then it means i and j are not compatible)
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

    def __get_constraint_combinations__(self, constraint_candidates):
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

    def __get_constraint_candidates__(self, ques_str, topic_entity, topic_entity_name, relation_path, links):
        candidates = []
        for link in links:
            if metricUtils.jaccard_ch(topic_entity_name.lower(), link.surface_form.lower()) > 0.4: continue
            if link.mid == topic_entity: continue
            if len(relation_path) == 2:
                cvt_constraints = self.sparql.get_cvt_constraint(topic_entity, relation_path, link.mid)
                for rel in cvt_constraints:
                    candidates.append(Constraint(link.mid, link.e_name, rel, False, link.surface_form, link.st_pos, link.length))
            ans_constraints = self.sparql.get_ans_constraint_rel(topic_entity, relation_path, link.mid, constraint_relations_to_filter=ANS_CONSTRAINT_RELATIONS)
            for rel in ans_constraints:
                candidates.append(Constraint(link.mid, link.e_name, rel, True, link.surface_form, link.st_pos, link.length))

        relation_id = str(relation_path)
        if relation_id in self.type_dict:
            type_mids_rels = self.type_dict[relation_id]
        else:
            type_mids_rels = self.sparql.get_ans_constraint_candidates(topic_entity, relation_path, ANS_CONSTRAINT_RELATIONS)
            self.type_dict[relation_id] = type_mids_rels
        for mid in type_mids_rels.keys():
            if mid in self.type_name_dict:
                names = self.type_name_dict[mid]
            else:
                names=self.sparql.get_names(mid)
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
            entities = set(self.sparql.eval_constraint(topic_entity, main_relation, combination))
            common = entities.intersection(answer_entities)
            reward = float(len(common)) / max(1.0,(len(entities) + len(answer_entities) - len(common)))
            cand = {"relations": main_relation,
                    "entities": list(entities),
                    "constraints": [ob.__dict__ for ob in combination],
                    "reward": reward}
            cands.append(cand)
        return cands

    def verify_labels(self):
        files = [f for f in os.listdir(DEST_LABEL_PATH) if os.path.isfile(os.path.join(DEST_LABEL_PATH, f))]
        for f in files:
            file_json = json.load(open(os.path.join(DEST_LABEL_PATH, f), 'r'))
            q_id = f.replace(".json", "")
            has_true = False
            for entity in file_json.keys():
                parses = file_json[entity]
                for p in parses:
                    if p["true_label"] == 1:
                        has_true = True
                        break
            if not has_true:
                print(q_id)

    def get_labels(self):
        file_json = json.load(open(QUESTION_PATH, 'r'))
        questions = file_json["Questions"]
        ground_parses_dict = {}
        for question in questions:
            parses_dict = []
            questionId = question["QuestionId"]
            parses = question["Parses"]
            for parse in parses:
                relationPath = parse["InferentialChain"]
                if relationPath is None: relationPath = []
                constraintRels = [c["NodePredicate"] for c in parse["Constraints"]]
                parses_dict.append({"InferentialChain": relationPath, "Constraints": constraintRels})
            ground_parses_dict[questionId] = parses_dict
            print("questionId " + str(questionId) + " parses " + str(parses_dict))

        if not os.path.exists(DEST_LABEL_PATH):
            os.makedirs(DEST_LABEL_PATH)
        files = [f for f in os.listdir(DEST_F1_PATH) if os.path.isfile(os.path.join(DEST_F1_PATH, f))]
        for f in files:
            try:
                file_json = json.load(open(os.path.join(DEST_F1_PATH, f), 'r'))
            except:
                continue
            q_id = f.replace(".json", "")
            ground_parses = ground_parses_dict[q_id]
            for entity in file_json.keys():
                parses = file_json[entity]
                for p in parses:
                    ref_inf_path = set(p["relations"])
                    ref_constraints = set([c["relation"] for c in p["constraints"]])
                    is_true = False
                    for g in ground_parses:
                        ground_inf_path = set(g["InferentialChain"])
                        ground_constraints = set(g["Constraints"])
                        if ref_inf_path == ground_inf_path and ground_constraints == ref_constraints:
                            is_true = True
                            print("found")
                            break
                    p["true_label"] = 0
                    if is_true: p["true_label"] = 1
            with open(os.path.join(DEST_LABEL_PATH, q_id + ".json"), 'w+') as fp:
                json.dump(file_json, fp)

    def compare(self):
        files = [f for f in os.listdir(SRC_PATH) if os.path.isfile(os.path.join(SRC_PATH, f))]
        failed = 0
        for f in files:
            q_id = f.replace(".json", "")
            base_json = json.load(open(os.path.join(SRC_PATH, f), 'r'))
            if not os.path.exists(os.path.join(DEST_PATH, q_id + ".json")):
                print("does not exists " + str(q_id))
                failed += 1
                continue
            new_json = json.load(open(os.path.join(DEST_PATH, q_id + ".json"), 'r'))
            if len(base_json.keys()) > len(new_json.keys()):
                print("fewer topic entities in " + str(q_id))
            for topic_entity in base_json.keys():
                if len(base_json[topic_entity]) > len(new_json[topic_entity]):
                    print("should have had more candidates " + str(q_id) + " " + str(topic_entity))
        print failed
        print len(files)

    def recompute_rewards(self):
        file_json = json.load(open(QUESTION_PATH, 'r'))
        try:
            questions = file_json["Questions"]
        except:
            return
        ans_dict = {}
        for question in questions:
            entity_ans_dict = {}
            questionId = question["QuestionId"]
            parses = question["Parses"]
            for parse in parses:
                topic_entity = parse["TopicEntityMid"]
                answer_entities = [a["AnswerArgument"] for a in parse["Answers"]]
                entity_ans_dict[topic_entity] = answer_entities
            ans_dict[questionId] = entity_ans_dict

        if not os.path.exists(DEST_F1_PATH):
            os.makedirs(DEST_F1_PATH)
        files = [f for f in os.listdir(DEST_PATH) if os.path.isfile(os.path.join(DEST_PATH, f))]
        for f in files:
            q_id = f.replace(".json", "")
            print(q_id)
            try:
                file_json = json.load(open(os.path.join(DEST_PATH, f), 'r'))
            except:
                continue
            is_valid = False
            for topic_entity in file_json.keys():
                ground_ans = ans_dict[q_id][topic_entity]
                if len(ground_ans) == 0:
                    continue
                is_valid = True
                for path in file_json[topic_entity]:
                    predicted_ans = path["entities"]
                    path["reward"] = metricUtils.compute_f1(ground_ans, predicted_ans)
            if is_valid:
                with open(os.path.join(DEST_F1_PATH, q_id+".json"), 'w+') as fp:
                    json.dump(file_json, fp)


    def process_dataset(self):
        file_json = json.load(open(QUESTION_PATH, 'r'))
        questions = file_json["Questions"]
        ans_dict = {}
        ques_str_dict = {}
        topic_entity_dict = {}
        for question in questions:
            entity_ans_dict = {}
            questionId = question["QuestionId"]
            ques_str_dict[questionId] = question["ProcessedQuestion"]
            parses = question["Parses"]
            for parse in parses:
                topic_entity = parse["TopicEntityMid"]
                topic_entity_dict[topic_entity] = parse["TopicEntityName"]
                answer_entities = [a["AnswerArgument"] for a in parse["Answers"]]
                entity_ans_dict[topic_entity] = answer_entities
            ans_dict[questionId] = entity_ans_dict

        if not os.path.exists(DEST_PATH):
            os.makedirs(DEST_PATH)
        files = [f for f in os.listdir(SRC_PATH) if os.path.isfile(os.path.join(SRC_PATH, f))]
        for f in files:
            q_id = f.replace(".json", "")
            if os.path.exists(os.path.join(DEST_PATH, q_id+".json")):
                print("exists " + str(q_id))
                continue
            ques_query_graph_cands = {}
            try:
                file_json = json.load(open(os.path.join(SRC_PATH, f), 'r'))
            except:
                print(f)
                continue
            links = []
            if q_id in self.q_links_dict:
                links = self.q_links_dict[q_id]
            print("Question " + q_id)
            for topic_entity in file_json.keys():
                answers = ans_dict[q_id][topic_entity]
                paths = file_json[topic_entity]
                entity_query_graph_cands = []
                for path in paths:
                    main_relation = path["relations"]
                    constraints = self.__get_constraint_candidates__(ques_str_dict[q_id], topic_entity, topic_entity_dict[topic_entity], main_relation, links)
                    cands = self.__get_query_graph_cands__(topic_entity, main_relation, constraints, answers)
                    entity_query_graph_cands.extend(cands)
                ques_query_graph_cands[topic_entity] = entity_query_graph_cands
                print("topic e candidates size " + str(len(entity_query_graph_cands)))
            with open(os.path.join(DEST_PATH, q_id+".json"), 'w+') as fp:
                json.dump(ques_query_graph_cands, fp)

    def get_relation_constraints_dict(self):
        relation_constraints_map = {}
        constraints_names_map = {}
        constraints_rel_map = {}
        files = [f for f in os.listdir(SRC_PATH) if os.path.isfile(os.path.join(SRC_PATH, f))]
        for f in files:
            print(f)
            file_json = json.load(open(os.path.join(SRC_PATH, f), 'r'))
            for topic_entity in file_json.keys():
                paths = file_json[topic_entity]
                for path in paths:
                    main_relation = path["relations"]
                    if str(main_relation) in relation_constraints_map:
                        continue
                    constraints_map = self.sparql.get_skeleton_constraint_candidates(relation_path=main_relation, constraint_relations_to_filter=ANS_CONSTRAINT_RELATIONS)
                    for mid in constraints_map:
                        if not mid in constraints_names_map:
                            names = self.sparql.get_names(mid)
                            constraints_names_map[mid] = names
                        updated_constraint_rels = []
                        if mid in constraints_rel_map: updated_constraint_rels = constraints_rel_map[mid]
                        updated_constraint_rels.extend(constraints_map[mid])
                        constraints_rel_map[mid] = updated_constraint_rels

                        rel_constraints = []
                        if str(main_relation) in relation_constraints_map: rel_constraints = relation_constraints_map[str(main_relation)]
                        rel_constraints.append(mid)
                        relation_constraints_map[str(main_relation)] = rel_constraints
            print(len(relation_constraints_map))
            print(len(constraints_names_map))
            print(len(constraints_rel_map))
        with open(os.path.join(WORK_DIR, 'relation_constraints_'+ SPLIT + ".json"), 'w+') as fp:
            json.dump(relation_constraints_map, fp)
        with open(os.path.join(WORK_DIR, 'constraint_names_'+ SPLIT + ".json"), 'w+') as fp:
            json.dump(constraints_names_map, fp)
        with open(os.path.join(WORK_DIR, 'constraint_rels_'+ SPLIT + ".json"), 'w+') as fp:
            json.dump(constraints_rel_map, fp)

    def test(self):
        file_json = json.load(open(QUESTION_PATH, 'r'))
        questions = file_json["Questions"]
        ans_dict = {}
        ques_str_dict = {}
        topic_entity_dict = {}
        for question in questions:
            entity_ans_dict = {}
            questionId = question["QuestionId"]
            ques_str_dict[questionId] = question["ProcessedQuestion"]
            parses = question["Parses"]
            for parse in parses:
                topic_entity = parse["TopicEntityMid"]
                topic_entity_dict[topic_entity] = parse["TopicEntityName"]
                answer_entities = [a["AnswerArgument"] for a in parse["Answers"]]
                entity_ans_dict[topic_entity] = answer_entities
            ans_dict[questionId] = entity_ans_dict

        q_id = "WebQTest-184"
        q_str = "what state does romney live in"
        if q_id in self.q_links_dict:
            links = self.q_links_dict[q_id]
        file_json = json.load(open(os.path.join(SRC_PATH, q_id + ".json"), 'r'))
        for topic_entity in file_json.keys():
            paths = file_json[topic_entity]
            answers = ans_dict[q_id][topic_entity]
            for path in paths:
                main_relation = path["relations"]
                constraints = self.__get_constraint_candidates__(q_str, topic_entity, topic_entity_dict[topic_entity], main_relation, links)
                cands = self.__get_query_graph_cands__(topic_entity, main_relation, constraints, answers)


if __name__ == '__main__':
    candgen = CandGen()
    #candgen.process_dataset()
    #candgen.get_relation_constraints_dict()
    #candgen.verify_labels()
    #candgen.test()
    #candgen.compare()
    #candgen.recompute_rewards()
    candgen.get_labels()