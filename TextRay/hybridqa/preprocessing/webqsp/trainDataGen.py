import os
import json
from kbEndPoint.utils.sparql import sparqlUtils
from preprocessing import stringUtils
from preprocessing import metricUtils
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # will do for now?

import codecs
import pandas as pd

# PREFIX = "/Users/funke/WebQSPData"


PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final"
QUESTION_PATH = os.path.join(PREFIX, "data/WebQSP.train.json")
CANDS_DIR = os.path.join(PREFIX, "cands-train")
CANDS_WTIH_CONSTRAINTS_DIR = os.path.join(PREFIX, "cands_with_constraints-train")
CANDS_WTIH_CONSTRAINTS_DIR_DEDUP = os.path.join(PREFIX, "cands_with_constraints-train")
CANDS_WTIH_CONSTRAINTS_RESCALED_DIR = os.path.join(PREFIX, "cands_with_constraints-scaled-train")
CANDS_WTIH_CONSTRAINTS_RESCALED_NO_PRIOR_DIR = os.path.join(PREFIX, "cands_with_constraints_no_prior-scaled-train")
SMART_TOPIC_PATH = os.path.join(PREFIX, "SMART/webquestions.examples.train.e2e.top10.filter.tsv")
ALL_TOPIC_PATH = os.path.join(PREFIX, "topics/train.csv")



# QUESTION_PATH = os.path.join(PREFIX, "data/WebQSP.test.json")
# CANDS_DIR = os.path.join(PREFIX, "cands-test")
# CANDS_WTIH_CONSTRAINTS_DIR = os.path.join(PREFIX, "cands_with_constraints-test")
# CANDS_WTIH_CONSTRAINTS_DIR_DEDUP = os.path.join(PREFIX, "cands_with_constraints-test")
# CANDS_WTIH_CONSTRAINTS_RESCALED_DIR = os.path.join(PREFIX, "cands_with_constraints-scaled-test")
# CANDS_WTIH_CONSTRAINTS_RESCALED_NO_PRIOR_DIR = os.path.join(PREFIX, "cands_with_constraints_no_prior-scaled-test")
# SMART_TOPIC_PATH = os.path.join(PREFIX, "SMART/webquestions.examples.test.e2e.top10.filter.tsv")
# ALL_TOPIC_PATH = os.path.join(PREFIX, "topics/test.csv")

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
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()

        def write_top_entities(self, entity_linking_path, ques_src, dest_topic_path):
            names = ['ques_id', 'mention', 'begin_index', 'length', 'mid', 'name', 'score']
            df = pd.read_csv(entity_linking_path, delimiter='\t', names=names)
            df = df.dropna()
            df['mid'] = df['mid'].apply(lambda mid: mid[1:].replace('/', '.'))

            file_json = json.load(open(ques_src, 'r'))
            questions = file_json["Questions"]
            matched_set = []
            for question in questions:
                questionId = question["QuestionId"]
                print questionId
                ques_string = question["ProcessedQuestion"]
                parses = question["Parses"]
                for parse in parses:
                    mid = parse.get("TopicEntityMid", None)
                    e_name = parse.get("TopicEntityName", None)
                    e_mention = parse.get("PotentialTopicEntityMention", None)
                    if mid is None or e_mention is None or e_name is None:
                        continue
                    try:
                        begin_index = ques_string.index(e_mention)
                    except:
                        continue
                    matched_entity = {"ques_id": questionId, "mention": e_mention, "begin_index": begin_index,
                                      "length": begin_index + len(e_mention), \
                                      "mid": mid, "name": e_name, "score": 1.0}
                    matched_set.append(matched_entity)

            df2 = pd.DataFrame.from_records(matched_set, columns=names)
            df = df.append(df2)
            df = df.sort_values(['ques_id', 'score'], ascending=[True, False])
            df = df.drop_duplicates(subset=['ques_id', 'mid'])

            # df = df.groupby('ques_id').reset_index(drop=True)
            df.to_csv(dest_topic_path, index=False, encoding='utf-8')

        def get_ground_inferential_chains(self, question):
            parses = question["Parses"]
            if parses is None or len(parses) == 0:  # for now assume the first parse only?
                return []
            chains = {}
            for parse in parses:
                topic_entity = parse["TopicEntityMid"]
                inferential_chain = parse["InferentialChain"]
                if inferential_chain is not None:
                    to_update = chains.get(topic_entity, [])
                    key = ":".join(inferential_chain)
                    to_update.append(key)
                    chains[topic_entity] = to_update
            return chains

        def get_cands(self, ques_src, topic_src, dest_dir):
            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)
            topics_df = pd.read_csv(topic_src)
            file_json = json.load(open(ques_src, 'r'))
            questions = file_json["Questions"]

            for question in questions:
                questionId = question["QuestionId"]
                # if questionId != "WebQTrn-158":
                #     continue
                print questionId
                dest_path = os.path.join(dest_dir, questionId + ".json")
                if os.path.exists(dest_path):
                    continue
                topic_entities = topics_df[topics_df["ques_id"] == questionId].to_dict(orient='records')
                ground_chains = self.get_ground_inferential_chains(question)
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
                        ground_topic_chains = ground_chains.get(topic_entity, set())
                        for cand in one_step:
                            relations = [cand[0]]
                            key =relations[0]
                            true_label = 0
                            if key in ground_topic_chains: true_label = 1
                            cands.append({"relations": relations, "true_label": true_label, "counts": cand[1],
                                          "entities": self.sparql.eval_one_hop_expansion(topic_entity, rel1=cand[0])})
                        two_step = self.sparql.two_hop_expansion(topic_entity)
                        for cand in two_step:
                            relations = [cand[0], cand[1]]
                            key = relations[0] + ":" + relations[1]
                            true_label = 0
                            if key in ground_topic_chains: true_label = 1
                            cands.append({"relations": relations, "true_label": true_label, "counts": cand[2],
                                          "entities": self.sparql.eval_two_hop_expansion(topic_entity, rel1=cand[0], rel2=cand[1])})
                    candidates[topic_entity] = cands
                    self.all_path_entity_cache[topic_entity] = cands

                with open(dest_path, 'w+') as fp:
                    json.dump(candidates, fp, indent=4)

        def generate_query_graph_cands(self, ques_src, topic_src, core_chain_path, dest_dir):
            topics_df = pd.read_csv(topic_src)
            questions = json.load(open(ques_src, 'r'))["Questions"]

            ans_dict = {}
            ques_str_dict = {}
            for question in questions:
                entity_ans_dict = {}
                qid = question["QuestionId"]
                ques_str_dict[qid] = question["ProcessedQuestion"]
                for parse in question["Parses"]:
                    entity_ans_dict[parse["TopicEntityMid"]] = [a["AnswerArgument"] for a in parse["Answers"]]
                ans_dict[qid] = entity_ans_dict

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
                    topic_entity_name = topic_entity_names[0]
                    answers = ans_dict[q_id].get(mid, [])
                    paths = file_json[mid]
                    entity_query_graph_cands = []
                    for path in paths:
                        main_relation = path["relations"]
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
                entities = set(self.sparql.eval_all_constraints(topic_entity, main_relation, combination, False))
                # common = entities.intersection(answer_entities)
                # reward = float(len(common)) / max(1.0, (len(entities) + len(answer_entities) - len(common)))
                if len(answer_entities)  == 0:
                    reward = 0,0,0
                else:
                    reward = metricUtils.compute_f1(answer_entities, entities)
                cand = {"relations": main_relation,
                        "entities": list(entities),
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
            questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))["Questions"]
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


        def add_labels(self, ques_src, cands_dir, cands_labels_dir):
            file_json = json.load(open(ques_src, 'r'))
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
                # print("questionId " + str(questionId) + " parses " + str(parses_dict))

            if not os.path.exists(cands_labels_dir):
                os.makedirs(cands_labels_dir)
            files = [f for f in os.listdir(cands_dir) if os.path.isfile(os.path.join(cands_dir, f))]
            for f in files:
                try:
                    file_json = json.load(open(os.path.join(cands_dir, f), 'r'))
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
                                break
                        p["true_label"] = 0
                        if is_true: p["true_label"] = 1
                with open(os.path.join(cands_labels_dir, q_id + ".json"), 'w+') as fp:
                    json.dump(file_json, fp)

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
                        path["rescaled_reward"] = path["reward"]
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

        def sanity_check(self, src_dir):
            reward_files = [f for f in os.listdir(src_dir)]
            total = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            has_pos = 0
            for r in reward_files:
                print r
                ques_json = json.load(open(os.path.join(src_dir, r)))
                total += 1
                pos = False
                for topic in ques_json.keys():
                    for path in ques_json[topic]:
                        # reward = path["reward"][2]
                        true_label = path["true_label"]
                        threshold = 0.1
                        if true_label > 0:
                            pos = True
                        # if reward > threshold and true_label > 0:
                        #     tp += 1
                        # elif reward > threshold and true_label == 0:
                        #     fp += 1
                        # elif reward < threshold and true_label > 0:
                        #     fn += 1
                        # elif reward < threshold and true_label == 0:
                        #     tn += 1
                if pos:
                    has_pos+= 1

            print total
            print("tp {} tn {} fp {} fn {}".format(tp, tn, fp,fn))
            print(has_pos)

        def eval_accuracy(self, ques_src, cands_dir):
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
                main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                for topic in main_entity_paths:
                    for path in main_entity_paths[topic]:
                        if path["true_label"] == 1:
                            is_true = True
                            break
                if is_true:
                    true_pos_ct += 1
            print(true_pos_ct)
            print(total_ct)
            print(float(true_pos_ct) * 1.0 / float(total_ct))


        def eval_quality(self, ques_src, cands_dir):
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
                ques_path = os.path.join(cands_dir, ques_id + ".json")
                if not os.path.exists(ques_path):
                    continue
                print(ques_id)
                main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                best_f1 = 0
                for topic in main_entity_paths:
                    ground_ans = entity_ans_dict.get(topic, [])
                    if len(ground_ans) == 0:
                        continue
                    for path in main_entity_paths[topic]:
                        predicted_ans = path["entities"]
                        f1 = metricUtils.compute_f1(ground_ans, predicted_ans)[2]
                        if f1 > best_f1:
                            best_f1 = f1
                all_f1s.append(best_f1)
            print(np.mean(all_f1s))

        def estimate_match_priors(self, input_path, src_dir, dest_dir):
            questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            for q in questions["Questions"]:
                ques_id = q["QuestionId"]
                ques_str = q["ProcessedQuestion"]
                ques_tokens = set(nltk.word_tokenize(ques_str.lower()))
                ques_tokens = set([self.lemmatizer.lemmatize(w) for w in ques_tokens])
                ques_path = os.path.join(src_dir, ques_id + ".json")
                if not os.path.exists(ques_path):
                    continue
                print(ques_id)
                main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                priors_map = {}
                for topic in main_entity_paths:
                    for p in main_entity_paths[topic]:
                        key = str(p["relations"])
                        if key in priors_map:
                            score = priors_map[key]
                        else:
                            score = self.get_rel_surface_match(ques_tokens, p["relations"])
                            priors_map[key] = score
                        p["prior_match_score"] = score

                # rescaled prior_match_scores
                max_prior_match = self.prior_match_with_max(main_entity_paths)
                for topic in main_entity_paths:
                    for path in main_entity_paths[topic]:
                        reward = path["prior_match_score"]
                        if max_prior_match > 0:
                            path["scaled_prior_match_score"] = float(reward) * 1.0 / max_prior_match
                        else:
                            path["scaled_prior_match_score"] = 1.0
                with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as fp:
                    json.dump(main_entity_paths, fp, indent=4)

        def evaluate_rescaled_prior_match_threshold(self, src_dir, dest_dir, threshold=0.5):
            reward_files = [f for f in os.listdir(src_dir) if
                            os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
            false_positives = []
            false_negatives = []
            true_positives = []
            true_negatives = []
            positive_examples_ct = 0
            negative_examples_ct = 0
            for r in reward_files:
                ques_json = json.load(open(os.path.join(src_dir, r)))
                true_positive_ct = 0
                true_negative_ct = 0
                false_positive_ct = 0
                false_negative_ct = 0
                for topic in ques_json.keys():
                    for path in ques_json[topic]:
                        reward = path["reward"][2]
                        prior_match_score = path["prior_match_score"]
                        path_reward = reward + prior_match_score
                        path["rescaled_reward_no_prior"] = [path["reward"][0], path["reward"][1], path["reward"][2]]
                        path["rescaled_reward_no_prior"][2] = path_reward

                        # path_label = path["true_label"]
                        #
                        del path['prior_match_score']
                        del path['scaled_prior_match_score']
                        # if path["reward"][2] >= 0.5:
                        #     positive_examples_ct += 1
                        # else:
                        #     negative_examples_ct += 1
                        # if path_reward > threshold and path_label == 1:
                        #     true_positive_ct += 1
                        # elif path_reward > threshold and path_label == 0:
                        #     # print("{} has false positive {} for entity {}".format(r.replace(".json", ""), path["relations"], topic))
                        #     false_positive_ct += 1
                        # elif path_reward < threshold and path_label == 0:
                        #     true_negative_ct += 1
                        # elif path_reward < threshold and path_label == 1:
                        #     false_negative_ct += 1
                # with open(os.path.join(dest_dir, r), 'w+') as fp:
                #     json.dump(ques_json, fp, indent=4)
                # print(
                #     "{} has tp {} tn {} fp {} fn {}".format(r.replace(".json", ""), true_positive_ct, true_negative_ct,
                #                                             false_positive_ct, false_negative_ct))
                # true_positives.append(true_positive_ct)
                # true_negatives.append(true_negative_ct)
                # false_positives.append(false_positive_ct)
                # false_negatives.append(false_negative_ct)
            # print("true positive median {}, mean {} ".format(np.median(true_positives), np.mean(true_positives)))
            # print("true negative median {}, mean {} ".format(np.median(true_negatives), np.mean(true_negatives)))
            # print("false positive median {}, mean {} ".format(np.median(false_positives), np.mean(false_positives)))
            # print("false negative median {}, mean {} ".format(np.median(false_negatives), np.mean(false_negatives)))
            # print positive_examples_ct
            # print negative_examples_ct

        def get_rel_surface_match(self, ques_tokens, relations):
            relation_keywords = self.relation_tokens(relations)
            if len(relation_keywords) == 0:
                return 1.0
            keywords_in_ques = relation_keywords.intersection(ques_tokens)
            return float(len(keywords_in_ques)) * 1.0 / float(len(relation_keywords))

        def relation_tokens(self, relations):
            '''
            :param relation: standard KB relation, starts with http
            :param only_namespace: only consider namespace of the relation
            :return: list of tokens of relations, without domain
            '''
            results = []
            for relation in relations:
                tokens = relation.split('.')
                for token in tokens:
                    results = results + token.split('_')
            results = [r.lower() for r in results if r not in self.stopwords]
            return set(results)

        def prior_match_with_max(self, main_entity_paths):
            max_reward = 0.0
            for topic in main_entity_paths:
                for path in main_entity_paths[topic]:
                    if path["prior_match_score"] > max_reward:
                        max_reward = path["prior_match_score"]
            return max_reward


    if __name__ == '__main__':
        endPoint = WebQuestionsEndPoint()
        # endPoint.write_top_entities(SMART_TOPIC_PATH, QUESTION_PATH, ALL_TOPIC_PATH)
        # endPoint.get_cands(QUESTION_PATH, ALL_TOPIC_PATH, CANDS_DIR)
        # endPoint.generate_query_graph_cands(QUESTION_PATH, ALL_TOPIC_PATH, CANDS_DIR, CANDS_WTIH_CONSTRAINTS_DIR)
        # endPoint.deduplicate(QUESTION_PATH, CANDS_WTIH_CONSTRAINTS_DIR, CANDS_WTIH_CONSTRAINTS_DIR_DEDUP)
        # endPoint.add_labels(QUESTION_PATH, CANDS_WTIH_CONSTRAINTS_DIR_DEDUP, CANDS_WTIH_CONSTRAINTS_DIR_DEDUP)
        # endPoint.eval_quality(QUESTION_PATH, CANDS_WTIH_CONSTRAINTS_DIR_DEDUP)
        #endPoint.eval_accuracy(QUESTION_PATH, CANDS_WTIH_CONSTRAINTS_DIR_DEDUP)
        # endPoint.rescale_rewards_max(CANDS_WTIH_CONSTRAINTS_DIR_DEDUP, CANDS_WTIH_CONSTRAINTS_RESCALED_DIR)
        # endPoint.estimate_match_priors(QUESTION_PATH, CANDS_WTIH_CONSTRAINTS_RESCALED_DIR, CANDS_WTIH_CONSTRAINTS_RESCALED_NO_PRIOR_DIR)
        # endPoint.evaluate_rescaled_prior_match_threshold(CANDS_WTIH_CONSTRAINTS_RESCALED_NO_PRIOR_DIR, CANDS_WTIH_CONSTRAINTS_RESCALED_NO_PRIOR_DIR)
        endPoint.sanity_check(CANDS_WTIH_CONSTRAINTS_DIR)


        '''debugging'''
        # DATA_PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final"
        # QUESTION_PATH = os.path.join(DATA_PREFIX, "data/WebQSP.test.json")
        # PREFIX = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final'
        # CANDS_WTIH_CONSTRAINTS_DIR_DEDUP = os.path.join(PREFIX, "cands_with_constraints-test")
        # CANDS_WTIH_CONSTRAINTS_RESCALED_NO_PRIOR_DIR = os.path.join(PREFIX, "cands_with_constraints_no_prior-scaled-test")
        #
        # endPoint.estimate_match_priors(QUESTION_PATH, CANDS_WTIH_CONSTRAINTS_DIR_DEDUP, CANDS_WTIH_CONSTRAINTS_RESCALED_NO_PRIOR_DIR)