import json
from kbEndPoint.utils.sparql import sparqlUtils
import pandas as pd
from subgraphInterface import SubGraph
import os
from numpy import median

RELATIONS_FILTER = "/Users/funke/WebQSP/all_relations.txt"
QUESTION_PATH = "/Users/funke/WebQSP/data/WebQSP.train.json"
SUBGRAPHS_PATH = "/Users/funke/WebQSP/webqsp_subgraphs.json"
#DEST_DIR_PATH = "/Users/funke/WebQSP/cand_gen_train_unbounded"
DEST_DIR_PATH = "/Users/funke/WebQSP/cand_gen_train_unbounded_no_entites"
# QUESTION_PATH = "/Users/funke/WebQSP/data/WebQSP.test.json"
# SUBGRAPHS_PATH = "/Users/funke/WebQSP/webqsp_subgraphs.json"
# DEST_DIR_PATH = "/Users/funke/WebQSP/cand_gen_test"


class CoreRelationCandidateGen(object):

    def __init__(self):
        self.relations_to_filter = set(pd.read_csv(RELATIONS_FILTER, names=['rel']).rel)
        self.validate_subgraph = False
        self.sparql = sparqlUtils()
        self.entity_cache = {}
        self.is_bounded = True

    def __is_valid_subgraph(self, subgraph, topic_entity, answer_entities):
        if not subgraph.graph.has_node(topic_entity):
            return False
        for entity in answer_entities:
            if entity == topic_entity:
                return False
        for entity in answer_entities:
            if subgraph.graph.has_node(entity): # any of the entities exist
                return True
        return False

    def __get_subgraphs(self, subgraph_src, split):
        subgraphs = {}
        with open(subgraph_src, 'r') as f:
            for line in f:
                question_obj = json.loads(line)
                question_id = question_obj["id"]
                if not question_id.startswith(split):
                    continue
                print("processing subgraph for {}".format(question_id))
                subgraphs[question_id] = SubGraph.read(question_obj["subgraph"]["tuples"])
        return subgraphs

    def __compute_reward(self, answer_entities, cands):
        rewards = []
        for cand in cands:
            entities = set(cand["entities"])
            answer_entities = set(answer_entities)

            common = entities.intersection(answer_entities)
            reward =  float(len(common)) / (len(entities) + len(answer_entities) - len(common))
            cand["reward"] = reward
            rewards.append(cand)
        return rewards

    def __find_unbounded_candidates(self, question, subgraph=None):
        questionId = question["QuestionId"]
        candidates = {}
        if self.validate_subgraph and subgraph is None:
            print("missing subgraph for question {}".format(questionId))
            return candidates
        print("processing candidates for {}".format(questionId))
        parses = question["Parses"]
        if parses is None or len(parses) == 0:  # for now assume the first parse only?
            return
        for parse in parses:
            topic_entity = parse["TopicEntityMid"]
            if topic_entity in candidates:
                continue
            answer_entities = [a["AnswerArgument"] for a in parse["Answers"] if a["AnswerType"] == "Entity"]
            if self.validate_subgraph and not self.__is_valid_subgraph(subgraph, topic_entity, answer_entities):
                print("missing valid subgraph for question {}".format(questionId))
                continue
            cands = []
            keyset = set()
            for tgt in answer_entities:
                cand_preds = []  # only keep candidates that are supported by subgraph
                one_step_path = self.sparql.one_hop_path(topic_entity, tgt)
                if self.validate_subgraph:
                    cand_preds += [cand for cand in one_step_path if subgraph.has_one_step(topic_entity, cand[0])]
                else:
                    cand_preds += one_step_path
                if len(one_step_path) == 0:
                    two_step_path = self.sparql.two_hop_path(topic_entity, tgt)
                    if self.validate_subgraph:
                        cand_preds += [cand for cand in two_step_path if subgraph.has_two_step(topic_entity, cand[0], cand[1])]
                    else:
                        cand_preds += two_step_path
                for cand in cand_preds:
                    if len(cand) == 2:
                        key = cand[0]+ " " + cand[1]
                        if key in keyset:
                            continue
                        cands.append({"relations": [cand[0], cand[1]],"entities": self.sparql.eval_two_hop_expansion(topic_entity, rel1=cand[0], rel2=cand[1])})
                        keyset.add(key)
                    else:
                        if cand in keyset:
                            continue
                        cands.append({"relations": [cand], "entities": self.sparql.eval_one_hop_expansion(topic_entity, rel1=cand)})
                        keyset.add(cand)
            cands_with_rewards = self.__compute_reward(answer_entities, cands)
            #print("{} candidates".format(len(cands_with_rewards)))
            candidates[topic_entity] = cands_with_rewards
        return candidates

    def __find_bounded_candidates(self, question, subgraph=None):
            questionId = question["QuestionId"]
            candidates = {}
            if self.validate_subgraph and subgraph is None:
                print("missing subgraph for question {}".format(questionId))
                return candidates
            print("processing candidates for {}".format(questionId))
            parses = question["Parses"]
            if parses is None or len(parses) == 0: # for now assume the first parse only?
                return
            for parse in parses:
                topic_entity = parse["TopicEntityMid"]
                inferential_chain = parse["InferentialChain"]
                if topic_entity in candidates:
                    continue
                answer_entities = [a["AnswerArgument"] for a in parse["Answers"]]
                if topic_entity is None:
                    continue
                if topic_entity in self.entity_cache:
                    cands = self.entity_cache[topic_entity]
                    print("found in cache {}".format(questionId))
                else:

                    if self.validate_subgraph and not self.__is_valid_subgraph(subgraph, topic_entity, answer_entities):
                        print("missing valid subgraph for question {}".format(questionId))
                        continue
                    # one_step = self.sparql.one_hop_expansion(topic_entity, self.relations_to_filter)
                    # two_step = self.sparql.two_hop_expansion(topic_entity, self.relations_to_filter)
                    one_step = self.sparql.one_hop_expansion(topic_entity)
                    two_step = self.sparql.two_hop_expansion(topic_entity)

                    cand_preds = [] # only keep candidates that are supported by subgraph
                    if self.validate_subgraph:
                        cand_preds += [cand for cand in one_step if subgraph.has_one_step(topic_entity, cand[0])]
                        cand_preds += [cand for cand in two_step if subgraph.has_two_step(topic_entity, cand[0], cand[1])]
                    else:
                        cand_preds += one_step
                        cand_preds += two_step

                    cands = []
                    for cand in cand_preds:
                        true_label = 0
                        if len(cand) == 3:
                            relations = [cand[0], cand[1]]
                            if relations == inferential_chain: true_label = 1
                            cands.append({"relations": relations, "true_label": true_label, "counts": cand[2], "entities": self.sparql.eval_two_hop_expansion(topic_entity, rel1=cand[0], rel2=cand[1])})
                        else:
                            relations = [cand[0]]
                            if relations == inferential_chain: true_label = 1
                            cands.append({"relations": [cand[0]], "true_label": true_label, "counts": cand[1], "entities": self.sparql.eval_one_hop_expansion(topic_entity, rel1=cand[0])})
                    self.entity_cache[topic_entity] = cands
                cands_with_rewards = self.__compute_reward(answer_entities, cands)
                print("{} candidates".format(len(cands_with_rewards)))
                candidates[topic_entity] = cands_with_rewards
            return candidates

    def process_dataset(self, questions_src, des_dir, split, subgraph_src):
        if not os.path.exists(des_dir):
            os.makedirs(des_dir)
        if self.validate_subgraph:
            subgraphs = self.__get_subgraphs(subgraph_src, split)
        file_json = json.load(open(questions_src, 'r'))
        questions = file_json["Questions"]
        for question in questions:
            questionId = question["QuestionId"]
            subgraph = None
            if self.validate_subgraph:
                subgraph = subgraphs[question["QuestionId"]]
            if self.is_bounded:
                cand_with_rewards = self.__find_bounded_candidates(question, subgraph)
            else:
                cand_with_rewards = self.__find_unbounded_candidates(question, subgraph)
            if cand_with_rewards is None or len(cand_with_rewards) == 0:
                continue
            with open(os.path.join(des_dir, questionId+".json"), 'w+') as fp:
                json.dump(cand_with_rewards, fp)

if __name__ == '__main__':
    generator = CoreRelationCandidateGen()
    generator.process_dataset(questions_src=QUESTION_PATH, des_dir=DEST_DIR_PATH, split="WebQTrn", subgraph_src=SUBGRAPHS_PATH)
    #generator.process_dataset(questions_src=QUESTION_PATH, des_dir=DEST_DIR_PATH, split="WebQTest", subgraph_src=SUBGRAPHS_PATH)
    files = [f for f in os.listdir(DEST_DIR_PATH) if os.path.isfile(os.path.join(DEST_DIR_PATH, f))]
    all_paths = []
    for f in files:
        print f
        file_json = json.load(open(os.path.join(DEST_DIR_PATH, f), 'r'))
        total_paths = 0
        for entity in file_json.keys():
            paths = file_json[entity]
            rewarded_paths = [p for p in paths if p["reward"] > 0]
            total_paths += len(rewarded_paths)
        all_paths.append(total_paths)
    print(sum(all_paths) / float(len(all_paths)))

    print(median(all_paths))
