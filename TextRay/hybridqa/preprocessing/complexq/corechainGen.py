import json
import pandas as pd
import os
import codecs
from kbEndPoint.utils.sparql import sparqlUtils

MAX_NEGATIVES_FRAC = 5
MAX_DEGREE = 2000
ANS_CONSTRAINT_RELATIONS = ["people.person.gender", "common.topic.notable_types", "common.topic.notable_for"]


class CoreChainGen(object):
    def __init__(self):
        self.sparql = sparqlUtils()
        self.all_path_entity_cache = {}

    def generate_core_chain(self, raw_input_path, topic_src, dest_dir, is_subsequent=False):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        questions = json.load(codecs.open(raw_input_path, 'r', encoding='utf-8'))
        questions_to_process = [q["ID"] for q in questions]
        if is_subsequent: questions_to_process = [q["ID"] for q in questions if q["compositionality_type"] == "conjunction"]

        topics_df = pd.read_csv(topic_src)
        for ques_id, group in topics_df.groupby("ques_id"):
            print(ques_id)
            if os.path.exists(os.path.join(dest_dir, ques_id + ".json")):
                continue
            if not ques_id in questions_to_process:
                continue
            candidates = {}
            for row, data in group.iterrows():
                topic_entity = data["mid"]
                topic_entity_score = data["score"]
                cands = self.main_path_candidates(topic_entity)
                for c in cands:
                    c["topic_entity_score"] = topic_entity_score
                candidates[topic_entity] = cands
            with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as fp:
                json.dump(candidates, fp, indent=4)

    def main_path_candidates(self, topic_entity):
        cands = []
        # print(topic_entity)
        if topic_entity in self.all_path_entity_cache:
            cands = self.all_path_entity_cache[topic_entity]
        else:
            union_cands = []
            union_cands += self.main_path_candidates_forward(topic_entity)
            union_cands += self.main_path_candidates_reverse(topic_entity) # for reverse hops
            cand_keys = set()
            for c in union_cands:
                key = "{}_{}_{}".format(topic_entity, c["relations"], c["is_reverse"])
                if key in cand_keys:
                    continue
                else:
                    cands.append(c)
                    cand_keys.add(key)
            self.all_path_entity_cache[topic_entity] = cands
        # print("{} has {} candidates".format(topic_entity, len(cands)))
        return cands

    def main_path_candidates_reverse(self, topic_entity):
        cands = []
        cand_preds = list(self.sparql.one_hop_rev_expansion(topic_entity)) + list(self.sparql.two_hop_rev_expansion(topic_entity))
        for cand in cand_preds:
            parsed_cand = self.parse_as_cand(topic_entity, cand, is_reverse=True)
            if parsed_cand is not None:
                cands.append(parsed_cand)
        return cands


    def main_path_candidates_forward(self, topic_entity):
        # print topic_entity
        cands = []
        cand_preds = list(self.sparql.one_hop_expansion(topic_entity)) + list(self.sparql.two_hop_expansion(topic_entity))
        for cand in cand_preds:
            parsed_cand = self.parse_as_cand(topic_entity, cand, is_reverse=False)
            if parsed_cand is not None:
                cands.append(parsed_cand)
        return cands


    def parse_as_cand(self, topic_entity, cand, is_reverse=False):
        relations = []
        counts = 0
        entities = []
        if len(cand) == 3:
            relations = [cand[0], cand[1]]
            counts = cand[2]
            if is_reverse:
                entities = self.sparql.eval_two_hop_rev_expansion(topic_entity, rel1=cand[0], rel2=cand[1])
            else:
                if float(counts) > MAX_DEGREE:
                    return None
                entities = self.sparql.eval_two_hop_expansion(topic_entity, rel1=cand[0], rel2=cand[1])
        else:
            relations = [cand[0]]
            counts = cand[1]
            if is_reverse:
                entities = self.sparql.eval_one_hop_rev_expansion(topic_entity, rel1=cand[0])
            else:
                if float(counts) > MAX_DEGREE:
                    return None
                entities = self.sparql.eval_one_hop_expansion(topic_entity, rel1=cand[0])
        return {"relations": relations,
                "is_reverse": is_reverse,
                "counts": counts,
                "entities": entities}

    def evaluate_core_chain(self, raw_input_path, dest_dir):
        questions = json.load(codecs.open(raw_input_path, 'r', encoding='utf-8'))
        found_correct = 0
        total = 0
        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                total += 1
                sparql_str = q["sparql"]
                if not os.path.exists(os.path.join(dest_dir, ques_id + ".json")):
                  continue
                entity_paths = json.load(codecs.open(os.path.join(dest_dir, ques_id + ".json"), 'r', encoding='utf-8'))
                ground = False
                for entity_key in entity_paths.keys():
                    if entity_key in sparql_str:
                        paths = entity_paths[entity_key]
                        path_found = False
                        for p in paths:
                            relations = p["relations"]
                            has_match = True
                            for r in relations:
                                if not r in sparql_str:
                                    has_match = False
                                    break
                            if has_match:
                                path_found = True
                                break
                        if path_found:
                            ground = True
                            print ques_id + "\t" + str(path_found)
                            found_correct += 1
                            break
                if not ground:
                    print ques_id + "\t" + str(ground)
        print found_correct
        print total
