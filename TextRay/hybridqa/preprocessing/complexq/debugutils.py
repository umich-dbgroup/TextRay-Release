import json
import codecs
import os
from preprocessing.complexq.corechainGen import CoreChainGen

PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess"
MAX_NEGATIVES_FRAC = 5
MAX_DEGREE = 2000
ANS_CONSTRAINT_RELATIONS = ["people.person.gender", "common.topic.notable_types", "common.topic.notable_for"]


# RAW_INPUT_PATH = os.path.join(PREFIX, "annotated_orig/train.json")
# INPUT_PATH = os.path.join(PREFIX, "annotated/train.json")
# RAW_EL_PATH = os.path.join(PREFIX, "el/sub1/train_el.csv")
# TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/train_topic.csv")
# SUB1_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub1/train_topic.csv")
# SUB2_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub2/train_topic.csv")
#
#
# CANDIDATE_SUB1_DEST_PATH = os.path.join(PREFIX, "cands/train/sub1")
# CANDIDATE_SUB1_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/train/sub1_with_constraints")
# CANDIDATE_SUB2_DEST_PATH = os.path.join(PREFIX, "cands/train/sub2")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/train/main_with_constraints")


RAW_INPUT_PATH = os.path.join(PREFIX, "annotated_orig/dev.json")
INPUT_PATH = os.path.join(PREFIX, "annotated/dev.json")
RAW_EL_PATH = os.path.join(PREFIX, "el/sub1/dev_el.csv")
TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/dev_topic.csv")
SUB1_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub1/dev_topic.csv")
SUB2_TOPIC_PATH = os.path.join(PREFIX, "topic_entities/sub2/dev_topic.csv")


CANDIDATE_SUB1_DEST_PATH = os.path.join(PREFIX, "cands/dev/sub1")
CANDIDATE_SUB1_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/dev/sub1_with_constraints")
CANDIDATE_SUB2_DEST_PATH = os.path.join(PREFIX, "cands/dev/sub2")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/dev/main_with_constraints")


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
# CANDIDATE_SUB2_DEST_PATH = os.path.join(PREFIX, "cands/test/sub2")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/test/main_with_constraints")


def evaluate_sub1_rewards_quality(raw_input_path, sub1_cand_dir, main_cand_dir):
    questions = json.load(codecs.open(raw_input_path, 'r', encoding='utf-8'))
    for q in questions:
        ques_id = q["ID"]
        type = q["compositionality_type"]
        print(ques_id)
        if type == "conjunction" or type == "composition":
            sparql_str = q["sparql"]
            if not os.path.exists(os.path.join(sub1_cand_dir, ques_id + ".json")):
                continue
            if not os.path.exists(os.path.join(main_cand_dir, ques_id + ".json")):
                continue
            sub1_keys = parse_cands_as_set(os.path.join(sub1_cand_dir, ques_id + ".json"))
            main_rewards = parse_rewards_as_dict(os.path.join(main_cand_dir, ques_id+ ".json"))

            for sub1_key in sub1_keys:
                if not sub1_key in main_rewards:
                    print(sub1_key)
                    continue
                rewards = main_rewards[sub1_key]
                if rewards[2] > 0:
                    print("{} rewards {}".format(sub1_key, rewards))


def parse_cands_as_set(file_path):
    keys = set()
    main_entity_paths = json.load(codecs.open(file_path, 'r', encoding='utf-8'))
    for topic in main_entity_paths:
        for p in main_entity_paths[topic]:
            key = get_lookup_key(topic, p)
            keys.add(key)
    return keys


def parse_rewards_as_dict(file_path):
    rewards = {}
    main_entity_paths = json.load(codecs.open(file_path, 'r', encoding='utf-8'))
    for topic in main_entity_paths:
        for p in main_entity_paths[topic]:
            key = get_lookup_key(topic, p)
            reward = p["reward"]
            rewards[key] = reward
    return rewards


def get_lookup_key(topic, rel_data):
    look_up_key = topic + "_" + str(rel_data["relations"]) + "_" + str(rel_data["constraints"])
    return look_up_key

def main_path_cands(topic):
    corechainGen = CoreChainGen()
    cands = corechainGen.main_path_candidates(topic)
    for cand in cands:
        print(cand)


def ques_analysis(raw_input_path):
    questions = json.load(codecs.open(raw_input_path, 'r', encoding='utf-8'))
    ques_ct = 0
    for q in questions:
        ques_id = q["ID"]
        type = q["compositionality_type"]
        if type == "conjunction":
            if len(q["entities"]) < 2:
                print(ques_id)
                ques_ct += 1

    print(ques_ct)

if __name__ == '__main__':
    evaluate_sub1_rewards_quality(INPUT_PATH, CANDIDATE_SUB1_WITH_CONSTRAINTS_DEST_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH)
    # main_path_cands("m.010ffmhn")
    # ques_analysis(INPUT_PATH)