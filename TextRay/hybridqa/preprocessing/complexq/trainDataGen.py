import json
import os
import sys
import random

from kbEndPoint.utils.sparql import sparqlUtils
from preprocessing import metricUtils
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # will do for now?

import codecs
import numpy as np
from scipy import stats


#PREFIX = "/Users/funke/Documents/ComplexWebQuestions_preprocess"
PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess"


# INPUT_PATH = os.path.join(PREFIX, "annotated/train.json")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/train/main_with_constraints")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_PATH = os.path.join(PREFIX, "rewards/dedup/train")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH = os.path.join(PREFIX, "rewards/labels/train")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PATH = os.path.join(PREFIX, "rewards/rescaled_max/train")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_Z_SCORE_PATH = os.path.join(PREFIX, "rewards/rescaled_zscore/train")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_Z_SCORE_PRIORS_PATH = os.path.join(PREFIX, "rewards/rescaled_zscore_priors/train")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors/train")
#
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_1_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors_derived_0.1/train")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_5_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors_derived_0.5/train")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_DERIVED_1_PATH = os.path.join(PREFIX, "rewards/rescaled_max_derived_0.1/train")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_DERIVED_5_PATH = os.path.join(PREFIX, "rewards/rescaled_max_derived_0.5/train")
# TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/train_topic.csv")

# INPUT_PATH = os.path.join(PREFIX, "annotated/dev.json")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/dev/main_with_constraints")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_PATH = os.path.join(PREFIX, "rewards/dedup/dev")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH = os.path.join(PREFIX, "rewards/labels/dev")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PATH = os.path.join(PREFIX, "rewards/rescaled_max/dev")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_Z_SCORE_PATH = os.path.join(PREFIX, "rewards/rescaled_zscore/dev")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_Z_SCORE_PRIORS_PATH = os.path.join(PREFIX, "rewards/rescaled_zscore_priors/dev")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors/dev")
#
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_1_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors_derived_0.1/dev")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_5_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors_derived_0.5/dev")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_DERIVED_1_PATH = os.path.join(PREFIX, "rewards/rescaled_max_derived_0.1/dev")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_DERIVED_5_PATH = os.path.join(PREFIX, "rewards/rescaled_max_derived_0.5/dev")
# TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/dev_topic.csv")

INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/test/main_with_constraints")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_PATH = os.path.join(PREFIX, "rewards/dedup/test")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH = os.path.join(PREFIX, "rewards/labels/test")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PATH = os.path.join(PREFIX, "rewards/rescaled_max/test")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors/test")

CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_1_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors_derived_0.1/test")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_5_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors_derived_0.5/test")
TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/test_topic.csv")

#
# INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
# CANDIDATE_SUB1_DEST_PATH = os.path.join(PREFIX, "cands/test/sub1_with_constraints")
# CANDIDATE_SUB1_REWARDS_PATH = os.path.join(PREFIX, "rewards/test-all/sub1")
# CANDIDATE_SUB2_DEST_PATH = os.path.join(PREFIX, "cands/test/sub2")
# CANDIDATE_SUB2_REWARDS_PATH = os.path.join(PREFIX, "rewards/test-all/sub2")
# TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/test_topic.csv")


# INPUT_PATH = os.path.join(PREFIX, "annotated/dev.json")
# CANDIDATE_SUB1_DEST_PATH = os.path.join(PREFIX, "cands/dev/sub1_with_constraints")
# CANDIDATE_SUB1_REWARDS_PATH = os.path.join(PREFIX, "rewards/dev/sub1")
# CANDIDATE_SUB2_DEST_PATH = os.path.join(PREFIX, "cands/dev/sub2")
# CANDIDATE_SUB2_REWARDS_PATH = os.path.join(PREFIX, "rewards/dev/sub2")
# TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/dev_topic.csv")


class TrainDataGenerator(object):
    def __init__(self):
        self.sparql = sparqlUtils()
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def deduplicate(self, input_path, src_dir, dest_dir):
        questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
            # if True:
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
                        look_up_key = self.get_lookup_key(topic, path) + "_" + path.get("src", "sub1")
                        if look_up_key in look_up_keys:
                            continue
                        look_up_keys.add(look_up_key)
                        paths.append(path)
                    print("{} deduplicated to {}".format(len(main_entity_paths[topic]), len(paths)))
                    if len(paths) > 0:
                        main_entity_paths_dedup[topic] = paths
                with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as fp:
                    json.dump(main_entity_paths_dedup, fp, indent=4)

    def get_expected_labels(self, input_path, src_dir, dest_dir):
        questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                sparql_str = q["sparql"]
                ques_path = os.path.join(src_dir, ques_id + ".json")
                if not os.path.exists(ques_path):
                    continue
                print(ques_id)
                pos_label_paths = {}
                main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                for topic in main_entity_paths:
                    pos_paths = []
                    for path in main_entity_paths[topic]:
                        is_match = topic in sparql_str #topic should be in sparql
                        for r in path['relations']: #relations should be in sparql
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

                with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as fp:
                    json.dump(main_entity_paths, fp, indent=4)


    def get_topic_scores(self, input_path, src_dir, dest_dir, topic_path):
        questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
        topics_df = pd.read_csv(topic_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                ques_path = os.path.join(src_dir, ques_id + ".json")
                if not os.path.exists(ques_path):
                    continue
                print(ques_id)
                main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                topic_scores_df = topics_df[topics_df["ques_id"] ==ques_id].to_records(index=False)
                topic_scores = {e["mid"]: e["score"] for e in topic_scores_df}
                for topic in main_entity_paths:
                    topic_score = topic_scores.get(topic, 0.0)
                    for path in main_entity_paths[topic]:
                        path["topic_entity_score"] = topic_score
                with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as fp:
                    json.dump(main_entity_paths, fp, indent=4)

    def rescale_rewards_max(self, input_path, src_dir, dest_dir):
        questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                ques_path = os.path.join(src_dir, ques_id + ".json")
                if not os.path.exists(ques_path):
                    continue
                print(ques_id)
                main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                max_ques_reward = self.reward_with_max_f1(main_entity_paths)
                if max_ques_reward[2] > 0:
                    for topic in main_entity_paths:
                        for path in main_entity_paths[topic]:
                            reward = path["reward"]
                            reward[2] = float(reward[2]) * 1.0 / float(max_ques_reward[2])
                            if max_ques_reward[0] > 0:
                                reward[0] = min(1.0, float(reward[0]) * 1.0 / float(max_ques_reward[0])) # hacky way of clipping
                            if max_ques_reward[1] > 0:
                                reward[1] = min(1.0, float(reward[1]) * 1.0 / float(max_ques_reward[1])) # hacky way of clipping
                with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as fp:
                    json.dump(main_entity_paths, fp, indent=4)


    def rescale_reward_zscore(self, input_path, src_dir ,dest_dir):
        questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                ques_path = os.path.join(src_dir, ques_id + ".json")
                if not os.path.exists(ques_path):
                    continue
                print(ques_id)
                main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                zscores = self.zscores(main_entity_paths)
                for topic in main_entity_paths:
                    for path in main_entity_paths[topic]:
                        key = self.get_lookup_key(topic, path)
                        path["reward"][2] = zscores[key]
                with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as fp:
                    json.dump(main_entity_paths, fp, indent=4)


    def zscores(self, main_entity_paths):
        if len(main_entity_paths) == 0:
            return {}
        rewards = {}
        for topic in main_entity_paths:
            for p in main_entity_paths[topic]:
                key = self.get_lookup_key(topic, p)
                rewards[key] = p["reward"][2]
        reward_key_values = [[key, v] for key, v in rewards.items()]
        reward_values = [l[1] for l in reward_key_values]
        try:
            zscores = stats.zscore(np.array(reward_values))
        except:
            zscores = [0.0 for l in reward_values]
        for i, zscore in enumerate(zscores):
            if np.isnan(zscore):
                zscore = 0.0
            reward_key_values[i][1] = zscore
        return dict(reward_key_values)


    def reward_with_max_f1(self, main_entity_paths):
        max_reward = 0,0,0
        for topic in main_entity_paths:
            for path in main_entity_paths[topic]:
                if path["reward"][2] > max_reward[2]:
                    max_reward = path["reward"]
        return max_reward

    def get_lookup_key(self, topic, rel_data):
        if "constraints" in rel_data:
            look_up_key = topic + "_" + str(rel_data["relations"]) + "_" + str(rel_data["constraints"]) + "_" + str(rel_data["is_reverse"])
        else:
            look_up_key = topic + "_" + str(rel_data["relations"]) + "_" + str(rel_data["is_reverse"])
        return look_up_key


    def estimate_match_priors(self,input_path, src_dir ,dest_dir):
        questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for q in questions:
            ques_id = q["ID"]
            ques_str = q["question"]
            ques_tokens = set(nltk.word_tokenize(ques_str.lower()))
            ques_tokens = set([self.lemmatizer.lemmatize(w) for w in ques_tokens])
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
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

    def prior_match_with_max(self, main_entity_paths):
        max_reward = 0.0
        for topic in main_entity_paths:
            for path in main_entity_paths[topic]:
                if path["prior_match_score"] > max_reward:
                    max_reward = path["prior_match_score"]
        return max_reward

    def estimate_co_occur_priors(self,input_path, src_dir ,dest_dir, lexicon_src):
        questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
        lexicon_df = pd.read_csv(lexicon_src, names=['rel_keyword', 'ques_keyword'], sep='\t')
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for q in questions:
            ques_id = q["ID"]
            ques_str = q["question"]
            ques_tokens = set(nltk.word_tokenize(ques_str.lower()))
            ques_tokens = set([self.lemmatizer.lemmatize(w) for w in ques_tokens])
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
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
                            score = self.get_co_occur_score(ques_tokens, p["relations"], lexicon_df)
                            priors_map[key] = score
                        p["prior_co_occur_score"] = score

                # rescaled prior_match_scores
                max_prior_match = self.co_occur_with_max(main_entity_paths)
                for topic in main_entity_paths:
                    for path in main_entity_paths[topic]:
                        reward = path["prior_co_occur_score"]
                        if max_prior_match > 0:
                            path["scaled_prior_co_occur_score"] = float(reward) * 1.0 / max_prior_match
                        else:
                            path["scaled_prior_co_occur_score"] = 1.0
                with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as fp:
                    json.dump(main_entity_paths, fp, indent=4)

    def co_occur_with_max(self, main_entity_paths):
        max_reward = 0.0
        for topic in main_entity_paths:
            for path in main_entity_paths[topic]:
                if path["prior_co_occur_score"] > max_reward:
                    max_reward = path["prior_co_occur_score"]
        return max_reward

    def get_co_occur_score(self, ques_tokens, relations, lexicon_df):
        score = 0.0
        relation_keywords = self.relation_tokens(relations)
        if len(relation_keywords) == 0 or len(ques_tokens) == 0:
            return 0.0
        for r in relation_keywords:
            ques_lexicon_entries = lexicon_df[lexicon_df['rel_keyword'] == r]['ques_keyword'].tolist()
            for q in ques_lexicon_entries:
                if q in ques_tokens:
                    score += 1.0
        return score

    def get_rel_surface_match(self, ques_tokens, relations):
        relation_keywords = self.relation_tokens(relations)
        if len(relation_keywords) == 0:
            return 1.0
        keywords_in_ques = relation_keywords.intersection(ques_tokens)
        return float(len(keywords_in_ques)) * 1.0/ float(len(relation_keywords))


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

    def expected_labels_stats(self, src_dir):
        reward_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
        true_labels_ct = []
        has_pos_ct = 0
        for r in reward_files:
            ques_json = json.load(open(os.path.join(src_dir, r)))
            true_label_ct = 0
            for topic in ques_json.keys():
                for path in ques_json[topic]:
                    if path["approx_label"] == 1:
                        true_label_ct += 1
            print("{} has {} true labels".format(r, true_label_ct))
            true_labels_ct.append(true_label_ct)
            if true_label_ct > 0:
                has_pos_ct += 1
        print("median true labels {} ".format(np.median(true_labels_ct)))
        print("mean true labels {} ".format(np.mean(true_labels_ct)))
        print("ques with true label {}".format(has_pos_ct))


    def evaluate_rescaled_threshold(self, src_dir, threshold=0.5):
        reward_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
        false_positives = []
        false_negatives = []
        true_positives = []
        true_negatives = []
        for r in reward_files:
            ques_json = json.load(open(os.path.join(src_dir, r)))
            true_positive_ct = 0
            true_negative_ct = 0
            false_positive_ct = 0
            false_negative_ct = 0
            for topic in ques_json.keys():
                for path in ques_json[topic]:
                    path_reward = path["reward"][2]
                    path_label = path["approx_label"]
                    if path_reward > threshold and path_label == 1:
                        true_positive_ct += 1
                    elif path_reward > threshold and path_label == 0:
                        false_positive_ct += 1
                    elif path_reward < threshold and path_label == 0:
                        true_negative_ct += 1
                    elif path_reward < threshold  and path_label == 1:
                        false_negative_ct += 1
            print("{} has tp {} tn {} fp {} fn {}".format(r, true_positive_ct, true_negative_ct, false_positive_ct, false_negative_ct))
            true_positives.append(true_positive_ct)
            true_negatives.append(true_negative_ct)
            false_positives.append(false_positive_ct)
            false_negatives.append(false_negative_ct)
        print("true positive median {}, mean {} ".format(np.median(true_positives), np.mean(true_positives)))
        print("true negative median {}, mean {} ".format(np.median(true_negatives), np.mean(true_negatives)))
        print("false positive median {}, mean {} ".format(np.median(false_positives), np.mean(false_positives)))
        print("false negative median {}, mean {} ".format(np.median(false_negatives), np.mean(false_negatives)))

    def get_prior_match_threshold(self, src_dir):
        reward_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
        match_scores = []
        for r in reward_files:
            print(r)
            ques_json = json.load(open(os.path.join(src_dir, r)))
            ques_match_scores = []
            for topic in ques_json.keys():
                for path in ques_json[topic]:
                    path_label = path["approx_label"]
                    prior_match_score = path["scaled_prior_match_score"]
                    if path_label == 0:
                        ques_match_scores.append(prior_match_score)
            ques_mean_score = 0.0
            if len(ques_match_scores) > 0:
                ques_mean_score = np.mean(ques_match_scores)
            match_scores.append(ques_mean_score)
        return np.mean(match_scores)


    def get_scaled_reward_threshold(self, src_dir):
        reward_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
        match_scores = []
        for r in reward_files:
            print(r)
            ques_json = json.load(open(os.path.join(src_dir, r)))
            ques_match_scores = []
            for topic in ques_json.keys():
                for path in ques_json[topic]:
                    path_label = path["approx_label"]
                    prior_match_score = path["reward"][2]
                    if path_label == 1:
                        print(prior_match_score)
                        ques_match_scores.append(prior_match_score)
            ques_mean_score = 0.0
            if len(ques_match_scores) > 0:
                ques_mean_score = np.mean(ques_match_scores)
            match_scores.append(ques_mean_score)
        return np.mean(match_scores)


    def get_threshold(self, src_dir):
        reward_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
        match_scores = []
        for r in reward_files:
            print(r)
            ques_json = json.load(open(os.path.join(src_dir, r)))
            ques_match_scores = []
            for topic in ques_json.keys():
                for path in ques_json[topic]:
                    path_label = path["approx_label"]
                    reward_score = path["reward"][2]
                    prior_score = path["scaled_prior_match_score"]
                    if path_label == 0:
                        # print(reward_score + prior_score)
                        ques_match_scores.append(reward_score + prior_score)
            ques_mean_score = 0.0
            if len(ques_match_scores) > 0:
                ques_mean_score = np.mean(ques_match_scores)
            match_scores.append(ques_mean_score)
        return np.mean(match_scores)

    def get_prior_topic_threshold(self, src_dir):
        reward_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
        match_scores = []
        for r in reward_files:
            ques_json = json.load(open(os.path.join(src_dir, r)))
            ques_topic_scores = []
            for topic in ques_json.keys():
                for path in ques_json[topic]:
                    path_label = path["approx_label"]
                    topic_score = path["topic_entity_score"]
                    if path_label == 1:
                        ques_topic_scores.append(topic_score)
            ques_mean_score = 0.0
            if len(ques_topic_scores) > 0:
                ques_mean_score = np.mean(ques_topic_scores)
            match_scores.append(ques_mean_score)
        return np.mean(match_scores)

    def evaluate_rescaled_prior_match_threshold(self, src_dir, threshold=0.5, topic_threshold=0.9):
        reward_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
        false_positives = []
        false_negatives = []
        true_positives = []
        true_negatives = []
        for r in reward_files:
            ques_json = json.load(open(os.path.join(src_dir, r)))
            true_positive_ct = 0
            true_negative_ct = 0
            false_positive_ct = 0
            false_negative_ct = 0
            for topic in ques_json.keys():
                for path in ques_json[topic]:
                    topic_score = path["topic_entity_score"]
                    reward = path["reward"][2]
                    path_label = path["approx_label"]
                    prior_match_score = path["prior_match_score"]
                    path_reward = reward + prior_match_score
                    if path_reward > threshold and path_label == 1 and topic_score > topic_threshold:
                        true_positive_ct += 1
                    elif path_reward > threshold and path_label == 0 and topic_score > topic_threshold:
                        print("{} has false positive {} for entity {}".format(r.replace(".json", ""), path["relations"], topic))
                        false_positive_ct += 1
                    elif path_reward < threshold and path_label == 0:
                        true_negative_ct += 1
                    elif path_reward < threshold  and path_label == 1:
                        false_negative_ct += 1
            print("{} has tp {} tn {} fp {} fn {}".format(r.replace(".json", ""), true_positive_ct, true_negative_ct, false_positive_ct, false_negative_ct))
            true_positives.append(true_positive_ct)
            true_negatives.append(true_negative_ct)
            false_positives.append(false_positive_ct)
            false_negatives.append(false_negative_ct)
        print("true positive median {}, mean {} ".format(np.median(true_positives), np.mean(true_positives)))
        print("true negative median {}, mean {} ".format(np.median(true_negatives), np.mean(true_negatives)))
        print("false positive median {}, mean {} ".format(np.median(false_positives), np.mean(false_positives)))
        print("false negative median {}, mean {} ".format(np.median(false_negatives), np.mean(false_negatives)))

    def add_derived_labels(self, src_dir, dest_dir, threshold=0.5, topic_threshold=0.9, include_prior=False):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        reward_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
        for r in reward_files:
            print(r)
            ques_json = json.load(open(os.path.join(src_dir, r)))
            for topic in ques_json.keys():
                for path in ques_json[topic]:
                    topic_score = path["topic_entity_score"]
                    reward = path["reward"][2]
                    prior_match_score = path["prior_match_score"]
                    if include_prior:
                        path_reward = reward + (prior_match_score * 0.75)
                    else:
                        path_reward = reward
                    if path_reward > threshold and topic_score > topic_threshold:
                        path["derived_label"] = 1
                    else:
                        path["derived_label"] = 0
            with open(os.path.join(dest_dir, r), 'w+') as fp:
                json.dump(ques_json, fp, indent=4)

    def downsample(self, src_dir, dest_dir, kb_prop=0.5):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        reward_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
        for r in reward_files:
            ques_json = json.load(open(os.path.join(src_dir, r)))
            new_cands = {}
            for topic in ques_json.keys():
                keyset = set()
                to_keep = []
                for path in ques_json[topic]:
                    keyset.add(str(path["relations"]))
                subset_keyset = random.sample(keyset, int(kb_prop * len(keyset)))
                for path in ques_json[topic]:
                    if str(path["relations"]) in subset_keyset:
                        to_keep.append(path)
                new_cands[topic]= to_keep
            with open(os.path.join(dest_dir, r), 'w+') as fp:
                json.dump(new_cands, fp, indent=4)


    def estimate_upper_bound_f1(self, input_path, src_dir):
        questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
        all_f1s = []
        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            ground_answers = q["Answers"]
            if type == "conjunction" or type == "composition":
                ques_path = os.path.join(src_dir, ques_id + ".json")
                if not os.path.exists(ques_path):
                    continue
                print(ques_id)
                ques_json = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
                max_f1 = 0.0
                for topic in ques_json.keys():
                    for path in ques_json[topic]:
                        if path["src"] == "sub2" and path["approx_label"] == 1:
                            f1 = metricUtils.compute_f1(ground_answers, path["entities"])
                            max_f1 = max(f1[2], max_f1)
                        if path["src"] == "sub1" and path["approx_label"] == 1:
                            f1 = metricUtils.compute_f1(ground_answers, path["entities"])
                            max_f1 = max(f1[2], max_f1)
                all_f1s.append(max_f1)
        macro_avg_f1 = np.mean(all_f1s)
        print(macro_avg_f1)

    def get_gold_answers(self, ques_src, dest):
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
        for q in questions:
            ques_id = q["ID"]
            sparql = q["sparql"].replace("\n", " ")
            results = self.sparql.execute(sparql)[u"results"][u"bindings"]
            answer_strings  = []
            if results is not None:
                mids =  [self.sparql.remove_uri_prefix(r["x"]["value"]) for r in results]
                for mid in mids:
                    name = self.sparql.get_names(mid)
                    if name is not None:
                        answer_strings.append(name)
            print("{} has answers {}".format(ques_id, len(answer_strings)))
            q["Answers_names"] = answer_strings
        with open(dest, 'w+') as fp:
            json.dump(questions, fp, indent=4)

    def deduplicated_gold_answers(self, ques_src, dest):
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
        for q in questions:
            ques_id = q["ID"]
            if len(q["Answers"]) != len(q["Answers_names"]):
                print(ques_id)

if __name__ == '__main__':
    trainDataGen = TrainDataGenerator()
    # print(trainDataGen.relation_tokens(['religion.religious_leadership_in_jurisdiction.leader']))

    # deduplicate candidates
    # trainDataGen.deduplicate(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_PATH)

    # add approx labels field
    # trainDataGen.get_expected_labels(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH)
    # trainDataGen.expected_labels_stats(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH)

    # add topic scores
    # trainDataGen.get_topic_scores(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH, TOPIC_PATH)

    # eval no rescaled rewards
    # trainDataGen.evaluate_rescaled_threshold(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH)

    # rescaled rewards
    # trainDataGen.rescale_rewards_max(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PATH)
    # trainDataGen.evaluate_rescaled_threshold(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PATH,threshold=0.1)

    # trainDataGen.rescale_reward_zscore(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_Z_SCORE_PATH)
    # trainDataGen.evaluate_rescaled_threshold(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_Z_SCORE_PATH, threshold=0.5)

    # estimate priors
    # trainDataGen.estimate_match_priors(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_Z_SCORE_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_Z_SCORE_PRIORS_PATH)
    # trainDataGen.estimate_match_priors(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH)

    # print(trainDataGen.get_prior_match_threshold(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH))
    # print(trainDataGen.get_threshold(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH)) # 1.23 TP, 0.23
    # print(trainDataGen.get_scaled_reward_threshold(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH)) # 0.73
    # print(trainDataGen.get_prior_topic_threshold(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH))

    # trainDataGen.evaluate_rescaled_prior_match_threshold(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH, threshold=0.5)

    # estimate co-occur prior
    # trainDataGen.estimate_co_occur_priors(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH, "lexicon")

    # trainDataGen.add_derived_labels(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH,
    #                                 CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_1_PATH,
    #                                 threshold=0.1, include_prior=True)
    # trainDataGen.add_derived_labels(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH,
    #                                 CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_5_PATH,
    #                                 threshold=0.5, include_prior=True)
    # trainDataGen.add_derived_labels(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH,
    #                                 CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_DERIVED_1_PATH,
    #                                 threshold=0.1, include_prior=False)
    # trainDataGen.add_derived_labels(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH,
    #                                 CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_DERIVED_5_PATH,
    #                                 threshold=0.5, include_prior=False)

    # lemmatizer = WordNetLemmatizer()
    # print(lemmatizer.lemmatize("produced"))
    # print(lemmatizer.lemmatize("producer"))
    # trainDataGen.estimate_upper_bound_f1(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH)

    # trainDataGen.get_gold_answers(os.path.join(PREFIX, "annotated/train.json"), os.path.join(PREFIX, "annotated-names/train.json"))
    # trainDataGen.deduplicated_gold_answers(os.path.join(PREFIX, "annotated-names/train.json"),os.path.join(PREFIX, "annotated-names/train.json"))
    ### only for test
    # trainDataGen.deduplicate(INPUT_PATH, CANDIDATE_SUB1_DEST_PATH, CANDIDATE_SUB1_REWARDS_PATH)
    # trainDataGen.get_expected_labels(INPUT_PATH, CANDIDATE_SUB1_REWARDS_PATH, CANDIDATE_SUB1_REWARDS_PATH)
    # trainDataGen.estimate_match_priors(INPUT_PATH, CANDIDATE_SUB1_REWARDS_PATH, CANDIDATE_SUB1_REWARDS_PATH)
    # trainDataGen.estimate_co_occur_priors(INPUT_PATH,CANDIDATE_SUB1_REWARDS_PATH,CANDIDATE_SUB1_REWARDS_PATH,"lexicon")
    # trainDataGen.get_topic_scores(INPUT_PATH, CANDIDATE_SUB1_REWARDS_PATH, CANDIDATE_SUB1_REWARDS_PATH, TOPIC_PATH)
    #
    # trainDataGen.deduplicate(INPUT_PATH, CANDIDATE_SUB2_DEST_PATH, CANDIDATE_SUB2_REWARDS_PATH)
    # trainDataGen.get_expected_labels(INPUT_PATH, CANDIDATE_SUB2_REWARDS_PATH, CANDIDATE_SUB2_REWARDS_PATH)
    # trainDataGen.estimate_match_priors(INPUT_PATH, CANDIDATE_SUB2_REWARDS_PATH, CANDIDATE_SUB2_REWARDS_PATH)
    # trainDataGen.estimate_co_occur_priors(INPUT_PATH, CANDIDATE_SUB2_REWARDS_PATH, CANDIDATE_SUB2_REWARDS_PATH, "lexicon")
    # trainDataGen.get_topic_scores(INPUT_PATH, CANDIDATE_SUB2_REWARDS_PATH, CANDIDATE_SUB2_REWARDS_PATH, TOPIC_PATH)

    ### only for dev
    # trainDataGen.deduplicate(INPUT_PATH, CANDIDATE_SUB1_DEST_PATH, CANDIDATE_SUB1_REWARDS_PATH)
    # trainDataGen.get_expected_labels(INPUT_PATH, CANDIDATE_SUB1_REWARDS_PATH, CANDIDATE_SUB1_REWARDS_PATH)
    # trainDataGen.estimate_match_priors(INPUT_PATH, CANDIDATE_SUB1_REWARDS_PATH, CANDIDATE_SUB1_REWARDS_PATH)
    # trainDataGen.estimate_co_occur_priors(INPUT_PATH,CANDIDATE_SUB1_REWARDS_PATH,CANDIDATE_SUB1_REWARDS_PATH,"lexicon")
    # trainDataGen.get_topic_scores(INPUT_PATH, CANDIDATE_SUB1_REWARDS_PATH, CANDIDATE_SUB1_REWARDS_PATH, TOPIC_PATH)
    #
    # trainDataGen.deduplicate(INPUT_PATH, CANDIDATE_SUB2_DEST_PATH, CANDIDATE_SUB2_REWARDS_PATH)
    # trainDataGen.get_expected_labels(INPUT_PATH, CANDIDATE_SUB2_REWARDS_PATH, CANDIDATE_SUB2_REWARDS_PATH)
    # trainDataGen.estimate_match_priors(INPUT_PATH, CANDIDATE_SUB2_REWARDS_PATH, CANDIDATE_SUB2_REWARDS_PATH)
    # trainDataGen.estimate_co_occur_priors(INPUT_PATH, CANDIDATE_SUB2_REWARDS_PATH, CANDIDATE_SUB2_REWARDS_PATH, "lexicon")
    # trainDataGen.get_topic_scores(INPUT_PATH, CANDIDATE_SUB2_REWARDS_PATH, CANDIDATE_SUB2_REWARDS_PATH, TOPIC_PATH)

    ### for complete test
    # trainDataGen.deduplicate(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_PATH)
    # trainDataGen.get_expected_labels(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH)
    # trainDataGen.estimate_match_priors(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH)
    # trainDataGen.estimate_co_occur_priors(INPUT_PATH,CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH,CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH,"lexicon")
    # trainDataGen.get_topic_scores(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH, TOPIC_PATH)
    # trainDataGen.add_derived_labels(CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH,
    #                                 CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_5_PATH,
    #                                 threshold=0.5, include_prior=True)
    # trainDataGen.downsample("/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess/rewards/test/sub1", "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess/rewards/test_0.25/sub1", kb_prop=0.25)
    trainDataGen.expected_labels_stats("/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess/rewards/test_0.75/sub1")