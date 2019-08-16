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


INPUT_PATH = os.path.join(PREFIX, "annotated/train.json")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_PATH = os.path.join(PREFIX, "cands/train/main_with_constraints")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_PATH = os.path.join(PREFIX, "rewards/dedup/train")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_PATH = os.path.join(PREFIX, "rewards/labels/train")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PATH = os.path.join(PREFIX, "rewards/rescaled_max/train")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_Z_SCORE_PATH = os.path.join(PREFIX, "rewards/rescaled_zscore/train")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_Z_SCORE_PRIORS_PATH = os.path.join(PREFIX, "rewards/rescaled_zscore_priors/train")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors/train")
#
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_1_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors_derived_0.1/train")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_PRIORS_DERIVED_5_PATH = os.path.join(PREFIX, "rewards/rescaled_max_priors_derived_0.5/train")
# CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_DERIVED_1_PATH = os.path.join(PREFIX, "rewards/rescaled_max_derived_0.1/train")
CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_DERIVED_5_PATH = os.path.join(PREFIX, "rewards/rescaled_max_derived_0.5/train")
# TOPIC_PATH = os.path.join(PREFIX, "topic_entities/main/train_topic.csv")

def train_data_stats(input_path, src_dir):
    questions = json.load(codecs.open(input_path, 'r', encoding='utf-8'))
    positive_examples_ct = 0
    negative_examples_ct = 0
    for q in questions:
        ques_id = q["ID"]
        type = q["compositionality_type"]
        if type == "conjunction" or type == "composition":
            ques_path = os.path.join(src_dir, ques_id + ".json")
            if not os.path.exists(ques_path):
                continue

            #print(ques_id)
            main_entity_paths = json.load(codecs.open(ques_path, 'r', encoding='utf-8'))
            for topic in main_entity_paths:
                for path in main_entity_paths[topic]:
                    if path["derived_label"] == 0:
                        negative_examples_ct += 1
                    elif path["derived_label"] == 1:
                        positive_examples_ct += 1
    print(positive_examples_ct)
    print(negative_examples_ct)


if __name__ == '__main__':
    train_data_stats(INPUT_PATH, CANDIDATE_MAIN_WITH_CONSTRAINTS_DEST_REWARDS_DEDUP_LABEL_RESCALED_MAX_DERIVED_5_PATH)

