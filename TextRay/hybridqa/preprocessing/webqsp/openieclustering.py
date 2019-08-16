import os
import json
import codecs
import pandas as pd
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer # will do for now?

DATA_PREFIX = "/home/nbhutani/workspace/TextRay/datasets/WebQSP-final"
SPLIT = "test"

# TRAIN_DATA_SRC = os.path.join(DATA_PREFIX, "stanfordie", 'all', 'prior_cands_0.3', 'train_cands',  SPLIT)
# CLUSTER_DATA_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "cluster", SPLIT + "_0.3.json")
#
# CLUSTER_DATA_SRC_FILES = [os.path.join(DATA_PREFIX, "stanfordie", "all", "cluster", "train_0.3.json"),
#                           os.path.join(DATA_PREFIX, "stanfordie", "all", "cluster", "test_0.3.json")]
# ALL_CLUSTER_DATA_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "cluster", "all_0.3.json")

TRAIN_DATA_SRC = os.path.join(DATA_PREFIX, "stanfordie", 'all_v2', 'prior_cands_0.3', 'train_cands',  SPLIT)
CLUSTER_DATA_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all_v2", "cluster", SPLIT + "_0.3.json")

CLUSTER_DATA_SRC_FILES = [os.path.join(DATA_PREFIX, "stanfordie", "all_v2", "cluster", "train_0.3.json"),
                          os.path.join(DATA_PREFIX, "stanfordie", "all_v2", "cluster", "test_0.3.json")]
ALL_CLUSTER_DATA_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all_v2", "cluster", "all_0.3.json")

def combine_data_for_clustering(cluster_data_src_files, cluster_data_src):
    cluster_data_fp = codecs.open(cluster_data_src, 'w+')
    counter = 0
    for cluster_data_src in cluster_data_src_files:
        with open(cluster_data_src) as fp:
            for line in fp:
                triple = json.loads(line.rstrip())
                counter += 1
                triple['_id'] = counter
                json_str = json.dumps(triple)
                cluster_data_fp.write(json_str + "\n")
    cluster_data_fp.close()


def prepare_data_for_clustering(train_data_src, cluster_data_src):
    lemmatizer = WordNetLemmatizer()
    if os.path.exists(cluster_data_src):
        os.remove(cluster_data_src)
    if not os.path.exists(os.path.dirname(cluster_data_src)):
        os.makedirs(os.path.dirname(cluster_data_src))
    files = os.listdir(train_data_src)
    files = [f for f in files if f.endswith(".json")]
    cluster_data_fp = codecs.open(cluster_data_src, 'w+')
    triple_set = set()
    counter = 0
    for f in files:
        ques_id = f.replace(".json", "")
        # print ques_id
        content = json.load(codecs.open(os.path.join(train_data_src, f), encoding="ascii", errors="ignore"))
        if counter % 1000 == 0:
            print counter
        for topic in content:
            for path in content[topic]:
                subject = path["topic_entity_mention"].encode("ASCII", 'ignore')
                object = path["entities_mention"][0].encode("ASCII", 'ignore')
                subject_id = path["topic_entity"].encode("ASCII", 'ignore')
                object_id = path["entities"][0].encode("ASCII", 'ignore')
                relation = path['relations'][0].encode("ASCII", 'ignore')

                if path["is_reverse"]:
                    object = path["topic_entity_mention"].encode("ASCII", 'ignore')
                    subject = path["entities_mention"][0].encode("ASCII", 'ignore')
                    object_id = path["topic_entity"].encode("ASCII", 'ignore')
                    subject_id = path["entities"][0].encode("ASCII", 'ignore')

                triple_key = "{}--{}--{}".format(subject, relation, object)
                if len(relation) == 0 or len(subject)  == 0 or len(object) == 0:
                    continue
                if triple_key in triple_set:
                    continue
                triple_set.add(triple_key)
                counter += 1
                triple_id = counter
                triple = {
                    "_id": triple_id,
                    "triple": [
                        subject,
                        relation,
                        object,
                    ],
                    "triple_norm": [
                        lemmatize(lemmatizer, subject),
                        lemmatize(lemmatizer, relation),
                        lemmatize(lemmatizer, object)
                    ],
                    "true_link": {
                        "subject": subject_id,
                        "object": object_id
                    },
                    "src_sentences": [],
                    "entity_linking": {
                        "subject": subject_id,
                        "object": object_id
                    },
                    "kbp_info": []
                }
                json_str = json.dumps(triple)
                cluster_data_fp.write(json_str + "\n")
    cluster_data_fp.close()

def lemmatize(lemmatizer, target_str):
    tokens = [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(target_str.lower())]
    return ' '.join(tokens)

if __name__ == '__main__':
    # prepare_data_for_clustering(TRAIN_DATA_SRC, CLUSTER_DATA_SRC)
    combine_data_for_clustering(CLUSTER_DATA_SRC_FILES, ALL_CLUSTER_DATA_SRC)