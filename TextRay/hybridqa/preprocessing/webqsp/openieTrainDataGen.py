import os
import json
import codecs
import pandas as pd
from kbEndPoint.utils.sparql import sparqlUtils
from preprocessing.complexq.el_helper import EL,EL_helper
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # will do for now?
import copy
import numpy as np
import operator
from collections import defaultdict as ddict

class OpenieTrainDataGen(object):

    def __init__(self):
        self.sparql = sparqlUtils()
        self.entity_names_cache = {}
        self.linker = EL_helper()

    def get_openie_cands(self, ques_src, linked_triples_src, kb_train_data_src, relation_mapping_src, open_train_data_src):
        print linked_triples_src
        linked_triple_data = pd.read_json(linked_triples_src, orient='records')
        print linked_triple_data.shape
        if not os.path.exists(open_train_data_src):
            os.makedirs(open_train_data_src)
        if not os.path.exists(relation_mapping_src):
            os.makedirs(relation_mapping_src)

        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))["Questions"]
        for q in questions:
            ques_id = q["QuestionId"]
            kb_train_data_path = os.path.join(kb_train_data_src, ques_id + ".json")
            open_data_path = os.path.join(open_train_data_src, ques_id + ".json")
            relation_mapping_path = os.path.join(relation_mapping_src, ques_id + ".csv")
            train_data = json.load(codecs.open(kb_train_data_path, 'r', encoding='utf-8'))
            if os.path.exists(open_data_path):
                continue
            print ques_id
            topics_1 = train_data.keys()
            cands = linked_triple_data[(linked_triple_data['subject_mid'].isin(topics_1))
                                                        | (linked_triple_data['object_mid'].isin(topics_1))].to_dict('records')
            open_cands = {}
            equivalence_records = []
            dedup_records = []
            open_cands, equivalence_records = self.get_cands(cands, train_data, equivalence_records, open_cands)

            dedup_keys = set()

            for e in equivalence_records:
                key = e["open_relation"] + '_' + str(e['kb_relation'])
                if key in dedup_keys:
                    continue
                dedup_keys.add(key)
                dedup_records.append(e)
            df = pd.DataFrame.from_records(dedup_records)
            df.to_csv(relation_mapping_path, index=None, encoding='utf-8')
            with open(os.path.join(open_data_path), 'w+') as fp:
                json.dump(open_cands, fp, indent=4)

    def get_cands(self, sub1_cands, train_data, equivalence_records, open_cands):

        for cand in sub1_cands:
            subject_mid = cand['subject_mid']
            object_mid = cand['object_mid']

            topic_mid = subject_mid
            ans_mid = object_mid
            is_reverse = False
            if subject_mid in train_data:
                is_reverse = False
                topic_mid = subject_mid
                ans_mid = object_mid
            elif object_mid in train_data:
                is_reverse = True
                topic_mid = object_mid
                ans_mid = subject_mid

            kb_cands = train_data[topic_mid]
            equivalent_kb_paths = [kb_cand for kb_cand in kb_cands if ans_mid in kb_cand['entities']]
            equivalent_kb_records = [self.to_equivalence_record(cand, kb_cand, is_reverse) for kb_cand in
                                     equivalent_kb_paths]
            equivalence_records += equivalent_kb_records
            best_path = None
            best_reward = -1
            for path in equivalent_kb_paths:
                reward = path['reward'][2]
                if reward > best_reward:
                    best_path = path
                    best_reward = reward
            open_cand = self.to_cand(cand, best_path, is_reverse=is_reverse)
            topic = open_cand['topic_entity']
            open_topic_cands = open_cands.get(topic, [])
            open_topic_cands.append(open_cand)
            open_cands[topic] = open_topic_cands
        return open_cands, equivalence_records

    def to_equivalence_record(self, cand, path, is_reverse):
        return {
            "open_relation": cand["relation"],
            "is_reverse_open_relation": is_reverse,
            "kb_relation": path["relations"],
            "is_reverse_kb_relation": False
        }

    def to_cand(self, cand, best_kb_cand, is_reverse):
        topic_prefix = "subject"
        ans_prefix = "object"
        if is_reverse:
            topic_prefix = "object"
            ans_prefix = "subject"
        reward = [0.0, 0.0, 0.0]
        true_label = 0
        if best_kb_cand is not None and 'true_label' in best_kb_cand:
            reward = best_kb_cand['reward']
            true_label = best_kb_cand['true_label']
            if true_label == 1:
                print("{} == {}".format(cand['relation'], best_kb_cand['relations']))
        return {
            "topic_entity": cand[topic_prefix + "_mid"],
            "topic_entity_span": cand[topic_prefix + '_span'],
            "topic_entity_mention": cand[topic_prefix + '_mention'],
            "topic_entity_score": cand[topic_prefix + '_score'],
            "relations": [cand["relation"]],
            "entities": [cand[ans_prefix + "_mid"]],
            "entities_span": [cand[ans_prefix + '_span']],
            "entities_mention": [cand[ans_prefix + '_mention']],
            "entities_score": [cand[ans_prefix + '_score']],
            "true_label": true_label,
            "reward": reward,
            "is_reverse": is_reverse,
        }

    def filter_open_cands(self, open_train_data_src, open_train_data_filter_src, rel_mapping_src, rel_mapping_filter_src, threshold=0.75):
        files = os.listdir(open_train_data_src)
        files = [f for f in files if f.endswith('.json')]
        if not os.path.exists(open_train_data_filter_src):
            os.makedirs(open_train_data_filter_src)
        if not os.path.exists(rel_mapping_filter_src):
            os.makedirs(rel_mapping_filter_src)
        for f in files:
            print f
            ques_id = f.replace(".json", "")
            headers = ['open_relation', 'is_reverse_open_relation', 'kb_relation', 'is_reverse_kb_relation']
            rel_mapping_records = pd.DataFrame(columns=headers)
            rel_path = os.path.join(rel_mapping_src, ques_id + ".csv")
            try:
                rel_mapping_records = pd.read_csv(rel_path)
            except:
                rel_mapping_records = pd.DataFrame(columns=headers)
            cands = json.load(codecs.open(os.path.join(open_train_data_src, f), 'r', encoding='utf-8'))
            filter_cands = {}
            filtered_rels = set()
            for topic in cands:
                filtered_paths = []
                for path in cands[topic]:
                    is_valid_topic = self.is_valid_argument(path['topic_entity_span'], path['topic_entity_mention'], threshold=threshold)
                    is_valid_ans = self.is_valid_argument(path['entities_span'][0], path['entities_mention'][0], threshold=threshold)
                    if is_valid_topic and is_valid_ans:
                        filtered_paths.append(path)
                        filtered_rels.add(path['relations'][0])
                filter_cands[topic] = filtered_paths
            rel_mapping_filter_records = rel_mapping_records[(rel_mapping_records['open_relation'].isin(filtered_rels))]

            open_data_path = os.path.join(open_train_data_filter_src, ques_id + ".json")
            relation_mapping_path = os.path.join(rel_mapping_filter_src, ques_id + ".csv")

            rel_mapping_filter_records.to_csv(relation_mapping_path, index=None, encoding='utf-8')

            with open(os.path.join(open_data_path), 'w+') as fp:
                json.dump(filter_cands, fp, indent=4)


    def is_valid_argument(self, span, mention, threshold=0.75):
        return float(len(mention)) * 1.0 / float(len(span)) > threshold

    def analyze_open_cands(self, open_train_data_src):
        files = os.listdir(open_train_data_src)
        files = [f for f in files if f.endswith('.json')]
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        cands_ct = []
        pos_ct = []
        ques_with_pos_ct = 0
        open_relations = set()
        for f in files:
            # print f
            cands = json.load(codecs.open(os.path.join(open_train_data_src, f), 'r', encoding='utf-8'))
            ct = 0
            pos = 0
            for topic in cands:
                for path in cands[topic]:
                    # open_rel = path['relations'][0]
                    # ques_tokens = [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(open_rel.lower()) if w not in stop_words]
                    # open_rel = ' '.join(ques_tokens)
                    # if len(open_rel) > 2:
                    #     open_relations.add(open_rel.lower())
                    ct += 1
                    if path['true_label'] == 1:
                        pos += 1
                        # print f
            pos_ct.append(pos)
            if pos > 0:
                ques_with_pos_ct += 1
            cands_ct.append(ct)
        print(np.mean(cands_ct))
        print(np.mean(pos_ct))
        print(len(open_relations))
        print(ques_with_pos_ct)
        print(len(files))

    def read_cluster_map(self, cluster_src):
        rep2clust = ddict(list)
        with open(cluster_src) as f:
            for line in f:
                if not line.startswith('\t'):
                    rep = line.strip()
                else:
                    rep2clust[rep].append(line.strip())
        print(len(rep2clust))
        return rep2clust


    def normalize_open_cands(self, open_train_data_src, cluster_src, normalized_open_train_data_src):
        lemmatizer = WordNetLemmatizer()
        if not os.path.exists(normalized_open_train_data_src):
            os.makedirs(normalized_open_train_data_src)
        cluster_map = self.read_cluster_map(cluster_src)
        rev_cluster_map = {}
        for key, items in cluster_map.items():
            for item in items:
                rev_cluster_map[item] = key
        files = os.listdir(open_train_data_src)
        files = [f for f in files if f.endswith('.json')]
        rel_vocab = set()
        for f in files:
            print f
            cands = json.load(codecs.open(os.path.join(open_train_data_src, f), 'r', encoding="ascii", errors="ignore"))
            cands_update = {}
            for topic in cands:
                new_cands = []
                for path in cands[topic]:
                    subject = path["topic_entity_mention"].encode("ASCII", 'ignore')
                    object = path["entities_mention"][0].encode("ASCII", 'ignore')
                    relation = path['relations'][0].encode("ASCII", 'ignore')
                    if len(relation) == 0 or len(subject) == 0 or len(object) == 0:
                        continue


                    # path['unnormalized_relations'] = [r for r in path['relations']]
                    unnormalized_relations = []
                    normalized_relations = []
                    for r in path['relations']:
                        r = r.lower()
                        if not r in rev_cluster_map:
                            # print r
                            r = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(r)])
                            if r in rev_cluster_map:
                                unnormalized_relations.append(r)
                                normalized_relations.append(rev_cluster_map[r])
                            else:
                                if path['true_label'] == 1:
                                    print topic
                                    print path
                                break
                        else:
                            unnormalized_relations.append(r)
                            normalized_relations.append(rev_cluster_map[r])
                    if len(normalized_relations) > 0:
                        path['relations'] = normalized_relations
                        path['unnormalized_relations'] = unnormalized_relations
                        new_cands.append(path)
                        rel_vocab.add(normalized_relations[0])
                cands_update[topic] = new_cands

            with codecs.open(os.path.join(normalized_open_train_data_src, f), 'w+', encoding='ascii') as fp:
                json.dump(cands_update, fp, indent=4)
        print("{} unique relations found".format(len(rel_vocab)))

    def deduplicate(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        files = os.listdir(src)
        files = [f for f in files if f.endswith('.json')]
        counter = 0
        original_count = 0
        dedup_count = 0
        for f in files:
            # print f
            counter += 1
            if counter % 100 == 0:
                print counter
            cands = json.load(codecs.open(os.path.join(src, f), 'r', encoding="ascii", errors="ignore"))
            dedup_cands = {}
            for topic in cands:
                deduplicated = []
                entity_consolidation_map = {}
                deduplicated_map = {}

                for path in cands[topic]:
                    original_count += 1
                    path_key = str(path['relations']) + "_" + str(path['entities']) + "_" + str(path['topic_entity'])
                    prev_len = entity_consolidation_map.get(path_key, 0.0)
                    entity_len = len((path['entities_span'])) + len((path['topic_entity_span']))
                    if entity_len > prev_len:
                        entity_consolidation_map[path_key] = entity_len
                        deduplicated_map[path_key] = path
                for key, path in deduplicated_map.items():
                    dedup_count += 1
                    deduplicated.append(path)
                dedup_cands[topic] = deduplicated

            with codecs.open(os.path.join(dest, f), 'w+', encoding='ascii') as fp:
                json.dump(dedup_cands, fp, indent=4)
        print original_count
        print dedup_count

    def filter_mapping(self, cluster_src, relation_mapping_src, relation_mapping_tgt):
        lemmatizer = WordNetLemmatizer()
        if not os.path.exists(relation_mapping_tgt):
            os.makedirs(relation_mapping_tgt)
        cluster_map = self.read_cluster_map(cluster_src)
        rev_cluster_map = {}
        for key, items in cluster_map.items():
            for item in items:
                rev_cluster_map[item] = key
        files = os.listdir(relation_mapping_src)
        files = [f for f in files if f.endswith('.csv')]
        for f in files:
            print f
            df = pd.read_csv(os.path.join(relation_mapping_src, f)).to_dict('records')
            records = []
            for r in df:
                openie_rel = r['open_relation'].decode('utf-8').encode("ASCII", 'ignore')
                openie_rel = openie_rel.lower()
                if openie_rel in rev_cluster_map:
                    r['open_relation'] = openie_rel
                    records.append(r)
                else:
                    openie_rel = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(openie_rel)])
                    if openie_rel in rev_cluster_map:
                        r['open_relation'] = openie_rel
                        records.append(r)
            print(len(records))
            if len(records) == 0:
                headers = ['open_relation', 'is_reverse_open_relation', 'kb_relation', 'is_reverse_kb_relation']
                df_to_write = pd.DataFrame(columns=headers)
            else:
                df_to_write = pd.DataFrame.from_records(records)
            df_to_write.to_csv(os.path.join(relation_mapping_tgt, f), index=None, header=True, encoding='utf-8')

    def check_alignment(self, cluster_src, relation_mapping_dir):
        cluster_map = self.read_cluster_map(cluster_src)
        rev_cluster_map = {}
        for key, items in cluster_map.items():
            for item in items:
                rev_cluster_map[item] = key
        files = os.listdir(relation_mapping_dir)
        files = [f for f in files if f.endswith('.csv')]
        rel_align_ct = {}
        norm_rel_align_ct = {}
        for f in files:
            records = pd.read_csv(os.path.join(relation_mapping_dir, f)).to_dict(orient='records')
            for r in records:
                kb_relation = r['kb_relation']
                # if kb_relation == "[location.location.contains]":
                #     print f
                open_relation = r['open_relation']
                normalized_relation = rev_cluster_map[open_relation]
                rel_align_info = rel_align_ct.get(kb_relation, set())
                rel_align_info.add(open_relation)
                rel_align_ct[kb_relation] = rel_align_info

                norm_rel_align_info = norm_rel_align_ct.get(kb_relation, set())
                norm_rel_align_info.add(normalized_relation)
                norm_rel_align_ct[kb_relation] = norm_rel_align_info

        max_ct = 0
        max_entry = 0
        for rel, val in rel_align_ct.items():
            if len(val) > max_ct:
                max_ct = len(val)
                max_entry = rel
        print max_entry
        print max_ct
        print rel_align_ct[max_entry]

        max_ct = 0
        max_entry = 0
        for rel, val in norm_rel_align_ct.items():
            if len(val) > max_ct:
                max_ct = len(val)
                max_entry = rel
        print max_entry
        print max_ct
        print norm_rel_align_ct[max_entry]
        # print(max(rel_align_ct.iteritems(), key=operator.itemgetter(1)))
        # print(max(norm_rel_align_ct.iteritems(), key=operator.itemgetter(1)))


    def prune_relation_mapping(self, cluster_src, relation_mapping_src, relation_mapping_dest, train_data_dir):
        if not os.path.exists(relation_mapping_dest):
            os.makedirs(relation_mapping_dest)
        cluster_map = self.read_cluster_map(cluster_src)
        rev_cluster_map = {}
        for key, items in cluster_map.items():
            for item in items:
                rev_cluster_map[item] = key
        files = os.listdir(relation_mapping_src)
        files = [f for f in files if f.endswith('.csv')]
        support_threshold = 5
        open_rel_ct = {}
        for f in os.listdir(train_data_dir):
            train_data = json.load(codecs.open(os.path.join(train_data_dir, f.replace(".csv", ".json")), 'r'))
            for topic in train_data:
                for path in train_data[topic]:
                    open_rel = path['unnormalized_relations'][0]
                    rel_ct = open_rel_ct.get(open_rel, 0) + 1
                    open_rel_ct[open_rel] = rel_ct
        # print open_rel_ct

        support_map = {}
        for f in files:
            records = pd.read_csv(os.path.join(relation_mapping_src, f)).to_dict(orient='records')
            for r in records:
                kb_relation = r['kb_relation']
                open_relation = r['open_relation']
                support_set = support_map.get(kb_relation, {})
                support_ct = support_set.get(open_relation, 0)
                support_ct += 1
                support_set[open_relation] = support_ct
                support_map[kb_relation] = support_set

        pruned_support_map = set()
        for kb_relation, open_relations_ct in support_map.items():
            sorted_open_relations = sorted(open_relations_ct.items(), key=operator.itemgetter(1), reverse=True)
            sorted_open_relations = sorted_open_relations[:min(6, len(sorted_open_relations))]
            for open_tuple in sorted_open_relations:
                key = kb_relation + "_" + open_tuple[0]
                pruned_support_map.add(key)
                print key

        print(len(pruned_support_map))
        for f in files:
            # print f
            records = pd.read_csv(os.path.join(relation_mapping_src, f)).to_dict(orient='records')
            records_to_keep = []
            for r in records:
                kb_relation = r['kb_relation']
                open_relation = r['open_relation']
                key = kb_relation + "_" + open_relation
                if key in pruned_support_map:
                # if open_rel_ct[open_relation] >= support_threshold:
                    records_to_keep.append(r)
            # print(len(records_to_keep))
            if len(records_to_keep) == 0:
                headers = ['open_relation', 'is_reverse_open_relation', 'kb_relation', 'is_reverse_kb_relation']
                df_to_write = pd.DataFrame(columns=headers)
            else:
                df_to_write = pd.DataFrame.from_records(records_to_keep)
            df_to_write.to_csv(os.path.join(relation_mapping_dest, f), index=None, header=True, encoding='utf-8')
        self.check_alignment(cluster_src, relation_mapping_dest)


if __name__ == '__main__':
    # DATA_PREFIX = "/home/nbhutani/workspace/TextRay/datasets/WebQSP-final"
    DATA_PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final"
    SPLIT = "test"
    QUES_SRC = os.path.join(DATA_PREFIX, "data/WebQSP." + SPLIT + ".json")
    LINKED_TRIPLES_SRC = os.path.join(DATA_PREFIX, "stanfordie", 'all', 'links', 'triple_cands',  "all.json")
    KB_TRAIN_DATA_SRC = os.path.join(DATA_PREFIX, "cands_with_constraints-scaled-" + SPLIT)

    OPEN_TRAIN_DATA_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "cands", 'train_cands', SPLIT)
    RELATION_MAPPING_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "cands", 'relation_mapping', SPLIT)

    OPEN_TRAIN_DATA_FILTER_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "prior_cands_0.3", 'train_cands', SPLIT)
    RELATION_MAPPING_FILTER_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "prior_cands_0.3", 'relation_mapping',
                                               SPLIT)


    trainGen = OpenieTrainDataGen()
    # trainGen.get_snippets_coverage(QUES_SRC, SNIPPETS_SRC, TRAIN_DATA_SRC)
    # trainGen.get_relevant_triples_coverage(QUES_SRC, LINKED_TRIPLES_SRC, KB_TRAIN_DATA_SRC)

    # trainGen.get_openie_cands(ques_src=QUES_SRC, linked_triples_src=LINKED_TRIPLES_SRC, kb_train_data_src=KB_TRAIN_DATA_SRC, relation_mapping_src=RELATION_MAPPING_SRC, open_train_data_src=OPEN_TRAIN_DATA_SRC)
    # trainGen.analyze_open_cands(open_train_data_src=OPEN_TRAIN_DATA_SRC)
    # trainGen.filter_open_cands(open_train_data_src=OPEN_TRAIN_DATA_SRC,
    #                            open_train_data_filter_src=OPEN_TRAIN_DATA_FILTER_SRC,
    #                            rel_mapping_src=RELATION_MAPPING_SRC,
    #                            rel_mapping_filter_src=RELATION_MAPPING_FILTER_SRC,
    #                            threshold=0.3)

    # trainGen.analyze_open_cands(open_train_data_src=OPEN_TRAIN_DATA_FILTER_SRC)
    #
    # # KE_PREFIX = '/home/nbhutani/workspace/knowledge_embeddings/cesi/output/'
    # KE_PREFIX = '/media/nbhutani/Data/textray_workspace/knowledge_embeddings/cesi/output'
    # CLUSTER_SRC = os.path.join(KE_PREFIX, 'webqsp_train_run_chunk/cluster_rel.txt')
    # OPEN_TRAIN_DATA_NORMALZED_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands", 'train_cands_lemmatized', SPLIT)
    #
    #
    # # trainGen.normalize_open_cands(OPEN_TRAIN_DATA_FILTER_SRC, CLUSTER_SRC, OPEN_TRAIN_DATA_NORMALZED_SRC)
    # # trainGen.analyze_open_cands(open_train_data_src=OPEN_TRAIN_DATA_NORMALZED_SRC)
    #
    # OPEN_TRAIN_DATA_NORMALZED_DEST = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands",
    #                                               'train_cands_lemmatized', SPLIT + "-dedup")
    # # trainGen.deduplicate(OPEN_TRAIN_DATA_NORMALZED_SRC, OPEN_TRAIN_DATA_NORMALZED_DEST)
    # # trainGen.analyze_open_cands(open_train_data_src=OPEN_TRAIN_DATA_NORMALZED_DEST)
    #
    # MAPPING_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands", 'relation_mapping',SPLIT)
    # MAPPING_DEST = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands", 'relation_mapping_lemmatized', SPLIT)
    # trainGen.filter_mapping(CLUSTER_SRC, MAPPING_SRC, MAPPING_DEST)


    '''-----webqsp fix--------'''
    LINKED_TRIPLES_SRC = os.path.join(DATA_PREFIX, "stanfordie", 'all_final', 'links', 'triple_cands', "all.json")
    KB_TRAIN_DATA_SRC = os.path.join(DATA_PREFIX, "cands_with_constraints-scaled-" + SPLIT)

    OPEN_TRAIN_DATA_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all_final", "cands", 'train_cands', SPLIT)
    RELATION_MAPPING_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all_final", "cands", 'relation_mapping', SPLIT)

    OPEN_TRAIN_DATA_FILTER_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all_final", "prior_cands_0.3", 'train_cands', SPLIT)
    RELATION_MAPPING_FILTER_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all_final", "prior_cands_0.3", 'relation_mapping',
                                               SPLIT)

    # trainGen.get_openie_cands(ques_src=QUES_SRC, linked_triples_src=LINKED_TRIPLES_SRC,
    #                           kb_train_data_src=KB_TRAIN_DATA_SRC, relation_mapping_src=RELATION_MAPPING_SRC,
    #                           open_train_data_src=OPEN_TRAIN_DATA_SRC)

    # trainGen.filter_open_cands(open_train_data_src=OPEN_TRAIN_DATA_SRC,
    #                            open_train_data_filter_src=OPEN_TRAIN_DATA_FILTER_SRC,
    #                            rel_mapping_src=RELATION_MAPPING_SRC,
    #                            rel_mapping_filter_src=RELATION_MAPPING_FILTER_SRC,
    #                            threshold=0.3)

    KE_PREFIX = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final/stanfordie/all/cluster_info'
    CLUSTER_SRC = os.path.join(KE_PREFIX, 'cluster_rel.txt')
    OPEN_TRAIN_DATA_NORMALZED_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all_final", "normalized_cands", 'train_cands_lemmatized', SPLIT)
    # trainGen.normalize_open_cands(OPEN_TRAIN_DATA_FILTER_SRC, CLUSTER_SRC, OPEN_TRAIN_DATA_NORMALZED_SRC)

    MAPPING_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all_final", "prior_cands_0.3", 'relation_mapping',SPLIT)
    MAPPING_DEST = os.path.join(DATA_PREFIX, "stanfordie", "all_final", "normalized_cands", 'relation_mapping_lemmatized', SPLIT)
    # trainGen.filter_mapping(CLUSTER_SRC, MAPPING_SRC, MAPPING_DEST)

    OPEN_TRAIN_DATA_NORMALZED_DEST = os.path.join(DATA_PREFIX, "stanfordie", "all_final", "normalized_cands",
                                                  'train_cands_lemmatized', SPLIT + "-dedup")
    # trainGen.deduplicate(OPEN_TRAIN_DATA_NORMALZED_SRC, OPEN_TRAIN_DATA_NORMALZED_DEST)
    trainGen.analyze_open_cands(open_train_data_src=OPEN_TRAIN_DATA_NORMALZED_DEST)

    #trainGen.check_alignment(CLUSTER_SRC, MAPPING_DEST)
    MAPPING_DEST_PRUNED = os.path.join(DATA_PREFIX, "stanfordie", "all_final", "normalized_cands",
                                'relation_mapping_lemmatized_pruned', SPLIT)
    # trainGen.prune_relation_mapping(CLUSTER_SRC, MAPPING_DEST, MAPPING_DEST_PRUNED, OPEN_TRAIN_DATA_NORMALZED_SRC)





