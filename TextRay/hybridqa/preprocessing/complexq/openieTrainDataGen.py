import os
import json
import codecs
import pandas as pd
from kbEndPoint.utils.sparql import sparqlUtils
from  el_helper import EL,EL_helper
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # will do for now?
import copy
import numpy as np
import csv
from collections import defaultdict as ddict
import ast

class OpenieTrainDataGen(object):

    def __init__(self):
        self.sparql = sparqlUtils()
        self.entity_names_cache = {}
        self.linker = EL_helper()

    def get_snippets_coverage(self, ques_src, snippet_src, train_data_src):
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
        snippets_df = pd.read_json(snippet_src, orient='records', encoding='utf-8')
        snippets_df = snippets_df[snippets_df['split_source'].str.join(' ').str.contains('ptrnet split')]
        snippet_pos_ct = 0
        total_ct = 0
        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                total_ct += 1
                snippets = snippets_df[snippets_df["question_ID"] == ques_id]
                snippet_part1 = snippets[snippets['split_type'] == 'split_part1'].to_dict('records')
                snippet_part2 = snippets[snippets['split_type'] == 'split_part2'].to_dict('records')
                train_data_path = os.path.join(train_data_src, ques_id + ".json")
                if not os.path.exists(train_data_path):
                    continue
                print ques_id
                train_data = json.load(codecs.open(train_data_path, 'r', encoding='utf-8'))
                sub1_pos = []
                sub2_pos = []
                for topic in train_data:
                    for path in train_data[topic]:
                        if path['approx_label'] == 0:
                            continue
                        path['topic_entity'] = topic
                        if path['src'] == 'sub1': sub1_pos.append(path)
                        elif path['src'] == 'sub2': sub2_pos.append(path)
                sub1_pos_ct = 0
                sub2_pos_ct = 0
                if type == 'composition': ##composition questions in dev/train set have the main topic entity as the root
                    sub2_pos_rev = []
                    for sub1 in sub1_pos:
                        for sub2 in sub2_pos:
                            if sub1['topic_entity'] == sub2['topic_entity']:
                                sub2_topics = sub1['entities']
                                for sub2_topic in sub2_topics:
                                    sub2_rev = copy.deepcopy(sub2)
                                    sub2_rev['topic_entity'] = sub2_topic
                                    sub2_pos_rev.append(sub2_rev)
                    sub2_pos = sub2_pos_rev
                for sub1 in sub1_pos:
                    # print sub1
                    # print len(snippet_part1)
                    if self.has_positive_snippet(sub1, snippet_part1):
                        sub1_pos_ct += 1
                        break
                for sub2 in sub2_pos:
                    # print sub2
                    # print len(snippet_part2)
                    if self.has_positive_snippet(sub2, snippet_part2):
                        sub2_pos_ct += 1
                        break
                if sub1_pos_ct > 0 and sub2_pos_ct > 0:
                    print('found hit')
                    snippet_pos_ct += 1

        print("{} of {} has matches for snippets".format(snippet_pos_ct, total_ct))

    def has_positive_snippet(self, train_data_cand, snippet_part):
        topic_mid = train_data_cand['topic_entity']
        target_mids = train_data_cand['entities']
        # print topic_mid
        # print target_mids
        for snippet_record in snippet_part:
            for web_snippet in snippet_record['web_snippets']:
                has_topic = False
                has_target = False
                snippet_text = web_snippet['snippet']
                # print snippet_text
                linked_entities = self.linker.link_text(snippet_text)
                for l in linked_entities:
                    mid = l.entity.sparql_name()
                    # print mid
                    if mid == topic_mid:
                        has_topic = True
                        # print('mid matches topic')
                    if mid in target_mids:
                        has_target = True
                        # print('mid is target')
                if has_topic and has_target:
                    return True
        return False

    def get_relevant_triples_coverage(self,ques_src, linked_triples_src, train_data_src):
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
        triple_pos_ct = 0
        total_ct = 0
        at_least_one_pos_ct = 0
        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                train_data_path = os.path.join(train_data_src, ques_id + ".json")
                if not os.path.exists(train_data_path):
                    continue
                linked_triple_path = os.path.join(linked_triples_src, ques_id + ".json")
                if not os.path.exists(linked_triple_path):
                    continue
                print ques_id
                total_ct += 1
                train_data = json.load(codecs.open(train_data_path, 'r', encoding='utf-8'))
                linked_triple_data = json.load(codecs.open(linked_triple_path, 'r', encoding='utf-8'))

                sub1_pos = []
                sub2_pos = []
                for topic in train_data:
                    for path in train_data[topic]:
                        if path['approx_label'] == 0:
                            continue
                        path['topic_entity'] = topic
                        if path['src'] == 'sub1':
                            sub1_pos.append(path)
                        elif path['src'] == 'sub2':
                            sub2_pos.append(path)
                if type == 'composition':  ##composition questions in dev/train set have the main topic entity as the root
                    sub2_pos_rev = []
                    for sub1 in sub1_pos:
                        for sub2 in sub2_pos:
                            if sub1['topic_entity'] == sub2['topic_entity']:
                                sub2_topics = sub1['entities']
                                for sub2_topic in sub2_topics:
                                    sub2_rev = copy.deepcopy(sub2)
                                    sub2_rev['topic_entity'] = sub2_topic
                                    sub2_pos_rev.append(sub2_rev)
                    sub2_pos = sub2_pos_rev
                sub1_pos_ct = 0
                sub2_pos_ct = 0

                sub1_triples = []
                sub2_triples = []
                for triple in linked_triple_data:
                    if 'split_part1' in triple['snippet_id']:
                        sub1_triples.append(triple)
                    elif 'split_part2' in triple['snippet_id']:
                        sub2_triples.append(triple)
                for sub1_cand in sub1_pos:
                    for sub1_triple in sub1_triples:
                        if sub1_triple['subject_mid'] == sub1_cand['topic_entity'] or sub1_triple['object_mid'] == sub1_cand['topic_entity']:
                            if sub1_triple['subject_mid'] in sub1_cand['entities'] or sub1_triple['object_mid'] in sub1_cand['entities']:
                                sub1_pos_ct += 1
                                break
                                break
                for sub2_cand in sub2_pos:
                    for sub2_triple in sub2_triples:
                        if sub2_triple['subject_mid'] == sub2_cand['topic_entity'] or sub2_triple['object_mid'] == sub2_cand['topic_entity']:
                            if sub2_triple['subject_mid'] in sub2_cand['entities'] or sub2_triple['object_mid'] in sub2_cand['entities']:
                                sub2_pos_ct += 1
                                break
                                break
                if sub1_pos_ct > 0 and sub2_pos_ct > 0:
                    print('found hit')
                    triple_pos_ct += 1
                if sub1_pos_ct > 0 or sub2_pos_ct > 0:
                    print('found at least one hit')
                    at_least_one_pos_ct += 1
        print("{} of {} has full matches for triples".format(triple_pos_ct, total_ct))
        print("{} of {} has at least one matches for triples".format(at_least_one_pos_ct, total_ct))

    def get_openie_cands(self, ques_src, linked_triples_src, kb_train_data_src, relation_mapping_src, open_train_data_src):
        if not os.path.exists(open_train_data_src):
            os.makedirs(open_train_data_src)
        if not os.path.exists(relation_mapping_src):
            os.makedirs(relation_mapping_src)
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
        for q in questions:
            ques_id = q["ID"]
            type = q["compositionality_type"]
            if type == "conjunction" or type == "composition":
                kb_train_data_path = os.path.join(kb_train_data_src, ques_id + ".json")
                if not os.path.exists(kb_train_data_path):
                    continue
                linked_triple_path = os.path.join(linked_triples_src, ques_id + ".json")
                open_data_path = os.path.join(open_train_data_src, ques_id + ".json")
                relation_mapping_path = os.path.join(relation_mapping_src, ques_id + ".csv")
                if not os.path.exists(linked_triple_path):
                    continue

                if os.path.exists(open_data_path):
                    continue
                print ques_id
                train_data = json.load(codecs.open(kb_train_data_path, 'r', encoding='utf-8'))
                linked_triple_data = pd.DataFrame.from_records(json.load(codecs.open(linked_triple_path, 'r', encoding='utf-8')))

                open_cands = {}
                equivalence_records = []
                dedup_records = []

                if len(linked_triple_data) > 0:
                    topics_1 = set()
                    topics_2 = set()
                    topics_2_src_map = {}

                    for topic in train_data:
                        for path in train_data[topic]:
                            if path['src'] == 'sub1':
                                topics_1.add(topic)
                                if type == "composition":
                                    for e in path['entities']:
                                        topics_2.add(e)
                                        src_topics = topics_2_src_map.get(e, set())
                                        src_topics.add(topic)
                                        topics_2_src_map[e] = src_topics
                            elif path['src'] == 'sub2' and type == 'conjunction':
                                topics_2.add(topic)
                                src_topics = topics_2_src_map.get(topic, set())
                                src_topics.add(topic)
                                topics_2_src_map[topic] = src_topics
                    # print("topics2 src map: {}".format(topics_2_src_map))

                    sub1_cands = linked_triple_data[(linked_triple_data['subject_mid'].isin(topics_1))
                                                    | (linked_triple_data['object_mid'].isin(topics_1))].to_dict(
                        'records')

                    sub2_cands = linked_triple_data[(linked_triple_data['subject_mid'].isin(topics_2))
                                                    | (linked_triple_data['object_mid'].isin(topics_2))].to_dict(
                        'records')

                    open_cands, equivalence_records = self.get_sub1_cands(sub1_cands, train_data, equivalence_records,
                                                                          open_cands)
                    open_cands, equivalence_records = self.get_sub2_cands(sub2_cands, train_data, equivalence_records,
                                                                          open_cands, topics_2_src_map)

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

    def get_sub1_cands(self, sub1_cands, train_data, equivalence_records, open_cands):

        for cand in sub1_cands:
            cand["src"] = "sub1"
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


    def get_sub2_cands(self, sub2_cands, train_data, equivalence_records, open_cands, topics_2_src_map):
        for cand in sub2_cands:
            cand["src"] = "sub2"
            subject_mid = cand['subject_mid']
            object_mid = cand['object_mid']

            # print("topics {}".format(train_data.keys()))


            '''special handling for compositional questions'''
            topic_mid = None
            ans_mid = None
            is_reverse = False
            kb_cands = []
            if subject_mid in topics_2_src_map:
                subj_topic_cands = topics_2_src_map[subject_mid]
                # print("subjects {}".format(subj_topic_cands))
                for s in subj_topic_cands:
                    if s in train_data:
                        is_reverse = False
                        topic_mid = s
                        ans_mid = object_mid
                        kb_cands = train_data[s]
                        break
            if topic_mid is None:
                if object_mid in topics_2_src_map:
                    object_topic_cands = topics_2_src_map[object_mid]
                    # print("object {}".format(object_topic_cands))
                    for o in object_topic_cands:
                        if o in train_data:
                            is_reverse = True
                            topic_mid = o
                            ans_mid = subject_mid
                            kb_cands = train_data[o]
                            break
            if topic_mid is None:
                print("should not happen {} {}".format(subject_mid, object_mid))
                continue
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
            topic = topic_mid
            open_topic_cands = open_cands.get(topic, [])
            open_topic_cands.append(open_cand)
            open_cands[topic] = open_topic_cands
        return open_cands, equivalence_records

    def to_equivalence_record(self, cand, path, is_reverse):
        return {
            "open_relation": cand["relation"],
            "is_reverse_open_relation": is_reverse,
            "kb_relation": path["relations"],
            "is_reverse_kb_relation": path["is_reverse"]
        }

    def to_cand(self, cand, best_kb_cand, is_reverse):
        topic_prefix = "subject"
        ans_prefix = "object"
        if is_reverse:
            topic_prefix = "object"
            ans_prefix = "subject"
        reward = [0.0, 0.0, 0.0]
        derived_label = 0
        if best_kb_cand is not None and 'derived_label' in best_kb_cand:
            reward = best_kb_cand['reward']
            derived_label = best_kb_cand['derived_label']
            # if derived_label == 1:
            #     print("{} == {}".format(cand['relation'], best_kb_cand['relations']))
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
            "derived_label": derived_label,
            "reward": reward,
            "is_reverse": is_reverse,
            "src": cand['src']
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
            rel_mapping_filter_records = rel_mapping_records[~(rel_mapping_records['open_relation'].isin(filtered_rels))]

            open_data_path = os.path.join(open_train_data_filter_src, ques_id + ".json")
            relation_mapping_path = os.path.join(rel_mapping_filter_src, ques_id + ".csv")

            rel_mapping_filter_records.to_csv(relation_mapping_path, index=None, encoding='utf-8')

            with open(os.path.join(open_data_path), 'w+') as fp:
                json.dump(filter_cands, fp, indent=4)


    def is_valid_argument(self, span, mention, threshold=0.75):
        return float(len(mention)) * 1.0 / float(len(span)) > threshold


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

    def fix_format(self, relation_mapping_src, relation_mapping_tgt):
        files = os.listdir(relation_mapping_src)
        files = [f for f in files if f.endswith('.csv')]
        if not os.path.exists(relation_mapping_tgt):
            os.makedirs(relation_mapping_tgt)
        for f in files:
            df = pd.read_csv(os.path.join(relation_mapping_src, f)).to_dict('records')
            fp = open(os.path.join(relation_mapping_tgt, f), 'w+')
            fp.write(','.join(str(x) for x in ['open_relation', 'is_reverse_open_relation', 'kb_relation', 'is_reverse_kb_relation']) + "\n")
            for r in df:
                rel = r['kb_relation']
                rel = rel.replace('[', '["').replace(', ', '", "').replace(']', '"]')
                rel_list = ast.literal_eval(rel)
                if len(rel_list) == 1 or len(rel_list) == 2:
                    r['kb_relation'] = rel_list
                    fp.write("{},{},{},{}\n".format(r['open_relation'], r['is_reverse_open_relation'], r['kb_relation'], r['is_reverse_kb_relation']))
                else:
                    print(r)
            # if len(records) == 0:
            #     headers = ['open_relation', 'is_reverse_open_relation', 'kb_relation', 'is_reverse_kb_relation']
            #
            #     df_to_write = pd.DataFrame(columns=headers)
            # else:
            #     print f
            #     print(len(records))
            #     df_to_write = pd.DataFrame.from_records(records)
            # df_to_write.to_csv(os.path.join(relation_mapping_tgt, f), index=None, header=True, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
            fp.close()


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
            print f
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
                    if path['derived_label'] == 1:
                        pos += 1
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
                            r = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(r)])
                            if r in rev_cluster_map:
                                unnormalized_relations.append(r)
                                normalized_relations.append(rev_cluster_map[r])
                            else:
                                if path['derived_label'] == 1:
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

    def verify_open_cands(self, open_train_data_src):
        files = os.listdir(open_train_data_src)
        files = [f for f in files if f.endswith('.json')]
        counter = 0
        for f in files:
            # print f
            counter += 1
            if counter % 1000 == 0:
                print counter
            cands = json.load(codecs.open(os.path.join(open_train_data_src, f), 'r', encoding='utf-8'))
            for topic in cands:
                for path in cands[topic]:
                    relation = path['relations'][0]
                    if len(relation.strip()) == 0:
                        print f
                        print path
                        print topic

    def split_test_cands(self, test_src, test_dest, src_key="src1"):
        print test_dest
        if not os.path.exists(test_dest):
            os.makedirs(test_dest)
        files = os.listdir(test_src)
        files = [f for f in files if f.endswith('.json')]
        counter = 0
        for f in files:
            # print f
            counter += 1
            if counter % 1000 == 0:
                print counter
            cands = json.load(codecs.open(os.path.join(test_src, f), 'r', encoding='ascii'))
            new_cands = {}
            for topic in cands:
                paths_to_keep = []
                for path in cands[topic]:
                    if path["src"] == src_key:
                        paths_to_keep.append(path)
                if len(paths_to_keep) > 0:
                    new_cands[topic] = paths_to_keep
            with codecs.open(os.path.join(test_dest, f), 'w+', encoding='ascii') as fp:
                json.dump(new_cands, fp, indent=4)

    def fix_openie_dir(self, src):
        files = os.listdir(src)
        files = [f for f in files if f.endswith('.json')]
        counter = 0
        for f in files:
            # print f
            counter += 1
            if counter % 100 == 0:
                print counter
            cands = json.load(codecs.open(os.path.join(src, f), 'r', encoding="ascii", errors="ignore"))
            for topic in cands:
                for path in cands[topic]:
                    if path['topic_entity'] == topic:
                        path['is_reverse'] = False
                    else:
                        path['is_reverse'] = True
            with codecs.open(os.path.join(src, f), 'w+', encoding='ascii') as fp:
                json.dump(cands, fp, indent=4)

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

if __name__ == '__main__':
    DATA_PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess"
    SNIPPET_PREFIX = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_1_1'
    KE_PREFIX = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess/stanfordie/cluster_info'
    SPLIT = "test"

    SNIPPETS_SRC = os.path.join(SNIPPET_PREFIX, "Data", "web_snippets", "web_snippets_" + SPLIT + ".json")

    QUES_SRC = os.path.join(DATA_PREFIX, "annotated", SPLIT + ".json")
    LINKED_TRIPLES_SRC = os.path.join(DATA_PREFIX, "stanfordie", 'all', 'links', 'triple_cands', SPLIT)
    KB_TRAIN_DATA_SRC = os.path.join(DATA_PREFIX, "rewards", "rescaled_max_priors_derived_0.5", SPLIT)

    OPEN_TRAIN_DATA_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "cands", 'train_cands', SPLIT)
    RELATION_MAPPING_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "cands", 'relation_mapping', SPLIT)

    OPEN_TRAIN_DATA_FILTER_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "prior_cands_0.3", 'train_cands', SPLIT)
    RELATION_MAPPING_FILTER_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "prior_cands_0.3", 'relation_mapping',SPLIT)


    CLUSTER_SRC = os.path.join(KE_PREFIX, 'cluster_rel.txt')
    OPEN_TRAIN_DATA_NORMALZED_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands", 'train_cands_lemmatized', SPLIT)

    trainGen = OpenieTrainDataGen()
    # trainGen.get_snippets_coverage(QUES_SRC, SNIPPETS_SRC, TRAIN_DATA_SRC)
    # trainGen.get_relevant_triples_coverage(QUES_SRC, LINKED_TRIPLES_SRC, KB_TRAIN_DATA_SRC)

    # trainGen.get_openie_cands(ques_src=QUES_SRC, linked_triples_src=LINKED_TRIPLES_SRC, kb_train_data_src=KB_TRAIN_DATA_SRC, relation_mapping_src=RELATION_MAPPING_SRC, open_train_data_src=OPEN_TRAIN_DATA_SRC)
    trainGen.analyze_open_cands(open_train_data_src=OPEN_TRAIN_DATA_NORMALZED_SRC)
    # trainGen.filter_open_cands(open_train_data_src=OPEN_TRAIN_DATA_SRC,
    #                            open_train_data_filter_src=OPEN_TRAIN_DATA_FILTER_SRC,
    #                            rel_mapping_src=RELATION_MAPPING_SRC,
    #                            rel_mapping_filter_src=RELATION_MAPPING_FILTER_SRC,
    #                            threshold=0.3)

    # trainGen.analyze_open_cands(open_train_data_src=OPEN_TRAIN_DATA_FILTER_SRC)
    # trainGen.normalize_open_cands(OPEN_TRAIN_DATA_FILTER_SRC, CLUSTER_SRC, OPEN_TRAIN_DATA_NORMALZED_SRC)
    # OPEN_TRAIN_DATA_NORMALZED_DEST = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands",
#                                                 'train_cands_lemmatized', SPLIT+"-dedup")
    # trainGen.deduplicate(OPEN_TRAIN_DATA_NORMALZED_SRC, OPEN_TRAIN_DATA_NORMALZED_DEST)
    # OPEN_TRAIN_DATA_NORMALZED_SRC_SUBSET = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands",
    #                                                     'train_cands_lemmatized', SPLIT + "_sub1-dedup")
    #
    # # trainGen.fix_openie_dir(OPEN_TRAIN_DATA_NORMALZED_SRC)
    # trainGen.split_test_cands(OPEN_TRAIN_DATA_NORMALZED_DEST, OPEN_TRAIN_DATA_NORMALZED_SRC_SUBSET, src_key="sub1")

    # MAPPING_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands", 'relation_mapping',SPLIT)
    # MAPPING_DEST = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands", 'relation_mapping_lemmatized', SPLIT)
    # trainGen.filter_mapping(CLUSTER_SRC, MAPPING_SRC, MAPPING_DEST)

    # MAPPING_SRC = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands", 'relation_mapping_lemmatized',SPLIT)
    # MAPPING_DEST = os.path.join(DATA_PREFIX, "stanfordie", "all", "normalized_cands", 'relation_mapping_lemmatized_fixed', SPLIT)
    # trainGen.fix_format(MAPPING_SRC, MAPPING_DEST)

