import json
import codecs
import pandas as pd
import re
import csv
import nltk
from preprocessing import stringUtils
entity_pattern = re.compile(r'ns:([a-z]\.([a-zA-Z0-9_]+)) ')


def eval_topic_entity_candidates(original_dataset, dest):
    questions = json.load(codecs.open(original_dataset, 'r', encoding='utf-8'))
    names = ['ques_id', 'mention', 'begin_index', 'end_index', 'mid', 'name', 'score']
    df = pd.read_csv(dest, names=names)
    ct = 0
    total = len(questions)
    for q in questions:
        ques_id = q["ID"]
        # print ques_id
        entities = q["entities"]
        if entities is None or len(entities) == 0:
            print("warning----\t" + str(ques_id))
        cids = set(df[df['ques_id'] == ques_id]["mid"].values)
        c_names = df[df['ques_id'] == ques_id]["mention"].values
        eids = set([e["mid"] for e in entities])
        match = len(eids.intersection(cids)) > 0
        if match:
            ct += 1
        else:
            print ques_id
            print q["question"]
            print entities
            print c_names
            print cids
    print str(ct) + " of " + str(total) + " have matches"

def write_sub_topic_entity_candidates(ques_src, part_id, topic_entities_src, topic_entities_dest):
    questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
    main_topics = pd.read_csv(topic_entities_src)
    topics_frame = []
    no_links = 0
    for q in questions:
        # print(q["ID"])
        topics = main_topics[main_topics["ques_id"] == q["ID"]].to_records(index=False)
        if len(topics) == 0:
            no_links += 1
            continue
        ques_str = q[part_id]["sub_ques"]
        matches = []
        for t in topics:
            try:
                if t['mention'].lower() in ques_str.lower(): matches.append(t)
            except:
                continue
        if len(matches) == 0:
            print q["ID"] + "\t" + ques_str
            print topics
            #  this is because the dataset is very noisy in splits (even empty sub1 string)
            matches.extend(topics[0:min(len(topics), 2)])  # keep up to candidates in case of exceptions!!

        topics_frame += matches
    names = ['ques_id', 'mention', 'begin_index', 'end_index', 'mid', 'name', 'score']
    df = pd.DataFrame.from_records(topics_frame, columns=names)
    df.to_csv(topic_entities_dest, index=False, encoding='utf-8')
    print(no_links)
    print(len(questions))

    filter_entities(ques_src, topic_entities_dest)


def write_topic_entity_candidates(ques_src, raw_links_src, dest, threshold=0.4, max_cand=10):
    names = ['ques_id', 'mention', 'begin_index', 'end_index', 'mid', 'name', 'score']
    df = pd.read_csv(raw_links_src, names=names, delimiter='\t')
    df = df.dropna()
    df = df[~(df['mention'].str.contains("\?"))]
    df = df[~(df['mention'].str.contains("\%"))]
    df = df[df['mention'].str.len() >= 2]
    df = df[df.apply(lambda x: is_valid(x['mention'], x['name']), axis=1)]
    df['score'] = df['score'].astype(float)
    df = df[df['score'] >= threshold]

    # check if this valid! aqqu misses out some exact matches of mentions..include them heuristically
    print df.shape

    matched_set = []
    questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
    for q in questions:
        # print q["ID"]
        ques_str = q["question"]
        entities = q["entities"]
        matched_entities = find_mentions(q["ID"], ques_str, entities)
        matched_set += matched_entities

    df2 = pd.DataFrame.from_records(matched_set, columns=names)
    df = df.append(df2)
    print df2.shape
    df = df.sort_values(['ques_id', 'score'], ascending=[True, False])
    df = df.drop_duplicates(subset=['ques_id', 'mid'])

    print df.shape
    df = df.groupby('ques_id').head(max_cand).reset_index(drop=True)
    df.to_csv(dest, index=False, encoding='utf-8')

    filter_entities(ques_src, dest)


def filter_entities(ques_src, topic_path):
    questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
    entities = set()
    for q in questions:
        sparql_str = q["sparql"]
        try:
            matches = re.findall(entity_pattern, str(sparql_str))
        except:
            matches = []
        for m in matches:
            entities.add(m[0])

    df = pd.read_csv(topic_path)
    print(df.shape)

    df = df[df.mid.str[1] == '.']


    top_mids = df.groupby('mid').size().reset_index(name='counts')
    top_mids = top_mids.sort_values('counts', ascending=False)
    entities_to_filter = top_mids[top_mids['counts'] > 10]['mid'].tolist()
    entities_to_filter = [e for e in entities_to_filter if e not in entities]
    for e in entities_to_filter:
         df = df[df['mid'] != e]
    print(df.shape)

    df.to_csv(topic_path, index=False)


def find_mentions(ques_id, text, entities_list):
    matched_entities = []
    for e in entities_list:
        names = e["mention"]
        if names is None:
            continue
        mention, score = stringUtils.match_entities(text, [names])
        if mention is None:
            continue
        matched_entity = {"ques_id": ques_id, "mention": mention[0], "begin_index": mention[1], "end_index": mention[2], \
                          "mid": e["mid"], "name": e["mention"], "score": 1.0}
        matched_entities.append(matched_entity)
    return matched_entities


def write_entities_from_sparql(self, src, dest):
    questions = json.load(codecs.open(src, 'r', encoding='utf-8'))
    for q in questions:
        print(q["ID"])
        sparql_str = q["sparql"]
        true_entities = {}
        try:
            matches = re.findall(entity_pattern, str(sparql_str))
        except:
            matches = []
        for m in matches:
            entity = m[0]
            true_entities[entity] = self.sparql.get_names(entity)
        true_entities_list = [{"mid": key, "mention": value} for key, value in true_entities.items()]
        q["entities"] = true_entities_list
        if 'entities' in q['split_part1']: del q['split_part1']['entities']
        if 'entities' in q['split_part2']: del q['split_part2']['entities']
    json.dump(questions, codecs.open(dest, 'w+', encoding='utf-8'), indent=4)


def is_valid(mention, name, threshold=0.1):
    try:
        s1 = set(nltk.word_tokenize(str(mention).lower()))
        s2 = set(nltk.word_tokenize(str(name).lower()))
    except:
        return True
    inter = s1.intersection(s2)
    un = len(s1) + len(s2) - len(inter)
    score = float(len(inter)) / max(1.0, un)
    return score > threshold