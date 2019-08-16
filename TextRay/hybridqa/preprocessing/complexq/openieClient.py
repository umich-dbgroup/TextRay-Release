#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import pandas as pd
import codecs
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from string import Template
from nltk.corpus import stopwords
from nltk import word_tokenize


class ElasticSearchClient(object):

    def __init__(self):
        self.host = "localhost"
        self.port = 9200
        self.es = Elasticsearch([{'host': self.host, 'port': self.port}])
        self.stopwords = set(stopwords.words('english'))

        self.query_temp = Template('''
                                {
                                  "query": {
                                    "bool": {
                                      "must": {
                                        "match": {"text_span": "${text_span}"}
                                      },
                                      "filter": {
                                        "term": {
                                          "ques_id": "${ques_id}"
                                        }
                                      }
                                    }
                                  },
                                  "size": ${size}
                                }
                                        ''')

    def index_triples(self, src_dir, split='dev'):
        src_dir = os.path.join(src_dir, split)
        index_name = split + "triples"
        doc_type = 'triples'

        bulk(self.es,
                self.gen_triples(src_dir),
                index=index_name,
                doc_type=doc_type,
                chunk_size=1000  # keep the batch sizes small for appearances only)
             )

    def gen_triples(self, src_dir):
        files = os.listdir(src_dir)
        files = [f for f in files if f.endswith(".json")]
        for f in files:
            triples_list = json.load(codecs.open(os.path.join(src_dir, f), 'r', encoding='utf-8'))
            counts = {}
            for triple in triples_list:
                src_id = "{}:{}:{}".format(triple['ques_id'], triple['snippet_id'], triple['sentence_id'])
                triple_ct = counts.get(src_id, 0)
                triple_id = src_id + ":" + str(triple_ct)
                triple_ct += 1
                counts[src_id] = triple_ct
                triple['triple_id'] = triple_id
                yield triple

    # def gen_triples(self, src_dir):
    #     files = os.listdir(src_dir)
    #     files = [f for f in files if f.endswith(".json")]
    #     for f in files:
    #         triples_list = json.load(codecs.open(os.path.join(src_dir, f), 'r', encoding='utf-8'))
    #         for triple in triples_list:
    #             triple['triple_id'] = "{}:{}:{}".format(triple['ques_id'], triple['snippet_id'], triple['sentence_id'])
    #             yield triple

    def count_triples(self, split='dev'):
        index_name = split + "triples"
        doc_type = 'triples'
        results = self.es.search(index=index_name, doc_type=doc_type, body={
            'query': {
                "match_all": {}
            }
        })
        print('Total %d triples found in %dms' % (results['hits']['total'], results['took']))

    def get_relevant_triples(self, data_dir, dest_dir, split='dev'):
        questions = json.load(codecs.open(os.path.join(data_dir, split + ".json"), 'r', encoding='utf-8'))
        dest_dir = os.path.join(dest_dir, split)

        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        for q in questions:
            ques_id = q['ID']
            print ques_id
            split_part1 = self.query_string(q['split_part1']['sub_ques'])
            split_part2 = self.query_string(q['split_part2']['sub_ques'].replace('%composition', ''))
            ques = self.query_string(q['question'])

            # find best 100 triples
            triples1 = self.query(ques_id, split_part1, 200, split)
            triples_split1 = [t for t in triples1 if t['snippet_id'].startswith('split_part1')]
            triples_split1 = triples_split1[:min(100, len(triples_split1))]
            for t in triples_split1:
                t['split_type'] = 'split_part1'

            # for t in triples_split1:
            #     print "{} \t {} \t {}".format(t['subject'], t['relation'], t['object'])

            # find best 100 triples
            if len(triples1) == 0: # this is for backup, for questions that didn't have a split
                triples1 = self.query(ques_id, ques, 200, split)
                triples_split1 = [t for t in triples1 if t['snippet_id'].startswith('full_question')]
                triples_split1 = triples_split1[:min(100, len(triples_split1))]
                for t in triples_split1:
                    t['split_type'] = 'full_question'

            triples2 = self.query(ques_id, split_part2, 200, split)
            triples_split2 = [t for t in triples2 if t['snippet_id'].startswith('split_part2')]
            triples_split2 = triples_split2[:min(100, len(triples_split2))]
            for t in triples_split2:
                t['split_type'] = 'split_part2'

            # for t in triples_split2:
            #     print "{} \t {} \t {}".format(t['subject'], t['relation'], t['object'])

            triple_ids = set()
            triples = []
            for t in triples_split1:
                if t['triple_id'] in triple_ids:
                    continue
                triple_ids.add(t['triple_id'])
                triples.append(t)
            for t in triples_split2:
                if t['triple_id'] in triple_ids:
                    continue
                triple_ids.add(t['triple_id'])
                triples.append(t)

            with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as f:
                json.dump(triples, f, encoding='utf-8', indent=4)


    def query_string(self, query):
        query_tokens = word_tokenize(query)
        query_tokens = [q for q in query_tokens if q not in self.stopwords]
        return ' '.join(query_tokens)


    def query(self, ques_id, text_span, size, split='dev'):
        index_name = split + "triples"
        doc_type = 'triples'
        query = self.query_temp.substitute(text_span=text_span, ques_id=ques_id, size=size)
        results = self.es.search(index=index_name, doc_type=doc_type, body=query)
        triples = []
        for hit in results['hits']['hits']:
            score = hit['_score']
            triple = hit['_source']
            triple['relevance_score'] = score
            triples.append(triple)
        return triples


    def sanity_check(self, data_dir, split='dev'):
        questions = json.load(codecs.open(os.path.join(data_dir, split + ".json"), 'r', encoding='utf-8'))
        invalid_ct = 0
        for q in questions:
            if not "split_part1" in q or not "split_part2" in q:
                invalid_ct += 1
        print invalid_ct

    def snippet_sanity_check(self, snippet_dir, split):
        snippet_src = os.path.join(snippet_dir, "web_snippets_" + split + ".json")
        snippets_df = pd.read_json(snippet_src, orient='records', encoding='utf-8')
        snippets_df = snippets_df[snippets_df['split_source'].str.join(' ').str.contains('ptrnet split')]
        valid_ct = 0
        total_ct = 0
        for ques_id, group in snippets_df.groupby(["question_ID"]):
            total_ct += 1
            has_split_1 = False
            has_split_2 = False
            # print ques_id
            for split_group_name, split_group in group.groupby('split_type'):
                # print split_group_name
                if split_group_name == 'split_part1': has_split_1 = True
                if split_group_name == 'split_part2': has_split_2 = True
            if has_split_1 and has_split_2: valid_ct += 1
        print total_ct - valid_ct

if __name__ == '__main__':
    esclient = ElasticSearchClient()
    split = 'test'

    data_prefix ='/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_1_1'
    preprocess_prefix = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess'
    ques_data_dir = os.path.join(preprocess_prefix, 'annotated')
    triples_dir = os.path.join(data_prefix, 'stanfordie')
    relevant_triples_dir = os.path.join(preprocess_prefix, 'stanfordie', 'relevant')

    # esclient.index_triples(triples_dir, split)
    # esclient.count_triples(split)


    # esclient.sanity_check(ques_data_dir, split)
    # esclient.snippet_sanity_check(os.path.join(data_prefix, 'Data', 'web_snippets'), split)

    esclient.get_relevant_triples(ques_data_dir, relevant_triples_dir, split=split)

