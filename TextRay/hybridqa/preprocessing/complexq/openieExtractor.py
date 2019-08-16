
import json
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
import pandas as pd
import re
import os
import itertools

nlpserver = StanfordCoreNLP('http://localhost', port=9000)
nlp_props = {'annotators': 'openie','tokenize.options': 'untokenizable=noneKeep', 'outputFormat': 'text', "openie.format": "reverb"}

DATA_PREFIX="/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_1_1/Data/web_snippets"
#OUTPUT_PREIFX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_1_1/stanfordie"
OUTPUT_PREIFX = "/home/nbhutani/Documents/stanfordie"

SPLIT="train"

sentence_pattern = re.compile('Sentence #(.*?) \(')

def get_all_tuples(snippet_src, dest_dir):
    snippets_df = pd.read_json(snippet_src, orient='records', encoding='utf-8')
    '''split source must be ptrnet and not noisy supervision'''
    snippets_df = snippets_df[snippets_df['split_source'].str.join(' ').str.contains('ptrnet split')]
    '''split type is full_question or split_part1 or split_part2'''
    counter = 0
    for ques_id, group in snippets_df.groupby("question_ID"):
        counter += 1
        print ques_id
        if counter % 100 == 0:
            print("{} questions processed".format(counter))
        if os.path.exists(os.path.join(dest_dir, ques_id + ".json")):
            continue
        get_openie_tuple_from_grp(ques_id, group, dest_dir)

def get_openie_tuple_from_grp(ques_id, group, dest_dir):
    df = group.reset_index()
    triples = df.apply(lambda x: get_openie_tuple_from_row(ques_id, x), axis=1)
    all_triples = list(itertools.chain.from_iterable(triples.values))
    with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as f:
        json.dump(all_triples, f, encoding='utf-8',indent=4)


def get_openie_tuple_from_row(ques_id, row):
    snippet_id_prefix = row['split_type']
    snippet_df = pd.DataFrame.from_dict(row['web_snippets'], orient='columns')
    snippet_df['snippet_id'] = snippet_df.index
    triples = snippet_df.apply(lambda x: get_openie_tuples_for_snippet_row(ques_id, snippet_id_prefix, x),axis=1).values.tolist()
    all_triples = list(itertools.chain.from_iterable(triples))
    return all_triples

def get_openie_tuples_for_snippet_row(src_id, snippet_prefix, row, parse_pop=nlp_props):
    snippet = row.snippet
    snippet_id = snippet_prefix + "_" + str(row.snippet_id)
    triple_records = []
    for sentence_id, sentence in enumerate(sent_tokenize(snippet)):
        sentence = sentence.encode('utf-8')
        if len(sentence) < 10: # to avoid sentences that are too short. (typical in web snippets)
            continue
        openie_output = nlpserver.annotate(sentence, parse_pop)
        chunks = openie_output.split("\nSentence #")
        if len(chunks) < 2:
            continue
        triple_chunks = chunks[1].split('Extracted the following Open IE triples:\n')
        if len(triple_chunks) < 2:
            continue
        triples = triple_chunks[1].rstrip().split("\n")

        sentence_triples = [get_triple(src_id, snippet_id, sentence_id, triple_row.rstrip()) for triple_row in triples]
        triple_records += sentence_triples
    return triple_records

def get_triple(src_id, snippet_id, sentence_id, triple_row):

    triple = triple_row.split("\t")
    subject_span = triple[2]
    relation_span = triple[3]
    object_span = triple[4]

    subject_span_begin = int(triple[5])
    subject_span_end = int(triple[6])

    relation_span_begin = int(triple[7])
    relation_span_end = int(triple[8])

    object_span_begin = int(triple[9])
    object_span_end = int(triple[10])

    confidence = float(triple[11])

    span = subject_span + " " + relation_span + " " + object_span

    triple_record = {
        "subject": subject_span,
        "relation": relation_span,
        "object": object_span,
        "subject_begin_index": subject_span_begin,
        "subject_end_index": subject_span_end,
        "relation_begin_index": relation_span_begin,
        "relation_end_index": relation_span_end,
        "object_begin_index": object_span_begin,
        "object_end_index": object_span_end,
        "confidence": confidence,
        "text_span": span,
        "ques_id": src_id,
        "snippet_id": snippet_id,
        "sentence_id": sentence_id
        }
    return triple_record

def test():
    sentence = 'Our restaurant opens at 10am and closes at 12pm.'

    class Row(object):
        pass
    row = Row()
    row.snippet = sentence
    row.snippet_id = 'snippet_1'
    triples = get_openie_tuples_for_snippet_row(0, 'split', row)
    for t in triples:
        print t


if __name__ == '__main__':
    snippet_src = os.path.join(DATA_PREFIX, "web_snippets_" + SPLIT + ".json")
    dest_dir = os.path.join(OUTPUT_PREIFX, SPLIT)
    #print snippet_src
    get_all_tuples(snippet_src, dest_dir)
    # test()