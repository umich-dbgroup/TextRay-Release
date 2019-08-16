import json
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
import re
import os

nlpserver = StanfordCoreNLP('http://localhost', port=9000)
nlp_props = {'annotators': 'openie','tokenize.options': 'untokenizable=noneKeep', 'outputFormat': 'text', "openie.format": "reverb"}

DATA_PREFIX="/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP"
OUTPUT_PREIFX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final/stanfordie"

sentence_pattern = re.compile('Sentence #(.*?) \(')

def get_all_tuples(snippet_src, dest_dir):
    counter = 0
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    with open(snippet_src) as snippet_fp:
        for line in snippet_fp:
            row = json.loads(line.rstrip())
            ques_id = str(row['documentId'])
            counter += 1
            if counter % 100 == 0:
                print("{} questions processed".format(counter))
            if os.path.exists(os.path.join(dest_dir, ques_id + ".json")):
                continue
            all_triples = get_openie_tuples_for_row(ques_id, row)
            with open(os.path.join(dest_dir, ques_id + ".json"), 'w+') as f:
                json.dump(all_triples, f, encoding='utf-8', indent=4)

def get_openie_tuples_for_row(src_id, row, parse_pop=nlp_props):
    snippet = row['document']['text']
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

        sentence_triples = [get_triple(src_id, sentence_id, triple_row.rstrip()) for triple_row in triples]
        triple_records += sentence_triples
    return triple_records

def get_triple(src_id, sentence_id, triple_row):
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
        "sentence_id": sentence_id
        }
    return triple_record

if __name__ == '__main__':
    snippet_src = os.path.join(DATA_PREFIX, "documents.json")
    dest_dir = os.path.join(OUTPUT_PREIFX, "all_triples")
    get_all_tuples(snippet_src, dest_dir)