import json
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
import pandas as pd
import argparse


nlpserver = StanfordCoreNLP('http://localhost', port=9000)
nlp_props = {'annotators': 'tokenize,ssplit,pos,lemma,depparse,natlog,openie', 'pipelineLanguage': 'en', 'tokenize.options': 'untokenizable=noneKeep'}

def parse(text):
    text = text.encode('ascii', 'ignore').decode(encoding='UTF-8')
    sentences = sent_tokenize(text)  # relying on sentence tokenization or else the stanford parser will throw errors for long text
    annotated = []
    for i in range(0, len(sentences)):
        u = json.loads(nlpserver.annotate(sentences[i], nlp_props))
        annotated.append(u['sentences'][0])
    return annotated

def get_openie_tuples(text):
    tuples = []
    annotated = parse(text)
    for sentence in annotated:
        for tp in sentence['openie']:
            open_tuple = {'subject': tp['subject'], 'relation': tp['relation'], 'object': tp['object']}
            tuples.append(open_tuple)
    return tuples

def get_tuples_from_json(infile, outfile):
    df = pd.read_json(infile, encoding="utf-8")
    tuples = open(outfile, "w", encoding="utf-8")
    for index, row in df.iterrows():
        for snippet in row.web_snippets:
            snippet_text = snippet["snippet"]
            tuples.write(snippet_text + "\n")
            snippet_tuples = get_openie_tuples(snippet_text)
            for t in snippet_tuples:
                tuples.write("(" + t["subject"] + "; " + t["relation"] + "; " + t["object"] + ")\n")
            tuples.write("\n")
    tuples.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Triple Coverage script')
    parser.add_argument('-I', '--input',
                        type=str,
                        help='Input file')
    parser.add_argument('-O', '--output',
                        type=str,
                        help='Output triples file')
    args = parser.parse_args()
    get_tuples_from_json(args.input, args.output)
