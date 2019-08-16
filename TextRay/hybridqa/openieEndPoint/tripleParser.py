import pandas as pd
import argparse
import os
import io
import re

"""
This reads triples in openIE5 format (sentenceids, sentences, triples) and writes to a pandas friendly format
"""


context_pat = re.compile(r"^(\d*\.?\d*)\sContext\((.*?),.*?:(.*?)$")
pat = re.compile(r"^(\d*\.?\d*)\s(.*?)$")
arg_pat = re.compile(r"^([LT]:)?(.*?)$")

def get_triples(triples_file):
    group = []
    for line in triples_file:
        if line == "\n":
            if group:
                yield group
                group = []
        else:
            group.append(line.rstrip('\n'))
    if group:
        yield group

def parse_triple(triple_string):
    context_m = context_pat.search(triple_string)
    conf = None
    context = None
    triple_content = None
    if context_m:
        conf = context_m.group(1).strip()
        context = context_m.group(2).strip()
        triple_content = context_m.group(3).strip()
    else:
        m = pat.search(triple_string)
        if m:
            conf = m.group(1).strip()
            triple_content = m.group(2).strip()
    if len(triple_content) > 0:
        triple_content = triple_content[1:-1]
        triple_args = triple_content.split(";")
        if len(triple_args) < 2:
            return None
        subj = arg_pat.search(triple_args[0].strip()).group(2)
        rel = arg_pat.search(triple_args[1].strip()).group(2)
        if subj is None or rel is None:
            return None
        args = []
        for x in range(2, len(triple_args)):
            arg = arg_pat.search(triple_args[x].strip()).group(2)
            if arg is None:
                continue
            args.append(arg)
        if context is None:
            return {"confidence": conf,
                "subject": subj,
                "relation": rel,
                "arguments": args}
        else:
            return {"confidence": conf,
                "context": context,
                "subject": subj,
                "relation": rel,
                "arguments": args}

    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reader script')
    parser.add_argument('-I', '--ids',
                        type=str)
    parser.add_argument('-S', '--sentences',
                        type=str)
    parser.add_argument('-T', '--triples',
                        type=str)
    parser.add_argument('-O', '--output',
                        type=str)
    args = parser.parse_args()

    ids = [line.rstrip('\n') for line in open(args.ids)]
    sentences = [line.rstrip('\n') for line in open(args.sentences)]

    assert len(ids) == len(sentences)

    groups = []
    with open(args.triples, 'r') as triples_file:
        groups = list(get_triples(triples_file))



    all_triples = []
    counter = 0
    for id, sentence in zip(ids,sentences):
        group = groups[counter]
        triples = []
        if group[0] == sentence: # sentence must have thrown an exception
            counter+= 1
            if len(group) > 1:
                triple_strings = group[1:]
                for triple_string in triple_strings:
                    triple = parse_triple(triple_string)
                    if triple is None:
                        continue
                    triples.append(triple)
        all_triples.append([id, sentence, triples])



    df = pd.DataFrame(all_triples, columns=['id', 'sentence', 'triples'])
    df.to_csv(args.output, index=None)


