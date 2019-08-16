import argparse
import json
import re


def get_rel_filter():
    relations = set(ComplexWebQuestionsRelations()).union(set(WebQSPRelations()))
    filters = ["^http://rdf.freebase.com/ns/" + r for r in relations]
    filter = '|'.join(filters)
    return filter

def ComplexWebQuestionsRelations():
    return ['visual_art',
            'meteorology',
            'topic_server',
            'people',
            'influence',
            'symbols',
            'media_common',
            'education',
            'event',
            'film',
            'geography',
            'astronomy',
            'dining',
            'tv',
            'travel',
            'tennis',
            'periodicals',
            'sports',
            'religion',
            'book',
            'music',
            'location',
            'automotive',
            'medicine',
            'olympics',
            'military',
            'common',
            'ice_hockey',
            'biology',
            'finance',
            'business',
            'government',
            'food',
            'celebrities',
            'internet',
            'award',
            'broadcast',
            'boats',
            'base',
            'baseball',
            'user',
            'measurement_unit',
            'aviation',
            'law',
            'kg',
            'basketball',
            'soccer',
            'theater',
            'language',
            'distilled_spirits',
            'fictional_universe',
            'american_football',
            'royalty',
            'architecture',
            'time',
            'organization',
            'chemistry',
            'interests',
            'protected_sites',
            'exhibitions',
            'conferences',
            'engineering',
            'zoos',
            'imdb',
            'transportation',
            'amusement_parks',
            'wine']

def WebQSPRelations():
    return ['visual_art',
            'meteorology',
            'topic_server',
            'people',
            'influence',
            'symbols',
            'computer',
            'media_common',
            'education',
            'event',
            'film',
            'geography',
            'astronomy',
            'dining',
            'tv',
            'travel',
            'tennis',
            'periodicals',
            'sports',
            'religion',
            'book',
            'music',
            'location',
            'automotive',
            'medicine',
            'olympics',
            'military',
            'common',
            'ice_hockey',
            'biology',
            'finance',
            'business',
            'government',
            'food',
            'celebrities',
            'internet',
            'award',
            'broadcast',
            'boats',
            'base',
            'baseball',
            'user',
            'measurement_unit',
            'aviation',
            'law',
            'kg',
            'basketball',
            'cvg',
            'soccer',
            'theater',
            'language',
            'distilled_spirits',
            'fictional_universe',
            'american_football',
            'royalty',
            'architecture',
            'time',
            'organization',
            'chemistry']

def get_namespaces(sparqls):
    namespaces = []
    pattern = re.compile(r'ns:(.*?)\s')
    for sparql in sparqls:
        try:
            matches = re.findall(pattern, str(sparql))
            for m in matches:
                if m.startswith("m."):
                    continue
                if len(m) == 0:
                    continue
                namespaces.append(m)
        except:
            continue
    topics = [n.split(".")[0] for n in namespaces]
    return set(topics)

def get_complex_relations(input_path):
    file = open(input_path, 'r')
    questions = json.load(file)
    relations = set()
    pattern = re.compile(r'(ns:.*?|\?\w)\sns:(.*?)\s(ns:.*?|\?\w)\s\.')
    for question in questions:
        sparql = question["sparql"]
        try:
            matches = re.findall(pattern, str(sparql))
            for m in matches:
                relations.add(m[1])
        except:
            continue
    return relations


def get_WebQSP_relations(input_path):
    file = open(input_path, 'r')
    file_json = json.load(file)
    questions = file_json["Questions"]
    relations = set()
    pattern = re.compile(r'(ns:.*?|\?\w)\sns:(.*?)\s(ns:.*?|\?\w)\s\.')
    for question in questions:
        for parse in question["Parses"]:
            if parse["InferentialChain"] is not None:
                for rel in parse["InferentialChain"]:
                    relations.add(rel)
            sparql = parse["Sparql"]
            try:
                matches = re.findall(pattern, str(sparql))
                for m in matches:
                    relations.add(m[1])
            except:
                continue
    return relations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='joining dataset')
    parser.add_argument('-I', '--input',
                        type=str,
                        help='Input')
    parser.add_argument('-O', '--output',
                        type=str,
                        help='Output')
    args = parser.parse_args()
    relations = get_complex_relations(input_path=args.input)
    #relations = get_WebQSP_relations(input_path=args.input)
    with open(args.output, 'w+') as f:
        for r in relations:
            f.write(r + "\n")

    # with open("/Users/funke/Data/subgraphs/test_relations.txt", "r") as f:
    #     test_relations = f.readlines()
    #     test_relations = [t.strip() for t in test_relations]
    #
    # with open("/Users/funke/Data/subgraphs/train_relations.txt", "r") as f:
    #     train_relations = f.readlines()
    #     train_relations = [t.strip() for t in train_relations]
    #
    # with open("/Users/funke/Data/subgraphs/dev_relations.txt", "r") as f:
    #     dev_relations = f.readlines()
    #     dev_relations = [t.strip() for t in dev_relations]
    #
    # s = set(test_relations)
    # s = s.union(train_relations)
    # s = s.union(dev_relations)
    #
    # with open("/Users/funke/Data/subgraphs/all_relations.txt", 'w+') as f:
    #     for r in s:
    #         f.write(r + "\n")

    # with open("/Users/funke/Data/subgraphs/all_relations.txt", 'r') as f:
    #     r1 = f.readlines()
    #     r1 = [t.strip() for t in r1]
    #
    # with open("/Users/funke/WebQSP/all_relations.txt", 'r') as f:
    #     r2 = f.readlines()
    #     r2 = [t.strip() for t in r2]
    #
    # s = set(r1)
    # s = s.union(r2)
    #
    # with open("/Users/funke/Data/subgraphs/union_relations.txt", 'w+') as f:
    #     for r in s:
    #         f.write(r + "\n")
