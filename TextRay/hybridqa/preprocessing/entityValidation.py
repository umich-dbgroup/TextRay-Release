from kbEndPoint.utils.sparql import sparqlUtils
import os
import json
import pandas as pd
from itertools import chain
import argparse


INPUT_PREFIX = "../../datasets/ComplexWebQuestions_1_1/Data/complex_web_questions/"
OUTPUT_PREFIX= "../../datasets/ComplexWebQuestions_1_1/Data/qid_mid_names/"
ERROR_FILE = "../../datasets/ComplexWebQuestions_1_1/Data/qid_mid_names/error_questions.txt"
MISSING_ENTITY_FILE = "../../datasets/ComplexWebQuestions_1_1/Data/qid_mid_names/missing_question_entity.txt"
CHOICES = ['test', 'train', 'dev']
SPARQL = sparqlUtils()
ENCODING = 'utf-8'

def check_question(raw_question):
    ID = raw_question[u"ID"].encode(ENCODING)
    entities_mids = SPARQL.get_entity_mids(raw_question[u"sparql"])
    if len(entities_mids) == 0:
        return True
    for mid in entities_mids:
        if (SPARQL.exists(mid)):
            return True
    return False

def entity_lookup(raw_question):
    ques_entities = []
    entities_mids = SPARQL.get_entity_mids(raw_question.sparql)
    for mid in entities_mids:
        names = SPARQL.get_names_alias(mid)
        ques_entities.append({"ID": raw_question.ID, "mid": mid, "names": names})
    if len(ques_entities) == 0:
        ques_entities.append({"ID": raw_question.ID})
    return ques_entities

def entity_lookups(input_dir, output_dir):
    for choice in CHOICES:
        print "start " + choice
        input_path = os.path.join(input_dir, "ComplexWebQuestions_" + choice + ".json")
        out_path = os.path.join(output_dir, choice + "_mid_names.tsv")
        df = pd.read_json(input_path)
        df['ques_entities'] = df.apply(lambda x: entity_lookup(x), axis=1)
        ques_entities_df = pd.DataFrame(list(chain.from_iterable(df['ques_entities'].tolist())))
        ques_entities_df.to_csv(out_path, index=False)

def process_question(raw_question, choice):
    '''
    :param raw_question objects from json file
    :return: list of writen strings
    '''
    written_strings = []
    ID = raw_question[u"ID"].encode(ENCODING)
    entities_mids = SPARQL.get_entity_mids(raw_question[u"sparql"])
    if len(entities_mids) == 0:
        error_file.write(choice + '\t' + ID + '\n')
        return []
    for mid in entities_mids:
        names = SPARQL.get_names_alias(mid)
        if len(names) == 0:
            missing_entity_file.write('\t'.join([choice, ID, mid]) + '\n')
            continue
        written_strings.append('\t'.join([ID, mid, ','.join(names)]))
    return written_strings


def qid_mid_names():
    global error_file
    error_file = open(ERROR_FILE, 'w+')
    global missing_entity_file
    missing_entity_file = open(MISSING_ENTITY_FILE, 'w+')
    for choice in CHOICES:
        question_file = open(os.path.join(INPUT_PREFIX, "ComplexWebQuestions_" + choice + ".json"))
        raw_questions = json.load(question_file)
        with open(os.path.join(OUTPUT_PREFIX, choice + "_mid_names.tsv"), 'w+') as written_file:
            for raw_question in raw_questions:
                written_strings = process_question(raw_question, choice=choice)
                for written_string in written_strings:
                    written_file.write((written_string + '\n').encode('utf-8'))
        question_file.close()


def check_topic_entities():
    for choice in CHOICES:
        print "start " + choice
        question_file = open(os.path.join(INPUT_PREFIX, "ComplexWebQuestions_" + choice + ".json"))
        raw_questions = json.load(question_file)
        with open(os.path.join(OUTPUT_PREFIX, choice + "_no_one_entity_exist.tsv"), 'w+') as written_file:
            for raw_question in raw_questions:
                if not check_question(raw_question):
                    written_file.write('\t'.join([choice, raw_question[u"ID"]]) + '\n')

        question_file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='finding entities')
    parser.add_argument('-I', '--input',
                        type=str,
                        help='Input',
                        default=INPUT_PREFIX)
    parser.add_argument('-O', '--output',
                        type=str,
                        help='Output',
                        default=OUTPUT_PREFIX)
    args = parser.parse_args()
    entity_lookups(input_dir=args.input, output_dir=args.output)
    check_topic_entities()















