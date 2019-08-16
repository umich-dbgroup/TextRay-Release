import json
import os
import time
from kbEndPoint.utils.sparql import sparqlUtils
from dataModels.kbModels import Relation,Entity, Node, Value
import pandas as pd
import re

from kbEndPoint.kbInterface import EndPoint
import sys

INPUT_PATH = "/Users/funke/WebQSP/data/WebQSP.test.json"
SPARQL = sparqlUtils()
RELATIONS_FILTER = "/Users/funke/WebQSP/all_relations.txt"
sparql = sparqlUtils()

def read_file(path):
    file = open(path, 'r')
    file_json = json.load(file)
    raw_questions = file_json["Questions"]
    return raw_questions

if __name__ == '__main__':
    questions = read_file(INPUT_PATH)
    relations_to_filter = set(pd.read_csv(RELATIONS_FILTER, names=['rel']).rel)
    success = 0
    total_parses = 0
    one_step_len = []
    two_step_len = []
    for question in questions:
        for parse in question["Parses"]:
            inferential_chain = parse["InferentialChain"]
            if inferential_chain is not None:
                total_parses += 1
                topic_entity = parse["TopicEntityMid"]
                print(topic_entity)
                one_step = sparql.one_step(topic_entity, relations_to_filter)
                print(len(one_step))
                one_step_len.append(len(one_step))
                two_step = sparql.two_steps(topic_entity, relations_to_filter)
                print(len(two_step))
                two_step_len.append(len(two_step))
                if len(inferential_chain) == 1:
                    if inferential_chain[0] in one_step:
                        print("found")
                        success += 1
                        continue
                elif len(inferential_chain) == 2:
                    if (inferential_chain[0], inferential_chain[1]) in two_step or (inferential_chain[1], inferential_chain[0]) in two_step:
                        print("found")
                        success += 1
                        continue

    print "total parses: ", total_parses
    print "success: ", success
    print "avg. one step", sum(one_step_len) / float(len(one_step_len))
    print "avg. two step", sum(two_step_len) / float(len(two_step_len))
