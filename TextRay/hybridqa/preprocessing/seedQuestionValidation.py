import logging
import sys
logging.basicConfig(filename='./test.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import json
import os
sys.path.append('/home/xinyi/TextRay_Root/TextRay/hybridqa')
import time
from kbEndPoint.utils.sparql import sparqlUtils
import numpy as np
import networkx as nx
from dataModels.kbModels import Relation,Entity, Node, Value

from kbEndPoint.kbInterface import EndPoint
import sys
import pickle

INPUT_PREFIX = "../../datasets/WebQSP/data/"
SPARQL = sparqlUtils()


def read_file(path):
    file = open(path, 'r')
    file_json = json.load(file)
    raw_questions = file_json[u"Questions"]
    return raw_questions


def question_entity_exist(raw_question):
    parses = raw_question[u"Parses"]
    for parse in parses:
        if parse[u"TopicEntityMid"]:
            if SPARQL.exists(parse[u"TopicEntityMid"]):
                return True
    return False

def answer_entity_exist(raw_question):
    parses = raw_question[u"Parses"]
    for parse in parses:
        answers = parse[u"Answers"]
        for answer in answers:
            if answer[u"AnswerType"] == u"Entity":
                if answer[u"AnswerArgument"]:
                    if SPARQL.exists(answer[u"AnswerArgument"]):
                        return True
            else:
                #print(answer[u"AnswerType"])
                return True
    return False

def get_entity_mention(parse):
    mention_str = parse["PotentialTopicEntityMention"]
    tokens = mention_str.strip().lower().split()
    return set(tokens)

def process_question_entities(raw_questions):
    topic_entity_wrong = 0
    answer_entity_wrong = 0
    for raw_question in raw_questions:
        if not question_entity_exist(raw_question):
            topic_entity_wrong += 1
        if not answer_entity_exist(raw_question):
            answer_entity_wrong += 1

    print("There are {} questions in total".format(len(raw_questions)))
    print("Topic entity doesn't exist in freebase: {}".format(topic_entity_wrong))
    print("Answer entity doesn't exist in freebase: {}".format(answer_entity_wrong))


def sparql_executable(parses):
    for parse in parses:
        query = parse[u"Sparql"]
        idx = query.find("PREFIX")
        try:
            if len(SPARQL.execute(query[idx:])[u"results"][u"bindings"]) > 0:
                return True
        except Exception as e:
            return True
    return False



def check_shortestPath(raw_questions):
    SPARQL = sparqlUtils()
    max_len = 0
    lens = []
    no_path = 0
    no_path_query = open(os.path.join(INPUT_PREFIX, 'no_path_query.txt'), 'w+')
    for raw_question in raw_questions:
        curr_len = -1
        parses = raw_question[u"Parses"]
        for parse in parses:
            topic_entity = parse[u"TopicEntityMid"]
            answer_entities = []
            answers = parse[u"Answers"]
            for answer in answers:
                answer_entities.append(answer[u"AnswerArgument"])
            for idx in range(len(answer_entities)):
                answer_entity = answer_entities[idx]
                if answer_entity[0:2] != 'm.':
                    answer_entity = '''"''' + answer_entity + '''^^^xsd:dateTime'''
                try:
                    this_len = SPARQL.shortestPathLength(topic_entity, answer_entity)
                    curr_len = max(this_len, curr_len)
                except Exception:
                    continue

        max_len = max(max_len, curr_len)
        if curr_len == -1:
            no_path_query.write(raw_question[u"QuestionId"] + '\n')
            no_path += 1
        else:
            lens.append(curr_len)
    print "There are {} questions in total".format(len(raw_questions))
    print "Maximum path is {}".format(max_len)
    print "Mean path is: {}, Median path is: {}".format(np.mean(np.array(lens)), np.median(np.array(lens)))
    print "No path: {}".format(no_path)



def check_sparqls(raw_questions):
    wrong_query_file = open(os.path.join(INPUT_PREFIX, "wrong_query.txt"),'a')
    unexecutable_sparql = 0
    question_entity_not_exist = 0
    for raw_question in raw_questions:
        parses = raw_question[u"Parses"]
        if not sparql_executable(parses):
            unexecutable_sparql += 1
            if not question_entity_exist(raw_question):
                question_entity_not_exist += 1
            else:
                wrong_query_file.write(raw_question[u"QuestionId"] + '\n')
                wrong_query_file.write(raw_question[u"Parses"][0][u"Sparql"].encode('utf-8') + '\n')
                wrong_query_file.write('\n')

    print "There are {} questions in total".format(len(raw_questions))
    print "Unexecutable Query in total: {}".format(unexecutable_sparql)
    print "Question entity not exist: {}".format(question_entity_not_exist)


def check_dataset_entities():
    print("------------------------------------------------")
    path = os.path.join(INPUT_PREFIX, "WebQSP.train.json")
    print("Reading from {}".format(path))
    raw_questions = read_file(path)
    process_question_entities(raw_questions)
    print("------------------------------------------------")
    path = os.path.join(INPUT_PREFIX, "WebQSP.test.json")
    print("Reading from {}".format(path))
    raw_questions = read_file(path)
    process_question_entities(raw_questions)


def ebd_questions():
    logging.info("Start Embed Questions")
    print "----embedding questions----"
    if os.path.exists('questionIDToEbd.pickle'):
        return pickle.load(open('questionIDToEbd.pickle', 'rb'))
    questionIDToEbd = {}
    path = os.path.join(INPUT_PREFIX, "WebQSP.test.json")
    raw_questions = read_file(path)
    for index, question in enumerate(raw_questions):
        parses = question["Parses"]
        for parse in parses:
            if not parse[u"TopicEntityMid"]:
                continue
            mention = get_entity_mention(parse)
            ques_ebd, _ = kb_endpoint.ebd_question(question["ProcessedQuestion"].encode('utf-8').lower().split(), mention)
            questionIDToEbd[question["QuestionId"]] = ques_ebd
            break
    print "Finish Embed all the questions"
    pickle.dump(questionIDToEbd, open('questionIDToEbd.pickle', 'wb+'))
    return questionIDToEbd




def check_recall_subgraph(question_embeddings, only_namespace=False, strict=False, threshold=0.5):
    INPUT_PREFIX = "/Users/funke/Downloads/Data/webqsp/"
    path = os.path.join(INPUT_PREFIX, "WebQSP.test.json")
    raw_questions = read_file(path)
    total = len(raw_questions)
    answers_in_subgraph = 0
    rel_ebd_cache = {}
    for index, question in enumerate(raw_questions):
        logging.info("processing question " + question["QuestionId"] + ", Index: " + str(index))
        parses = question["Parses"]
        for parse in parses:
            if not parse[u"TopicEntityMid"]:
                continue
            topic_entity = parse["TopicEntityMid"]
            answer_entities = []
            answers = parse["Answers"]
            for answer in answers:
                answer_entities.append(answer["AnswerArgument"])
            if len(answers) == 0:
                continue
            graph = nx.DiGraph()
            start = time.time()
            relations = SPARQL.one_step(topic_entity).union(SPARQL.two_steps(topic_entity, strict=strict))
            logging.info("--- %s seconds ---" % (time.time() - start))
            for rel in relations:
                rel_ebd = np.zeros(300)
                if rel.relation_id in rel_ebd_cache.keys():
                    rel_ebd = rel_ebd_cache[rel.relation_id]
                else:
                    rel_ebd = kb_endpoint.ebd_relation(rel.relation_id, only_namespace=only_namespace)
                    rel_ebd_cache[rel.relation_id] = rel_ebd
                ques_ebd = question_embeddings[question["QuestionId"]]
                score = 1 - kb_endpoint.calculate_similarity(ques_ebd, rel_ebd)
                if score >= threshold:
                    rel.update_score(score)
                    graph = Relation.add_relation(rel, graph)
            logging.info("--- %d nodes in the subgraph ---" % (graph.number_of_nodes()))
            has_answer = False
            for answer in answers:
                if answer["AnswerType"] == "Entity":
                    answer_node = Entity(mid=answer["AnswerArgument"])
                else:
                    answer_node = Value(literal=answer["AnswerArgument"])
                answer_hashable_node = answer_node.as_hashable()
                if graph.has_node(answer_hashable_node):
                    has_answer = True
                    break # no need to check each answer
            if has_answer:
                logging.info('found answer in subgraph')
                answers_in_subgraph += 1
                break # no need to check each parse
    logging.info('%d of %d questions have entities in the subgraph' %(answers_in_subgraph, total))
    print('%d of %d questions have entities in the subgraph' %(answers_in_subgraph, total))
    sys.stdout.flush()

def debug():
    kbEndPoint = EndPoint()
    INPUT_PREFIX = "/Users/funke/Downloads/Data/webqsp"
    path = os.path.join(INPUT_PREFIX, "WebQSP.test.json")
    raw_questions = read_file(path)
    total = len(raw_questions)
    answers_in_subgraph = 0
    for index, question in enumerate(raw_questions):
        if index > 1:
            break
        logger.info("processing question " + question["QuestionId"] + ", Index: " + str(index))
        parses = question["Parses"]
        for parse in parses:
            if not parse[u"TopicEntityMid"]:
                continue
            topic_entity = parse["TopicEntityMid"]
            answer_entities = []
            answers = parse["Answers"]
            for answer in answers:
                answer_entities.append(answer["AnswerArgument"])
            if len(answers) == 0:
                continue
            graph = nx.DiGraph()
            start = time.time()
            relations = SPARQL.one_step(topic_entity).union(SPARQL.two_steps(topic_entity))
            for rel in relations:
                graph = Relation.add_relation(rel, graph)
            logger.info("--- %s seconds ---" % (time.time() - start))
            logger.info("--- %d nodes in the subgraph ---" % (graph.number_of_nodes()))
            has_answer = False
            for answer in answers:
                if answer["AnswerType"] == "Entity":
                    answer_node = Entity(mid=answer["AnswerArgument"])
                else:
                    answer_node = Value(literal=answer["AnswerArgument"])
                answer_hashable_node = answer_node.as_hashable()
                if graph.has_node(answer_hashable_node):
                    has_answer = True
                    break # no need to check each answer
            if has_answer:
                logger.info('found answer in subgraph')
                answers_in_subgraph += 1
                break # no need to check each parse
    logger.info('%d of %d questions have entities in the subgraph' %(answers_in_subgraph, total))
    print('%d of %d questions have entities in the subgraph' %(answers_in_subgraph, total))
    sys.stdout.flush()

if __name__ == '__main__':
    debug()


# if __name__ == '__main__':
#     global questionIDToEbd
#     global kb_endpoint
#     kb_endpoint = EndPoint()
#     questionIDToEbd = ebd_questions()
#     # print "----------Strict: {}, Only_NameSpace: {},threshold: {} -----------".format(True, False, 0)
#     # sys.stdout.flush()
#     # check_recall_subgraph(questionIDToEbd, strict=True, only_namespace=False, threshold=0)
#     # print "------------------------------------------\n"
#     # sys.stdout.flush()
#
#     print "----------Strict: {}, Only_NameSpace: {},threshold: {} -----------".format(False, False, 0.2)
#     sys.stdout.flush()
#     check_recall_subgraph(questionIDToEbd, strict=False, only_namespace=False, threshold=0.2)
#     print "------------------------------------------\n"
#     sys.stdout.flush()
#
#     print "----------Strict: {}, Only_NameSpace: {},threshold: {} -----------".format(False, False, 0.3)
#     sys.stdout.flush()
#     check_recall_subgraph(questionIDToEbd, strict=False, only_namespace=False, threshold=0.3)
#     print "------------------------------------------\n"
#     sys.stdout.flush()
#
#     print "----------Strict: {}, Only_NameSpace: {},threshold: {} -----------".format(False, False, 0.4)
#     sys.stdout.flush()
#     check_recall_subgraph(questionIDToEbd, strict=False, only_namespace=False, threshold=0.4)
#     print "------------------------------------------\n"
#     sys.stdout.flush()
#
#     print "----------Strict: {}, Only_NameSpace: {},threshold: {} -----------".format(False, False, 0.5)
#     sys.stdout.flush()
#     check_recall_subgraph(questionIDToEbd, strict=False, only_namespace=False, threshold=0.5)
#     print "------------------------------------------\n"
#     sys.stdout.flush()
#
#     print "----------Strict: {}, Only_NameSpace: {},threshold: {} -----------".format(False, True, 0.3)
#     sys.stdout.flush()
#     check_recall_subgraph(questionIDToEbd, strict=False, only_namespace=True, threshold=0.3)
#     print "------------------------------------------\n"
#     sys.stdout.flush()









