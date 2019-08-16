import json
import numpy as np
from kbEndPoint.utils.sparql import sparqlUtils
from scipy.spatial import distance
import os
import heapq
import pickle
import logging
import math
import nltk

logging.basicConfig(filename="relation_semantics.log", level=logging.ERROR)
WEBQSP_TEST = "../../datasets/WebQSP/data/WebQSP.test.json"
GLOVE_FILE = "/home/xinyi/NLP_Resources/glove.6B/glove.6B.300d.txt"
GLOVE_PICKLE = "Glove.pickle"
WORD2VEC = {}
QUSTSIONWORDS = {'what', 'who', 'how', 'where', 'when'}
ENCODING = 'ISO-8859-1'
SPARQL = sparqlUtils()
VALID_WORD = {}
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def load_WebQSP():
    raw_questions = []
    with open(WEBQSP_TEST) as file:
        raw_questions = json.load(file)["Questions"]
    return raw_questions


def loadGlove():
    global VALID_WORD
    global WORD2VEC
    if os.path.exists(GLOVE_PICKLE):
        print "Loading Glove Model from Pickle"
        fp = open(GLOVE_PICKLE, 'rb')
        WORD2VEC = pickle.load(fp)
        VALID_WORD = set(WORD2VEC.keys())
        print "Finish Loading. Total Length: {}".format(str(len(WORD2VEC)))
        return
    print "Loading Glove Model from File"
    f = open(GLOVE_FILE, 'r')
    for line in f:
        tokens = line.split()
        word = tokens[0]
        embedding = np.asarray(tokens[1:], dtype='float32')
        WORD2VEC[word] = embedding
    with open(GLOVE_PICKLE, 'wb') as fp:
        pickle.dump(WORD2VEC, fp)
    VALID_WORD = set(WORD2VEC.keys())
    print "Finish Loading. Total Length: {}".format(str(len(WORD2VEC)))



def embed_relation_token(token):
    word_cnt = 0
    token_ebd = np.zeros(300)
    if token.lower() in VALID_WORD:
        return 1, WORD2VEC.get(token.lower())
    return word_cnt, token_ebd



def ebd_relation(rel_mid, domain=True, stopword=True, take_average = True):
    '''
    :param relation_mid: mid of relation, full uri type
    :param domain: whether consider the domain of relation or not
    :return: score
    '''
    tokens = SPARQL.getRelationTokens(rel_mid, domain=domain)
    relation_ebd = np.zeros(300)
    relation_word_cnt = .0
    for token in tokens:
        if token.lower() in QUSTSIONWORDS:
            continue
        if stopword and token.lower() in STOPWORDS:
            continue
        token_cnt, token_ebd = embed_relation_token(token)
        relation_word_cnt += token_cnt
        relation_ebd += token_ebd
        #logging.info("Legal relation word number: {}".format(str(relation_word_cnt)))
    if take_average:
        return relation_ebd / relation_word_cnt
    else:
        return relation_ebd


def ebd_question(tokens, entity_mention, stopword=True, take_average=True):
    question_ebd = np.zeros(300)
    question_word_cnt = .0
    for token in tokens:
        if token in QUSTSIONWORDS:
            continue
        if token in entity_mention:
            continue
        if stopword and token in STOPWORDS:
            continue
        if token in WORD2VEC.keys():
            question_ebd += WORD2VEC[token.lower()]
            question_word_cnt += 1
    logging.info("Legal Question Word Number: {}".format(str(question_word_cnt)))
    if take_average:
        if question_word_cnt == 0:
            return question_ebd
        question_ebd = question_ebd / question_word_cnt
        return question_ebd
    else:
        return question_ebd



class Relation:
    def __init__(self,rel_mid, score):
        self.mid = SPARQL.removePrefix(rel_mid)
        self.score = score

    def __cmp__(self, other):
        return self.score < other.score


def calcualateSimilarity(question_ebd, relation_ebd, metric):
    if metric == 'braycurtis':
        return distance.braycurtis(question_ebd, relation_ebd)
    elif metric == 'canberra':
        return distance.canberra(question_ebd, relation_ebd)
    elif metric == 'chebyshev':
        return distance.chebyshev(question_ebd, relation_ebd)
    elif metric == 'cityblock':
        return distance.cityblock(question_ebd, relation_ebd)
    elif metric == 'cosine':
        return distance.cosine(question_ebd, relation_ebd)
    elif metric == 'euclidean':
        return distance.euclidean(question_ebd, relation_ebd)
    elif metric == 'mahalanobis':
        return distance.mahalanobis(question_ebd, relation_ebd)
    elif metric == 'wminkowski':
        return distance.wminkowski(question_ebd, relation_ebd)





def process_question(rawquestion, domain=True, metric='cosine', stopword = True, take_average=True):
    '''
    :param question: raw questions from webqsp
    :param domain: whether relation reserves domain prefic
    :param metric: metric to compare relation vector and question vector
    :return: correct ranking, in top 1, in top 5, in top 10, in top 20
    None if two hop inference chain
    '''
    proccesed_question = rawquestion[u'ProcessedQuestion'].encode(ENCODING) #already case insensitive
    tokens = proccesed_question.split()
    entity_mention = rawquestion[u'Parses'][0][u'PotentialTopicEntityMention'].encode(ENCODING).split()
    entity_mid = rawquestion[u'Parses'][0][u'TopicEntityMid']
    correct_mids = rawquestion[u'Parses'][0][u"InferentialChain"]
    if correct_mids == None:
        return None, None
    if len(correct_mids) > 1:
        return None, None
    correct_mid = correct_mids[0].encode(ENCODING)
    #embed question seq
    question_ebd = ebd_question(tokens, entity_mention, stopword=stopword, take_average=take_average)
    #embed relations
    response_relations = SPARQL.getoneHop_Relation(mid=entity_mid)
    relationToscore = {}
    for response_relation in response_relations:
        relation = response_relation[u'p'][u'value'].encode(ENCODING)
        relation_ebd = ebd_relation(relation, domain=domain)
        if any(np.isnan(relation_ebd)) or any(np.isnan(question_ebd)):
            logging.error(proccesed_question)
            continue
        if np.all(relation_ebd == 0) or np.all(question_ebd == 0):
            logging.error(proccesed_question)
            continue
        relationToscore[SPARQL.removePrefix(relation)] = calcualateSimilarity(relation_ebd, question_ebd, metric=metric)
    sorted_relation = sorted(relationToscore.keys(), key=lambda x:relationToscore[x], reverse=False)
    for idx in range(len(sorted_relation)):
        if sorted_relation[idx] == correct_mid:
            return idx + 1, len(sorted_relation)
    return "<UNKNOWN>", len(sorted_relation)



def is_bad_edge(rank, length):
    if length > 150 and float(rank) / length > 0.25:
        return False
    if length <= 150 and length > 50 and 1.5e-5 * math.pow((length - 150), 2) + 0.25 < float(rank) / length:
        return False
    if length <= 50 and rank > 20:
        return False
    return True


def bad_edge_experiment():
    raw_questions = load_WebQSP()
    loadGlove()
    stats = []
    metrics = ['braycurtis', 'canberra', 'cosine']
    metrics = ['braycurtis']
    domains = [True]
    take_average = [False]
    for metric in metrics:
        for d in domains:
            for avg in take_average:
                print "Start Evaluating, domain: {}, metirc: {}, take average: {}".format(str(d), metric, avg)
                stats = []
                bad_edge_size = []
                global_edge_size = []
                for raw_question in raw_questions:
                    rank, length = process_question(raw_question, domain=d, metric=metric, take_average=avg)
                    if rank == None:
                        continue
                    if rank == "<UNKNOWN>":
                        continue
                    if not is_bad_edge(rank, length):
                        bad_edge_size.append(length)
                    logging.info("Rank: {}, Position: {}".format(rank, float(rank) / length))
                    stats.append(float(rank) / length)
                    global_edge_size.append(length)
                stats = np.array(stats)
                print "Total Number: {}, Mean: {:2f}, Median: {:2f}, Stddev: {:2f}".format(len(stats), np.mean(stats), np.median(stats), np.std(stats))
                global_edge_size = np.array(global_edge_size)
                print "Bad Edge Size: {}, Mean: {:2f}, Median: {:2f}, Stddev: {:2f}".format(len(bad_edge_size), np.mean(bad_edge_size),
                                                                              np.median(bad_edge_size),
                                                                              np.std(bad_edge_size))
                print "Global Edge Size: {}, Mean: {:2f}, Median: {:2f}, Stddev: {:2f}".format(len(global_edge_size), np.mean(global_edge_size),
                                                                                 np.median(global_edge_size),
                                                                                 np.std(global_edge_size))
                print "----------------------------------------------------------------------"





if __name__ == '__main__':
    bad_edge_experiment()



