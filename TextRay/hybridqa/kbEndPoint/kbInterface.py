import numpy as np
from utils.sparql import sparqlUtils
from scipy.spatial import distance
import os
import pickle
import logging
import nltk
import networkx as nx
from dataModels.kbModels import Relation,Entity, Node, Value
import pandas as pd

class EndPoint(object):
    def __init__(self):
        logging.basicConfig(filename="KBInterface.log", level=logging.ERROR)
        #self.GLOVE_FILE = "/home/xinyi/NLP_Resources/glove.6B/glove.6B.300d.txt"
        self.GLOVE_FILE = "/Users/funke/glove.6B.300d.txt"
        self.GLOVE_PICKLE = "Glove.pickle"
        self.RELATIONS_FILE = "/Users/funke/Data/subgraphs/all_relations.txt"
        self.RELATION_EMBEDDINGS_FILE = "/Users/funke/Data/subgraphs/all_relations_embeddings.pkl"
        self.QUESTIONWORDS = {'what', 'who', 'how', 'where', 'when'}
        self.ENCODING = 'utf-8'
        self.STOPWORDS = set(nltk.corpus.stopwords.words('english'))
        self.WORD2VEC = {}
        self.RELEBDCACHE = {}
        self.question_warn = False
        self.undefined_token_split_pool = []

        self.SPARQL = sparqlUtils()
        #self.__loadGlove__()
        self.__load_relation_embeddings__()


    def getEntityById(self, mid):
        name = self.SPARQL.get_names(mid)
        return Entity(mid, name)

    def get_subgraph(self, question, entity_mid, entity_mention, domain=True, metric='cosine', take_average=True, semantics_threshold=0.5):
        '''
        :param question: question text
        :param entity_mention: topic entity mention of sub question
        :param entity_mid: mid of entity in the format of m.xxx
        :param metric: metric to use in the pool of {braycurtis, canberra, cityblock, cosine, euclidean}
        :param entity_mention: list of words in topic entity mention
        :param hop: length of path returned in the graph instance
        :return: digraph instance of the topic entity
        '''
        graph = nx.DiGraph()
        question_ebd, _ = self.__ebd_question__(question.split(), entity_mention, take_average=take_average)
        relations = self.SPARQL.one_step(entity_mid, relations_to_filter=self.RELATIONS_TO_FILTER).union(self.SPARQL.two_steps(entity_mid, strict=True, relations_to_filter=self.RELATIONS_TO_FILTER))
        for rel in relations:
            if rel.relation_id in self.RELEBDCACHE.keys():
                rel_ebd = self.RELEBDCACHE[rel.relation_id]
            else:
                rel_ebd= self.__ebd_relation__(rel.relation_id)
            score = 1 - self.calculate_similarity(question_ebd, rel_ebd, metric='cosine')
            if score < 0:
                continue
            else:
                rel.update_score(score * score)
                graph = Relation.add_relation(rel, graph)
        return graph

    def get_shortestPathGraph(self, entity_1, entity_2, threshold=4):
        '''
        :param entity_1: Entity Instance
        :param entity_2: Entity Instance
        :return: instance of a digrah, containing the shortest path
        '''
        length, relations = self.SPARQL.shortestPathResponse(entity_1.mid, entity_2.mid, threshold=threshold, relationsToFilter=self.RELATIONS_TO_FILTER)
        graph = nx.DiGraph()
        for relation in relations:
            graph = Relation.add_relation(relation, graph)
        return graph

    def ebd_subquestion(self, subquestion):
        tokens = subquestion.sub_ques.encode(self.ENCODING).strip().lower().split()
        entity_mention_set = set()
        for entity_obj in subquestion.entities:
            this_mention = entity_obj.lower().split()
            for token in this_mention:
                entity_mention_set.add(token)
        question_ebd, _ = self.__ebd_question__(tokens, entity_mention_set)
        return question_ebd

    def ebd_openie_relation(self, relation_id, stopword=True, take_average=True):
        relation_ebd = np.zeros(300)
        tokens = relation_id.strip().encode(self.ENCODING).split()
        for token in tokens:
            if token.lower() in self.QUESTIONWORDS or token.lower() in self.STOPWORDS:
                continue
            _, token_ebd = self.__embed_relation_token__(token)
            relation_ebd += token_ebd
        return relation_ebd

    def calculate_similarity(self, question_ebd, relation_ebd, metric='cosine'):
        if metric == 'braycurtis':
            return distance.braycurtis(question_ebd, relation_ebd)
        elif metric == 'canberra':
            return distance.canberra(question_ebd, relation_ebd)
        elif metric == 'cityblock':
            return distance.cityblock(question_ebd, relation_ebd)
        elif metric == 'cosine':
            return distance.cosine(question_ebd, relation_ebd)
        elif metric == 'euclidean':
            return distance.euclidean(question_ebd, relation_ebd)

    def __load_relation_embeddings__(self):
        if os.path.exists(self.RELATIONS_FILE):
            self.RELATIONS_TO_FILTER = set(pd.read_csv(self.RELATIONS_FILE, names=['rel']).rel)
            if os.path.exists(self.RELATION_EMBEDDINGS_FILE):
                print "loading relations embeddings"
                fp = open(self.RELATION_EMBEDDINGS_FILE, 'rb')
                self.RELEBDCACHE = pickle.load(fp)
                print "finished loading relations embeddings. Total Length: {}".format(str(len(self.RELEBDCACHE)))
            else:
                print "loading relations"
                f = open(self.RELATIONS_FILE, 'r')
                for line in f:
                    rel = line.rstrip()
                    embedding = self.ebd_relation(rel)
                    self.RELEBDCACHE[rel] = embedding
                with open(self.RELATION_EMBEDDINGS_FILE, 'wb') as fp:
                    pickle.dump(self.RELEBDCACHE, fp)
                print "finished loading relations embeddings. Total Length: {}".format(str(len(self.RELEBDCACHE)))

    def __loadGlove__(self):
        if os.path.exists(self.GLOVE_PICKLE):
            logging.info("Loading Glove Model from Pickle")
            fp = open(self.GLOVE_PICKLE, 'rb')
            self.WORD2VEC = pickle.load(fp)
            logging.info("Finish Loading. Total Length: {}".format(str(len(self.WORD2VEC))))
            return
        print "Loading Glove Model from File"
        f = open(self.GLOVE_FILE, 'r')
        for line in f:
            tokens = line.split()
            word = tokens[0]
            embedding = np.asarray(tokens[1:], dtype='float32')
            self.WORD2VEC[word] = embedding
        with open(self.GLOVE_PICKLE, 'wb') as fp:
            pickle.dump(self.WORD2VEC, fp)
        print "Finish Loading. Total Length: {}".format(str(len(self.WORD2VEC)))
        logging.info("Finish Loading. Total Length: {}".format(str(len(self.WORD2VEC))))


    def __split_token__(self, sub_token, word_cnt, token_ebd):
        if len(sub_token) == 0:
            return word_cnt, token_ebd
        cursor = len(sub_token)
        while cursor > 1:
            if sub_token[:cursor].lower() in self.WORD2VEC.keys():
                if sub_token[:cursor].lower() not in self.STOPWORDS and sub_token[:cursor].lower() not in self.QUESTIONWORDS:
                    word_cnt += 1
                    token_ebd += self.WORD2VEC.get(sub_token[:cursor].lower())
                    self.__split_token__(sub_token[cursor:].lower(), word_cnt, token_ebd)
                    self.undefined_token_split_pool.append(sub_token[:cursor].lower())
                    break
            cursor -= 1
        return word_cnt, token_ebd

    def __embed_relation_token__(self, token):
        if token.lower() in self.WORD2VEC.keys():
            return 1, self.WORD2VEC.get(token.lower())
        else:
            self.undefined_token_split_pool = []
            return self.__split_token__(token.encode(self.ENCODING), 0, np.zeros(300))


    def __ebd_relation__(self, rel_mid, only_namespace=False, domain=True, namespace_weight=None, stopword=True, take_average=True):
        '''
        :param relation_mid: mid of relation, full uri type
        :param domain: whether consider the domain of relation or not
        :return: score
        '''
        if namespace_weight is not None:
            # weighted embedding
            namespace_token = self.SPARQL.getRelationNamespace(rel_mid)
            namespace_ebd = np.zeros(300)
            namespace_token_cnt = 0
            for token in namespace_token:
                if token.lower() in self.QUESTIONWORDS:
                    continue
                if stopword and token.lower() in self.STOPWORDS:
                    continue
                token_cnt, token_ebd = self.__embed_relation_token__(token)
                namespace_ebd += token_ebd
                namespace_token_cnt += token_cnt
            if take_average:
                namespace_ebd = namespace_ebd / namespace_token_cnt

            rel_name_token = self.SPARQL.getRelationTokens(rel_mid, domain=False)
            rel_name_ebd = np.zeros(300)
            rel_name_token_cnt = 0
            for token in rel_name_token:
                if token.lower() in self.QUESTIONWORDS:
                    continue
                if stopword and token.lower() in self.STOPWORDS:
                    continue
                token_cnt, token_ebd = self.embed_relation_token(token)
                rel_name_ebd += token_ebd
                rel_name_token_cnt += token_cnt
            if take_average:
                rel_name_ebd = rel_name_ebd / rel_name_token_cnt
            return namespace_weight * namespace_ebd + (1 - namespace_weight) * rel_name_ebd
        else:
            # normal embedding
            tokens = self.SPARQL.getRelationTokens(rel_mid, only_namespace=only_namespace, domain=domain)
            relation_ebd = np.zeros(300)
            relation_word_cnt = .0
            for token in tokens:
                if token.lower() in self.QUESTIONWORDS:
                    continue
                if stopword and token.lower() in self.STOPWORDS:
                    continue
                token_cnt, token_ebd = self.embed_relation_token(token)
                relation_word_cnt += token_cnt
                relation_ebd += token_ebd
                #logging.info("Legal relation word number: {}".format(str(relation_word_cnt)))
            if relation_word_cnt == 0:
                print "Undefined Relation: {}".format(rel_mid)
            if take_average:
                return relation_ebd / relation_word_cnt
            else:
                return relation_ebd

    def __ebd_question__(self, tokens, entity_mention, stopword=True, take_average=True, question_word=True):
        self.question_warn = False
        question_phrase = []
        question_ebd = np.zeros(300)
        question_word_cnt = .0
        for token in tokens:
            if question_word and token in self.QUESTIONWORDS:
                continue
            if token in entity_mention:
                continue
            if stopword and token in self.STOPWORDS:
                continue
            if token in self.WORD2VEC.keys():
                question_ebd += self.WORD2VEC[token.lower()]
                question_word_cnt += 1
                question_phrase.append(token)
        logging.info("Legal Question Word Number: {}".format(str(question_word_cnt)))
        if np.all(question_ebd == 0):
            self.question_warn = True
            return self.__ebd_question__(tokens, entity_mention, stopword=False, take_average=take_average, question_word=False)
        if take_average:
            if question_word_cnt == 0:
                self.question_warn = True
                return self.__ebd_question__(tokens, entity_mention, stopword=False, take_average=take_average, question_word=False)
            question_ebd = question_ebd / question_word_cnt
            return question_ebd, question_phrase
        else:
            return question_ebd, question_phrase

if __name__ == '__main__':
    kb = EndPoint()
    print kb.get_shortestPathGraph(entity_1=Entity('m.06w2sn5'), entity_2=Entity('m.0gxnnwq')).nodes









