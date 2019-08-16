from kbEndPoint.utils.sparql import sparqlUtils
import os
import json

INPUT_PREFIX = "../../datasets/ComplexWebQuestions_1_1/Data/complex_web_questions/"
SPARQL = sparqlUtils()
ENCODING = 'utf-8'

def answer_exist(raw_question):
    answers = raw_question[u"answers"]
    for answer in answers:
        names = []
        try:
            names = SPARQL.get_names_alias(answer[u"answer_id"])
        except Exception as e:
            return False
        if len(names) != 0:
            return True
    return False



def check_answer():
    '''
    check if answer entities exist
    :return: write file
    '''
    global MISSING_ANSWER_FILE
    MISSING_ANSWER_FILE = "../../datasets/ComplexWebQuestions_1_1/Data/qid_mid_names/missing_answer_entity.tsv"
    global missing_answer_file
    missing_answer_file = open(MISSING_ANSWER_FILE, 'w+')
    CHOICES = ['train', 'dev']
    for choice in CHOICES:
        question_file = open(os.path.join(INPUT_PREFIX, "ComplexWebQuestions_" + choice + ".json"))
        raw_questions = json.load(question_file)
        for raw_question in raw_questions:
            if not answer_exist(raw_question):
                missing_answer_file.write('\t'.join([choice, raw_question[u"ID"]]) + '\n')

        question_file.close()



def check_answer_test():
    global MISSING_ANSWER_FILE_TEST
    MISSING_ANSWER_FILE_TEST = "../../datasets/ComplexWebQuestions_1_1/Data/qid_mid_names/missing_answer_entity_test.tsv"
    global missing_answer_test_file
    missing_answer_test_file = open(MISSING_ANSWER_FILE_TEST, 'w+')
    question_file = open(os.path.join(INPUT_PREFIX, "ComplexWebQuestions_test.json"))
    raw_questions = json.load(question_file)
    missing_answer = 0
    for raw_question in raw_questions:
        sparql_query = raw_question[u"sparql"]
        results = SPARQL.execute(sparql_query)[u"results"][u"bindings"]
        if len(results) == 0:
            missing_answer += 1
            missing_answer_test_file.write(raw_question["ID"] + '\n')

def check_sparql(input_path):
    question_file= open(input_path)
    raw_questions = json.load(question_file)
    missing_answer = 0
    missing_answer_test_file = open("../../datasets/ComplexWebQuestions_1_1/Data/unexecutable_queries.txt",'a+')
    for raw_question in raw_questions:
        sparql_query = raw_question[u"sparql"]
        try:
            idx = sparql_query.find("PREFIX")
            if not len(SPARQL.execute(sparql_query[idx:])[u"results"][u"bindings"]) > 0:
                missing_answer += 1
                missing_answer_test_file.write(raw_question["ID"] + '\n')
        except Exception as e:
            print sparql_query

    print "There are {} questions in total".format(len(raw_questions))
    print "Unexecutable Query in total: {}".format(missing_answer)


def shortestPath(input_path):
    # TODO Close
    question_file = open(input_path)
    raw_questions = json.load(question_file)
    question_file.close()


if __name__ == '__main__':
    print("------------------------------------------------")
    path = os.path.join(INPUT_PREFIX, "ComplexWebQuestions_dev.json")
    print("Reading from {}".format(path))
    check_sparql(path)
    print("------------------------------------------------")
    path = os.path.join(INPUT_PREFIX, "ComplexWebQuestions_test.json")
    print("Reading from {}".format(path))
    check_sparql(path)
    print("------------------------------------------------")
    path = os.path.join(INPUT_PREFIX, "ComplexWebQuestions_train.json")
    print("Reading from {}".format(path))
    check_sparql(path)


