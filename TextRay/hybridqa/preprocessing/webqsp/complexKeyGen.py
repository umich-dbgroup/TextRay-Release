import json


WEB_QUESTION_TEST_PATH = "/Users/funke/WebQSP/data/WebQSP.test.json"
WEB_QUESTION_TRAIN_PATH = "/Users/funke/WebQSP/data/WebQSP.train.json"
TRAIN_PATH = "/Users/funke/ComplexWebQuestions/annotated_orig/train.json"
TEST_PATH = "/Users/funke/ComplexWebQuestions/annotated_orig/test.json"
DEV_PATH = "/Users/funke/ComplexWebQuestions/annotated_orig/dev.json"


WEB_QUESTION_COMPLEX_TEST_PATH = "/Users/funke/WebQSP/data/WebQSP.complex.test.json"
WEB_QUESTION_COMPLEX_TRAIN_PATH = "/Users/funke/WebQSP/data/WebQSP.complex.train.json"

if __name__ == '__main__':
    complex_questions = json.load(open(TRAIN_PATH, 'r'))
    complex_questions.extend(json.load(open(TEST_PATH, 'r')))
    complex_questions.extend(json.load(open(DEV_PATH, 'r')))

    ques_key_dict = {}
    for c in complex_questions:
        id = c["webqsp_ID"]
        ques = []
        if id in ques_key_dict:
            ques = ques_key_dict[id]
        ques.append(c["question"])
        ques_key_dict[id] = ques


    file_json = json.load(open(WEB_QUESTION_TRAIN_PATH, 'r'))
    web_train_ques = file_json["Questions"]
    for question in web_train_ques:
        compq = []
        if question["QuestionId"] in ques_key_dict: compq = ques_key_dict[question["QuestionId"]]
        question["ComplexQuestions"] = compq

    with open(WEB_QUESTION_COMPLEX_TRAIN_PATH, 'w+') as fp:
        json.dump(file_json, fp)

    file_json = json.load(open(WEB_QUESTION_TEST_PATH, 'r'))
    web_train_ques = file_json["Questions"]
    for question in web_train_ques:
        compq = []
        if question["QuestionId"] in ques_key_dict: compq = ques_key_dict[question["QuestionId"]]
        question["ComplexQuestions"] = compq

    with open(WEB_QUESTION_COMPLEX_TEST_PATH, 'w+') as fp:
        json.dump(file_json, fp)


