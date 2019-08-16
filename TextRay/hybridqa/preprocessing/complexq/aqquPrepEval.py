import json
import os
import codecs
from kbEndPoint.utils.sparql import sparqlUtils
import re

PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess"


def to_aqqu_json(ques_src, dest):
    questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
    aqqu_json = []
    for index, q in enumerate(questions):
        print(index)
        ques_str = q["question"].encode('ascii', 'ignore')
        ques_json = {
            "id": index,
            "utterance": ques_str,
            "targetOrigSparql": q["sparql"].replace('\n', ' '),
            "result": q["Answers"]
        }
        aqqu_json.append(ques_json)
    with open(dest, 'w+') as fp:
        json.dump(aqqu_json, fp, indent=4)

if __name__ == '__main__':
    INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
    OUTPUT_PATH = os.path.join(PREFIX, "aqqu_test.json")
    to_aqqu_json(INPUT_PATH,OUTPUT_PATH)


