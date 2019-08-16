import json
import os
import codecs
from kbEndPoint.utils.sparql import sparqlUtils
import re

PREFIX = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess"

line_pattern = re.compile("iter=0.test: example ([0-9]+)/[0-9]+: testinput:[0-9]+ \\{")
ans_pattern = re.compile("\\(name fb\\:en\\.(.*?)\\s+(.*?)\\)")

def to_sempre_json(ques_src, dest):
    questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
    sempre_json = []
    for q in questions:
        ques_str = q["question"].encode('ascii', 'ignore')
        ques_str = ques_str.replace(":", "")
        sempre_json.append({"utterance": ques_str})
    with open(dest, 'w+') as fp:
        json.dump(sempre_json, fp, indent=4)


def get_ans_from_line(line):
    s = "(value (list"
    line = line[line.find(s) + len(s): ]
    ans = []
    for m in re.finditer(ans_pattern, line):
        match = m.group(2).replace("\"", "")
        if len(match) > 0:
           ans.append(match)
    return ans

def get_predictions(log_src):
    answers_set = {}
    contents = open(log_src, "r")
    for line in contents:
        m = line_pattern.search(line)
        if m:
            index = m.group(1)
            all_ans = []
            for line in contents:
                if "Cumulative(iter=0.test):" in line:
                    break
                else:
                    if "(derivation (formula" in line:
                        ans = get_ans_from_line(line)
                        all_ans += ans
            answers_set[index] = all_ans
            print("{} in {}".format(index, all_ans))
    return answers_set

def evaluate(ques_src, log):
    sparql = sparqlUtils()
    questions = json.load(codecs.open(ques_src, encoding='utf-8'))
    ans_dict = {}
    for i, ques in enumerate(questions):
        answers = []
        for mid in ques["Answers"]:
            print(mid)
            name = sparql.get_names(mid)
            if name is not None:
                answers.append(name)
        print("{} has ans {}".format(i, answers))
        ans_dict[i] = answers

    predictions = get_predictions(log)
    no_preds_ct = 0
    correct_preds_ct = 0
    total_ct = 0
    predicted_ct = 0
    for q_index in predictions:
        total_ct += 1
        preds = predictions[q_index]
        if len(preds) == 0:
            no_preds_ct += 1
        else:
            predicted_ct += 1
            best_pred = preds[0]
            grounds_ans = ans_dict[q_index]
            if best_pred in grounds_ans:
                correct_preds_ct += 1

    precision = float(correct_preds_ct) * 1.0 / float(predicted_ct)
    recall = float(correct_preds_ct) * 1.0 / float(total_ct)

    f1_score = 2 * precision * recall / (precision + recall)

    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("f1: {}".format(f1_score))


if __name__ == '__main__':
    INPUT_PATH = os.path.join(PREFIX, "annotated/test.json")
    OUTPUT_PATH = "/home/nbhutani/Documents/sempre/complexTestInput"
    LOGS = "sempre.txt"
    to_sempre_json(INPUT_PATH, OUTPUT_PATH)
    # evaluate(INPUT_PATH, LOGS)
