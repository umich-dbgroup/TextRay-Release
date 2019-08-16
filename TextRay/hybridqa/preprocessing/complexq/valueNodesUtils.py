import json
import codecs
import os
from pycorenlp import StanfordCoreNLP



class ValueNodeIdentifier:
    def __init__(self):
        print 'initiating parser...'
        self.parser = StanfordCoreNLP('http://localhost:9000')

    def parse_mentions(self, sentence):
        try:
            sentence = str(sentence)
            annotated = self.parser.annotate(sentence, properties={'annotators': 'entitymentions', 'outputFormat': 'json'})['sentences'][0]
        except:
            print 'could not parse sentence'
            return []
        return annotated['entitymentions']

    def find_named_entities(self, ques_src, dest):
        questions = json.load(codecs.open(ques_src, 'r'))
        dest_file = open(dest, 'w+')
        for question in questions:
            question_str = question["question"]
            question_id = question["ID"]
            print(question_id)
            named_mentions = self.parse_mentions(question_str)
            if len(named_mentions) > 0:
                print("{} has {} entities".format(question_id, len(named_mentions)))
            for l in named_mentions:
                ner_tag = l["ner"]
                if ner_tag == "DATE" or ner_tag == "NUMBER" or ner_tag == "TIME":
                    begin_index = l["tokenBegin"]
                    end_index = l["tokenEnd"]
                    span = l["text"]
                    node_name = "\"" + span + "\""
                    if ner_tag == "DATE" or ner_tag == "TIME":
                        node_name += '^^xsd:dateTime'
                    line = question_id + "\t" + span + "\t" + str(begin_index) + "\t" + str(
                        end_index) + "\t" + node_name + "\t" + ner_tag + "\n"
                    print(line)
                    dest_file.write(line)
        dest_file.close()


if __name__ == '__main__':
    valueNodeIdentifier = ValueNodeIdentifier()

    valueNodeIdentifier.parse_mentions("what has lucy hale played in and that was released on 2009-09-08")

    valueNodeIdentifier.parse_mentions("What movie released in the fall of 2009 did Lucy Hale appear in ?")
    valueNodeIdentifier.parse_mentions("Which of the countries bordering China have country calling code 7 ?")
    valueNodeIdentifier.parse_mentions("What is my name ?")
