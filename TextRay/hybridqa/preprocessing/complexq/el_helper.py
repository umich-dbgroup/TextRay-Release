import sys
sys.path.insert(0, '/media/nbhutani/Data/textray_workspace/TextRay/hybridqa/aqqu')

from entity_linker import entity_linker, surface_index_memory
from corenlp_parser import parser
import json
import codecs

class EL:
    def __init__(self):
        base = '/media/nbhutani/Data/textray_workspace/TextRay/hybridqa/aqqu'
        print 'initiating parser...'
        # self.parser = parser.CoreNLPParser('http://localhost:4000/parse')
        self.parser = parser.CoreNLPParser('http://localhost:4000/parse')
        print 'initiating index...'
        self.surface_index = surface_index_memory.EntitySurfaceIndexMemory(base + '/data/entity-list',
                                                                           base + '/data/entity-surface-map',
                                                                           base + '/data/entity-index')
        print 'initiating entity_linker'
        self.entity_linker = entity_linker.EntityLinker(self.surface_index, 7)
        print 'initiation done.'

    def link(self, sentence):
        try:
            parse_result = self.parser.parse(sentence)
        except:
            print('{} error parsing'.format(sentence))
            return []
        identified = self.entity_linker.identify_entities_in_tokens(parse_result.tokens)
        return identified

    def parse(self, sentence):
        return self.parser.parse(sentence)

    def link_with_parse(self, tokens):
        return self.entity_linker.identify_entities_in_tokens(tokens)

class EL_helper:
    def __init__(self):
        self.is_data_loaded = False
        self.linker = None

    def load(self):
        self.linker = EL()

    def link_text(self, sentence):
        if not self.is_data_loaded:
            self.load()
            self.is_data_loaded = True
        raw_result = self.linker.link(sentence)
        el_items = []
        for item in raw_result:
            if isinstance(item.entity, entity_linker.KBEntity):
                el_items.append(item)
            elif isinstance(item.entity, entity_linker.DateValue):
                el_items.append(item)
        return el_items


def test(el_helper):
    sentence = "Barack Obama was the president of USA."
    linked_entities = el_helper.link_text(sentence)
    for l in linked_entities:
        begin_index = l.tokens[0].index
        end_index = l.tokens[-1].index
        print(begin_index)
        print(end_index)
        tokens = l.tokens
        print(l.entity.sparql_name())
        print(l.name)
        print(l.surface_score)

def link_subquestion(el_helper, part_id, src, dest):
    questions = json.load(codecs.open(src, 'r', encoding='utf-8'))
    dest_file = codecs.open(dest, 'w+', encoding='utf-8')
    for question in questions:
        question_str = question["split_part" + str(part_id)]["sub_ques"]
        question_id = question["ID"]
        print(question_id)
        linked_entities = el_helper.link_text(question_str)
        for l in linked_entities:
            begin_index = l.tokens[0].index
            end_index = l.tokens[-1].index
            mid = l.sparql_name()
            mention = ' '.join(["%s" % t.token for t in l.tokens]).replace("\t", "")
            name = l.name
            score = l.surface_score
            line = question_id + "\t" + mention + "\t" + str(begin_index) + "\t" + str(
                end_index) + "\t" + mid + "\t" + name + "\t" + str(score) + "\n"
            dest_file.write(line)
    dest_file.close()


def link_complex_questions(el_helper, src, dest):
    questions = json.load(codecs.open(src, 'r', encoding='utf-8'))
    dest_file = codecs.open(dest, 'w+', encoding='utf-8')
    for question in questions:
        question_str = question["question"]
        question_id = question["ID"]
        print(question_id)
        linked_entities = el_helper.link_text(question_str)
        for l in linked_entities:
            begin_index = l.tokens[0].index
            end_index = l.tokens[-1].index
            mid = l.sparql_name()
            mention = ' '.join(["%s" % t.token for t in l.tokens]).replace("\t", "")
            name = l.name
            score = l.surface_score
            line = question_id + "\t" + mention + "\t" + str(begin_index) + "\t" + str(end_index) + "\t" + mid + "\t" + name + "\t" + str(score) + "\n"
            dest_file.write(line)
    dest_file.close()

# def test():
#     el_helper = EL_helper()
#     return el_helper.link_text("who plays bob kelso in scrubs")

if __name__ == '__main__':
    SRC = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess/annotated/train.json"
    DEST = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess/el/main/train_el.csv"
    el_helper = EL_helper()
    # link_complex_questions(el_helper, src=SRC, dest=DEST)
    #link_subquestion(el_helper, part_id=2, src=SRC, dest=DEST)
    test(el_helper)