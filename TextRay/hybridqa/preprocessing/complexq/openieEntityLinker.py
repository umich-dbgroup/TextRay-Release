from  el_helper import EL,EL_helper
import os
import json
import codecs
import pandas as pd
from preprocessing.stringUtils import jaccard_ch

import sys
sys.path.insert(0, '/media/nbhutani/Data/textray_workspace/TextRay/hybridqa/aqqu')

from entity_linker import entity_linker, surface_index_memory


class OpenIEEntityLinker(object):
    def __init__(self):
        self.linker = EL_helper()

    def test(self):
        sentence = "what character does ellen play in finding nemo"
        linked_entities = self.linker.link_text(sentence)
        for l in linked_entities:
            begin_index = l.tokens[0].index
            end_index = l.tokens[-1].index
            print(begin_index)
            print(end_index)
            tokens = l.tokens
            print(l.entity.sparql_name())
            print(l.name)
            print(l.surface_score)

    def link_triples(self, src_dir, dest_dir):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        files = os.listdir(src_dir)
        files = [f for f in files if f.endswith(".json")]
        for f in files:
            print(f.replace(".json", ""))
            triples_list = json.load(codecs.open(os.path.join(src_dir, f), 'r', encoding='utf-8'))
            dest_path = os.path.join(dest_dir, f)
            if os.path.exists(dest_path):
                continue
            dest_file = codecs.open(dest_path, 'w+', encoding='utf-8')
            counts = {}
            for triple in triples_list:
                if not 'triple_id' in triple:
                    src_id = "{}:{}:{}".format(triple['ques_id'], triple['snippet_id'], triple['sentence_id'])
                    triple_ct = counts.get(src_id, 0)
                    triple_id = src_id + ":" + str(triple_ct)
                    triple_ct += 1
                    counts[src_id] = triple_ct
                    triple['triple_id'] = triple_id
                triple_id = triple['triple_id']
                if float(triple['confidence']) < 0.75: continue # to reduce the number of triples to link
                snippet_index = triple['snippet_id']
                snippet_index = snippet_index.replace("split_part1_", "")
                snippet_index = snippet_index.replace("split_part2_", "")
                snippet_index = snippet_index.replace("full_question_", "")
                snippet_index = int(snippet_index)
                if snippet_index > 25: # only consider top-25 snippets per question
                    continue
                linked_entities = self.linker.link_text(triple['text_span'])
                for l in linked_entities:
                    begin_index = l.tokens[0].index
                    end_index = l.tokens[-1].index
                    mid = l.sparql_name()
                    mention = ' '.join(["%s" % t.token for t in l.tokens]).replace("\t", "")
                    name = l.name
                    score = l.surface_score
                    line = triple_id + "\t" + mention + "\t" + str(begin_index) + "\t" + str(end_index) + "\t" + mid + "\t" + name + "\t" + str(score) + "\n"
                    dest_file.write(line)
            dest_file.close()

    def filter_links(self, links_dir, dest_dir):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        files = os.listdir(links_dir)
        files = [f for f in files if f.endswith(".json")]
        header = ['triple_id', 'mention', 'begin_index', 'end_index', 'mid', 'name', 'score']
        for f in files:
            print(f.replace(".json", ""))
            links_path = os.path.join(links_dir, f)
            links_list = pd.read_csv(links_path, names=header, delimiter='\t',error_bad_lines=False, engine='python')
            links_list['score'] = links_list['score'].astype(float)
            links_list = links_list[links_list['score'] > 0.25]
            links = links_list.to_dict('records')
            dest_path = os.path.join(dest_dir, f)
            dest_file = codecs.open(dest_path, 'w+')
            for l in links:
                score = l['score']
                if score == 1.0 or score < 0.75:  ## there seems to be some issue with the surface index of aqqu
                    if jaccard_ch(l["mention"], l["name"]) < 0.5:
                        continue
                line = str(l["triple_id"]) + "\t" + str(l["mention"]) + "\t" + str(l["begin_index"]) + "\t" + str(l["end_index"]) + "\t" + str(l["mid"]) + "\t" + str(l["name"]) + "\t" + str(score) + "\n"
                dest_file.write(line)
            dest_file.close()


    def triple_link_annotations(self, src_dir, links_dir, dest_dir):
        header = ['triple_id', 'mention', 'begin_index', 'end_index', 'mid', 'name', 'score']
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        files = os.listdir(src_dir)
        files = [f for f in files if f.endswith(".json")]
        successful_link_ct = 0
        total_ct = 0
        for f in files:
            print(f.replace(".json", ""))
            triples_list = json.load(codecs.open(os.path.join(src_dir, f), 'r', encoding='utf-8'))
            links_path = os.path.join(links_dir, f)
            if not os.path.exists(links_path):
                print '{} does not have any triples with links'.format(f)
                continue
            total_ct += 1
            links_list = pd.read_csv(links_path, names=header, delimiter='\t',error_bad_lines=False, engine='python')
            linked_triples = []
            counts = {}
            for triple in triples_list:
                if not 'triple_id' in triple:
                    src_id = "{}:{}:{}".format(triple['ques_id'], triple['snippet_id'], triple['sentence_id'])
                    triple_ct = counts.get(src_id, 0)
                    triple_id = src_id + ":" + str(triple_ct)
                    triple_ct += 1
                    counts[src_id] = triple_ct
                    triple['triple_id'] = triple_id
                triple_id = triple['triple_id']
                links = links_list[links_list['triple_id'] == triple_id].to_dict('records')
                if len(links) == 0:
                    continue
                subject = str(triple['subject'].encode('utf-8'))
                object = str(triple['object'].encode('utf-8'))
                subject_links = []
                object_links = []
                for l in links:
                    mention = str(l['mention'])
                    if mention in subject:
                        subject_links.append({"mention": mention, "name": l['name'], "mid": l['mid'], "score": l['score']})
                    elif mention in object:
                        object_links.append({"mention": mention, "name": l['name'], "mid": l['mid'], "score": l['score']})
                    # else:
                    #     print mention
                    #     print subject
                    #     print object
                    #     print('not found')
                if len(subject_links) > 0 and len(object_links) > 0:
                    triple['subject_links'] = subject_links
                    triple['object_links'] = object_links
                    linked_triples.append(triple)
            if len(linked_triples) > 0:
                successful_link_ct += 1
            with open(os.path.join(dest_dir, f), 'w+') as fp:
                json.dump(linked_triples, fp, encoding='utf-8', indent=4)
        print("{} of {} had at least one triple with subject and object links".format(successful_link_ct, total_ct))

    def get_triple_cands(self, src_dir, dest_dir):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        files = os.listdir(src_dir)
        files = [f for f in files if f.endswith(".json")]
        for f in files:
            print(f.replace(".json", ""))
            linked_triples_list = json.load(codecs.open(os.path.join(src_dir, f), 'r', encoding='utf-8'))
            triple_cands = []
            for triple in linked_triples_list:
                subject_links = triple['subject_links']
                object_links = triple['object_links']
                for subject_link in subject_links:
                    for object_link in object_links:
                        candidate = {
                            "triple_confidence": triple['confidence'],
                            "triple_src_id": triple['triple_id'],
                            "snippet_id": triple['snippet_id'],
                            "subject_mid": subject_link['mid'],
                            "subject_name": subject_link['name'],
                            "subject_mention": subject_link['mention'],
                            "subject_span": triple['subject'],
                            "subject_score": subject_link['score'],
                            "relation": triple['relation'],
                            "object_mid": object_link['mid'],
                            "object_name": object_link['name'],
                            "object_mention": object_link['mention'],
                            "object_span": triple['object'],
                            "object_score": object_link['score'],
                        }
                        triple_cands.append(candidate)
            with open(os.path.join(dest_dir, f), 'w+') as fp:
                json.dump(triple_cands, fp, encoding='utf-8', indent=4)

if __name__ == '__main__':
    openie_linker = OpenIEEntityLinker()
    # openie_linker.test()
    prefix = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess/stanfordie'
    split = 'test'

    # relevant_triples_dir = os.path.join(prefix, 'relevant_triples', split)
    # links_dir = os.path.join(prefix, 'entity_links', split)
    # openie_linker.link_triples(relevant_triples_dir, links_dir)

    # linked_triples_dir = os.path.join(prefix, 'relevant_triple_links', split)
    # openie_linker.triple_link_annotations(relevant_triples_dir, links_dir, linked_triples_dir)
    #
    # linked_candidates_dir = os.path.join(prefix, 'relevant_cands', split)
    # openie_linker.get_triple_cands(linked_triples_dir, linked_candidates_dir)

    '''======='''

    data_prefix = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_1_1/stanfordie'
    relevant_triples_dir = os.path.join(data_prefix, split)
    links_dir = os.path.join(prefix, 'all', 'links', 'all_entity_links', split)
    # openie_linker.link_triples(relevant_triples_dir, links_dir)

    print split

    filtered_links_dir = os.path.join(prefix, 'all', 'links', 'filtered_entity_links', split)
    #openie_linker.filter_links(links_dir, filtered_links_dir)

    linked_triples_dir = os.path.join(prefix, 'all', 'links', 'triple_links', split)
    # openie_linker.triple_link_annotations(relevant_triples_dir, filtered_links_dir, linked_triples_dir)
    #
    linked_candidates_dir = os.path.join(prefix, 'all', 'links', 'triple_cands', split)
    openie_linker.get_triple_cands(linked_triples_dir, linked_candidates_dir)

