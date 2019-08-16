# from el_helper import EL,EL_helper
import os
import json
import codecs
import pandas as pd
from preprocessing.stringUtils import jaccard_ch
import nltk

import sys
sys.path.insert(0, '/media/nbhutani/Data/textray_workspace/TextRay/hybridqa/aqqu')

from entity_linker import entity_linker, surface_index_memory


class OpenIEEntityLinker(object):
    def __init__(self):
        self.linker = None#EL_helper()

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
                    src_id = "{}:{}".format(triple['ques_id'], triple['sentence_id'])
                    triple_ct = counts.get(src_id, 0)
                    triple_id = src_id + ":" + str(triple_ct)
                    triple_ct += 1
                    counts[src_id] = triple_ct
                    triple['triple_id'] = triple_id
                triple_id = triple['triple_id']
                if float(triple['confidence']) < 0.75: continue # to reduce the number of triples to link
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
        counter = 0
        for f in files:
            counter += 1
            if counter % 1000 == 0:
                print counter
            # print(f.replace(".json", ""))
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
                    src_id = "{}:{}".format(triple['ques_id'], triple['sentence_id'])
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
        counter = 0
        for f in files:
            counter += 1
            if counter % 1000 == 0:
                print counter
            # print(f.replace(".json", ""))
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
                            "snippet_id": triple['ques_id'],
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

    def combine_all_cands(self, dest_dir, dest_path):
        files = os.listdir(dest_dir)
        all_cands = []
        files = [f for f in files if f.endswith(".json")]
        for f in files:
            all_cands += json.load(codecs.open(os.path.join(dest_dir, f), 'r', encoding='utf-8'))
        with open(dest_path, 'w+') as fp:
            json.dump(all_cands, fp, encoding='utf-8', indent=4)

    def update_links(self, src_dir, triples_dir, dest_dir, doc_path):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        counter = 0
        with open(doc_path) as snippet_fp:
            for line in snippet_fp:
                counter += 1
                if counter % 1000 == 0:
                    print counter
                row = json.loads(line.rstrip())
                ques_id = str(row['documentId'])
                triples_path = os.path.join(triples_dir, ques_id + ".json")
                links_path = os.path.join(src_dir, ques_id + ".json")
                text_tokens = nltk.word_tokenize(row['document']['text'])
                gold_links = row['document']['entities']
                gold_links_dict = {}
                for g in gold_links:
                    mid = g['text'].replace("<fb:", "").replace(">", "")
                    g['mid'] = mid
                    if g['end'] == g['start']:
                        g['end'] = g['start'] + 1
                    g['mention'] = ' '.join(text_tokens[g['start']: g['end']])
                    if (len(g['mention']) == 0):
                        continue
                    g['tokens'] = nltk.word_tokenize(g['mention'])
                    gold_links_dict[mid] = g
                if not os.path.exists(triples_path) and not os.path.exists(links_path):
                    continue
                links_info = []
                if os.path.exists(links_path):
                    header = ['triple_id', 'mention', 'begin_index', 'end_index', 'mid', 'name', 'score']
                    links_info = pd.read_csv(links_path, names=header, delimiter='\t', error_bad_lines=False, engine='python').to_dict(orient='records')
                    for link in links_info:
                        if link['mid'] in gold_links_dict:
                            # print ques_id
                            # print(link)
                            link['score'] = 1.0
                if os.path.exists(triples_path):
                    counts = {}
                    triples_list = json.load(codecs.open(triples_path, 'r', encoding='utf-8'))
                    for triple in triples_list:
                        if not 'triple_id' in triple:
                            src_id = "{}:{}".format(triple['ques_id'], triple['sentence_id'])
                            triple_ct = counts.get(src_id, 0)
                            triple_id = src_id + ":" + str(triple_ct)
                            triple_ct += 1
                            counts[src_id] = triple_ct
                            triple['triple_id'] = triple_id
                        triple_id = triple['triple_id']
                        triple_text_span = triple['text_span']
                        # print triple_text_span
                        triple_text_tokens = nltk.word_tokenize(triple_text_span)
                        for mid, g in gold_links_dict.items():
                            if g['mention'] in triple_text_span:
                                # print ques_id
                                # print g['mention']
                                start_token = g['tokens'][0]
                                if start_token in triple_text_tokens:
                                    start_index = triple_text_tokens.index(start_token)
                                    end_index = start_index + len(g['tokens']) - 1
                                    mention = str(' '.join(triple_text_tokens[start_index: end_index+1]).encode('utf-8'))
                                    name = str(g['name'].encode('utf-8'))
                                    mid = g['mid']
                                    score = 1.0
                                    link_info = {"triple_id": str(triple_id),
                                                 "mention": mention,
                                                 'begin_index': start_index,
                                                 'end_index': end_index,
                                                 'mid': str(mid),
                                                 'name': name,
                                                 'score': score}
                                    links_info.append(link_info)
                                    # print link_info
                links_dest_path = os.path.join(dest_dir, ques_id + ".json")
                with open(links_dest_path, 'w+') as dest_file:
                    for link in links_info:
                        line = str(link['triple_id']) + "\t" + str(link['mention']) + "\t" + str(link['begin_index']) + "\t" + str(link['end_index']) + "\t" + str(link['mid']) + "\t" + str(link['name']) + "\t" + str(link['score']) + "\n"
                        dest_file.write(line)

    def compare_entity_links(self, dir1, dir2):
        files = os.listdir(dir1)
        files = [f for f in files if f.endswith(".json")]
        missed = 0
        for f in files:
            ct1 = len(open(os.path.join(dir1, f)).readlines())
            ct2 = len(open(os.path.join(dir2, f)).readlines())
            if ct1 > ct2:
                print f
                missed += 1
        print missed

if __name__ == '__main__':
    openie_linker = OpenIEEntityLinker()
    # openie_linker.test()
    prefix = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final/stanfordie"
    split = 'all'
    '''======='''

    data_prefix = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final/stanfordie'
    relevant_triples_dir = os.path.join(data_prefix, "all_triples")
    links_dir = os.path.join(prefix, 'all', 'links', 'all_entity_links', split)
    # openie_linker.link_triples(relevant_triples_dir, links_dir)
    print split

    filtered_links_dir = os.path.join(prefix, 'all', 'links', 'filtered_entity_links', split)
    #openie_linker.filter_links(links_dir, filtered_links_dir)

    linked_triples_dir = os.path.join(prefix, 'all', 'links', 'triple_links', split)
    # openie_linker.triple_link_annotations(relevant_triples_dir, filtered_links_dir, linked_triples_dir)
    #
    linked_candidates_dir = os.path.join(prefix, 'all', 'links', 'triple_cands', split)
    # openie_linker.get_triple_cands(linked_triples_dir, linked_candidates_dir)

    all_linked_candidates_path = os.path.join(prefix, 'all', 'links', 'triple_cands', split + ".json")
    # openie_linker.combine_all_cands(linked_candidates_dir, all_linked_candidates_path)


    '''---------enhancements------------'''

    new_links_dir = os.path.join(prefix, 'all_v2', 'links', 'all_entity_links', split)
    doc_path = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP/documents.json'
    # openie_linker.update_links(filtered_links_dir, relevant_triples_dir, new_links_dir, doc_path)

    new_filtered_links_dir = os.path.join(prefix, 'all_v2', 'links', 'all_entity_links', split)
    # openie_linker.filter_links(new_links_dir, new_filtered_links_dir)

    new_linked_triples_dir = os.path.join(prefix, 'all_v2', 'links', 'triple_links', split)
    # openie_linker.triple_link_annotations(relevant_triples_dir, new_filtered_links_dir, new_linked_triples_dir)

    new_linked_candidates_dir = os.path.join(prefix, 'all_v2', 'links', 'triple_cands', split)
    # openie_linker.get_triple_cands(new_linked_triples_dir, new_linked_candidates_dir)

    new_all_linked_candidates_path = os.path.join(prefix, 'all_v2', 'links', 'triple_cands', split + ".json")
    # openie_linker.combine_all_cands(new_linked_candidates_dir, new_all_linked_candidates_path)

    # openie_linker.compare_entity_links(filtered_links_dir, new_filtered_links_dir)






