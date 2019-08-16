import json
from kbEndPoint.utils.sparql import sparqlUtils

JSON_PATH = "/Users/funke/rel_constraint_processed_test.json"
JSON_DEST_PATH = "/Users/funke/rel_constraint_processed_entities_test.json"

if __name__ == '__main__':
    sparql = sparqlUtils()
    ques_json = json.load(open(JSON_DEST_PATH, 'r'))
    no_topic = 0
    for q in ques_json:
        print q['QuestionId']
        print q['TopicTypes']
        if len(q['TopicTypes']) == 0:
            no_topic += 1
    print(no_topic)
    # ques_json = json.load(open(JSON_PATH, 'r'))
    # for q in ques_json:
    #     print q['QuestionId']
    #     topics = q['Topics']
    #     topic_types_dict = {}
    #     for topic in topics:
    #         types = sparql.get_entity_types(topic)
    #         if types is None:
    #             continue
    #         topic_types_dict[topic] = types
    #     q['TopicTypes'] = topic_types_dict
    # json.dump(ques_json, open(JSON_DEST_PATH, 'w+'), indent=4)
