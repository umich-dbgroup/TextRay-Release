import json
import codecs
import pandas as pd
import re
import ast
import numpy as np
import os
import nltk
import ranking
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # will do for now?
from ranking import RankSVM
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import resample
from sklearn import svm, linear_model
from sklearn.preprocessing import MinMaxScaler
from preprocessing import metricUtils


ANS_CONSTRAINT_RELATIONS = ["people.person.gender", "common.topic.notable_types", "common.topic.notable_for"]

entity_pattern = re.compile(r'ns:([a-z]\.([a-zA-Z0-9_]+)) ')
variable_pattern = re.compile(r'\?(([a-zA-Z0-9]+)) ')

class QueryRanker(object):

    def __init__(self, dev_src, dev_topic_path, dev_predictions_path, dev_features_path, test_src, test_topic_path, test_predictions_path, test_features_path, lexicon_src):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.dev_src = dev_src
        self.dev_topic_path = dev_topic_path
        self.dev_predictions_path = dev_predictions_path
        self.dev_features_path = dev_features_path

        self.test_src = test_src
        self.test_topic_path = test_topic_path
        self.test_predictions_path = test_predictions_path
        self.test_features_path = test_features_path

        self.test_ground_ans_dict = self.get_ans_dict(test_src)
        self.dev_ground_ans_dict = self.get_ans_dict(dev_src)

        self.lexicon_src = lexicon_src


    def get_ans_dict(self, src):
        ans_dict = {}
        for q in json.load(codecs.open(src, 'r', encoding='utf-8'))["Questions"]:
            ques_id = q["QuestionId"]
            parses = q.get("Parses", [])
            entity_ans_dict = {}
            for parse in parses:
                topic_entity = parse["TopicEntityMid"]
                answers = parse.get("Answers", [])
                entity_ans_dict[topic_entity] = [a["AnswerArgument"] for a in answers]
            ans_dict[ques_id] = entity_ans_dict
        return ans_dict

    def read_features(self, ques_src, topics_path, predictions_path, features_path, lexicon_src):
        if not os.path.exists(features_path):
            print('no exists')
            self.extract_all_features(ques_src, topics_path, predictions_path, features_path, lexicon_src)

        df = pd.read_csv(features_path)
        if not 'ques_id' in df.columns:
            df['ques_id'] = df['qid']
        if 'sub1_relation' in df.columns:
            df['sub1_relation'] = df['sub1_relation'].apply(lambda x: ast.literal_eval(x))
        elif 'pred_sequence' in df.columns:
            df['pred_sequence'] = df['pred_sequence'].apply(lambda x: ast.literal_eval(x))
        if 'sub1_constraints' in df.columns:
            df['sub1_constraints'] = df['sub1_constraints'].apply(lambda x: ast.literal_eval(x))
        elif 'constraint_sequence_x' in df.columns:
            df['constraint_sequence_x'] = df['constraint_sequence_x'].apply(lambda x: ast.literal_eval(x))
        # df['features'] = df['features'].apply(lambda x: ast.literal_eval(x))
        df['features'] = df['features'].apply(lambda x: json.loads(x))
        if 'pred_entities' in df.columns:
            df['pred_entities'] = df['pred_entities'].apply(lambda x: ast.literal_eval(x))
        if not 'f1' in df:
            print 'getting f1 scores'
            ans_dict = self.get_ans_dict(ques_src)
            df['f1'] = df.apply(lambda x: self.get_record_f1(x, ans_dict=ans_dict), axis=1)
        else:
            df['f1'] = df['f1'].apply(lambda x: float(x))
        return df

    def get_record_f1(self, row, ans_dict):
        ground_answers = ans_dict[row['qid']].get(row['topic'], [])
        if len(ground_answers) == 0:
            recall, precision, f1 = 0.0, 0.0, 0.0
        else:
            recall, precision, f1 = metricUtils.compute_f1(ground_answers, row['pred_entities'])
        return f1

    # def extract_all_features(self, ques_src, topics_path, predictions_path, features_path, lexicon_src):
    #     if os.path.exists(features_path):
    #         print('features found {} at '.format(features_path))
    #         return
    #     print('no features found {} at '.format(features_path))
    #     questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))["Questions"]
    #     questions_dict = {}
    #     for q in questions:
    #         ques_str = q["ProcessedQuestion"]
    #         ques_tokens = set(nltk.word_tokenize(ques_str.lower()))
    #         ques_tokens = set([self.lemmatizer.lemmatize(w) for w in ques_tokens])
    #         q["tokens"] = ques_tokens
    #         questions_dict[q["QuestionId"]] = q
    #     topics_df = pd.read_csv(topics_path)
    #     lexicon_df = pd.read_csv(lexicon_src, names=['rel_keyword', 'ques_keyword'], sep='\t')
    #     print("ques_count {}".format(len(questions_dict)))
    #     df = pd.read_csv(predictions_path)
    #     print(df.shape)
    #     df['sub1_relation'] = df['sub1_relation'].apply(lambda x: ast.literal_eval(x))
    #     df['sub1_constraints'] = df['sub1_constraints'].apply(lambda x: ast.literal_eval(x))
    #     df['pred_entities'] = df['pred_entities'].apply(lambda x: ast.literal_eval(x))
    #     df['entity_scores'] = df['entity_scores'].apply(lambda x: ast.literal_eval(x))
    #     df['topic_scores'] = df.apply(lambda x: self.topic_scores(topics_df, x), axis=1)
    #
    #
    #     self.features_ct = 0
    #     df['features'] = df.apply(lambda x: self.extract_features(questions_dict, topics_df, lexicon_df, x), axis=1)
    #     df.to_csv(features_path, index=False)
    #
    # def


    def extract_all_features(self, ques_src, topics_path, predictions_path, features_path, lexicon_src):
        if os.path.exists(features_path):
            print('features found {} at '.format(features_path))
            return
        print('no features found {} at '.format(features_path))
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))["Questions"]
        questions_dict = {}
        for q in questions:
            ques_str = q["ProcessedQuestion"]
            ques_tokens = set(nltk.word_tokenize(ques_str.lower()))
            ques_tokens = set([self.lemmatizer.lemmatize(w) for w in ques_tokens])
            q["tokens"] = ques_tokens
            questions_dict[q["QuestionId"]] = q

        topics_df = pd.read_csv(topics_path)
        lexicon_df = pd.read_csv(lexicon_src, names=['rel_keyword', 'ques_keyword'], sep='\t')
        print("ques_count {}".format(len(questions_dict)))
        df = pd.read_csv(predictions_path)
        print(df.shape)
        df['sub1_relation_str'] = df['sub1_relation']
        if 'sub1_relation' in df.columns:
            df['sub1_relation'] = df['sub1_relation'].apply(lambda x: ast.literal_eval(x))

        # elif 'pred_sequence' in df.columns:
        #     df['pred_sequence'] = df['pred_sequence'].apply(lambda x: ast.literal_eval(x))
        df['sub1_constraints_str'] = df['sub1_constraints']
        if 'sub1_constraints' in df.columns:
            df['sub1_constraints'] = df['sub1_constraints'].apply(lambda x: ast.literal_eval(x))
        # elif 'constraints_sequence' in df.columns:
        #     df['constraints_sequence'] = df['constraint_sequence_x'].apply(lambda x: ast.literal_eval(x))
        if 'pred_entities' in df.columns:
            df['pred_entities'] = df['pred_entities'].apply(lambda x: ast.literal_eval(x))
        if 'entity_scores' in df.columns:
            df['entity_scores'] = df['entity_scores'].apply(lambda x: ast.literal_eval(x))
        print('starting feature extraction')
        # df['topic_scores'] = df.apply(lambda x: self.topic_scores(topics_df, x), axis=1)
        df['topic_scores'] = self.join_topic_scores(df, topics_df)
        print('extracting detailed features')
        self.features_ct= 0
        # df['features'] = df.apply(lambda x: self.extract_features(questions_dict, topics_df, lexicon_df, x), axis=1)

        rel_df=self.extract_all_rel_features(df, questions_dict, lexicon_df)
        updated_df = pd.merge(df, rel_df, how='left', on=['qid', 'topic','sub1_relation_str', 'sub1_constraints_str'])

        unique_columns_df = updated_df[['qid', 'topic', 'sub1_relation_x', 'sub1_constraints_x', 'pred_entities_x', 'agg_score_x', 'entity_scores_x', 'sub1_index_x', 'sub1_score_x', 'topic_scores_x', 'features']]
        unique_columns_df = unique_columns_df.rename(index=str, columns={"sub1_relation_x": "sub1_relation",
                                                                         "sub1_constraints_x": "sub1_constraints",
                                                                         "pred_entities_x": "pred_entities",
                                                                         "agg_score_x": "agg_score",
                                                                         "entity_scores_x": "entity_scores",
                                                                         "sub1_index_x": "sub1_index",
                                                                         "sub1_score_x": "sub1_score",
                                                                         "topic_scores_x": "topic_scores"})
        unique_columns_df['features'] = unique_columns_df.apply(lambda x: self.extract_ans_features(x), axis=1)
        unique_columns_df['features'] = unique_columns_df['features'].apply(lambda x: json.dumps(x))
        unique_columns_df.to_csv(features_path, index=False)

    def extract_all_rel_features(self, df, questions_dict, lexicon_df):
        print('extracting rel features')
        sub_df = df.drop_duplicates(subset=['qid', 'topic', 'sub1_relation_str', 'sub1_constraints_str'])
        sub_df['rel_tokens'] = sub_df['sub1_relation'].apply(lambda x: self.relation_tokens(x))
        sub_df['features'] = sub_df.apply(lambda x: self.extract_rel_features(questions_dict, lexicon_df, x), axis=1)
        print('extracted rel features')
        return sub_df

    def extract_rel_features(self, questions_dict, lexicon_df, query_result):
        self.features_ct = self.features_ct + 1
        if self.features_ct % 1000 == 0:
            print self.features_ct
        qid = query_result['qid']
        question = questions_dict[qid]
        sub1_relations = query_result['sub1_relation']
        constraints = query_result['sub1_constraints']
        prior_match_score = self.get_rel_surface_match_keywords(question["tokens"], query_result['rel_tokens'])
        type_constraint_ct = self.get_type_constraints_ct(constraints)  # type of constraint predictate
        constraints_match_score = self.get_constraint_surface_match(constraints)  # no. of constraint words in question
        # co_occurence_score = self.get_co_occurence_keywords(question["tokens"], query_result['rel_tokens'], lexicon_df)
        rel_ct = len(sub1_relations)
        topic_scores = query_result['topic_scores']
        constraint_ct = len(constraints)  # no. of constraints
        features = {
            "rel_ct": rel_ct,
            "constraint_ct": constraint_ct,
            "topic_scores": topic_scores,
            # "semantic_score": semantic_score,
            "prior_match_score": prior_match_score,
            "type_constraint_ct": type_constraint_ct,
            "constraints_match_score": constraints_match_score,
            # "co_occurence_score": co_occurence_score,
            "is_open": self.is_open(sub1_relations)
        }
        return features

    def extract_ans_features(self, query_result):
        answers = query_result['pred_entities']
        answer_ct = len(answers)
        entity_scores = query_result['entity_scores']
        ans_entity_linking_score = 0.0
        if len(entity_scores) > 0:
            ans_entity_linking_score = np.mean(entity_scores)
        features = query_result['features']
        features["ans_score"] = ans_entity_linking_score
        features["answer_ct"] = answer_ct
        return features

    def join_topic_scores(self, pred_df, topics_df):
        result = pd.merge(pred_df, topics_df, how='left', left_on=['qid', 'topic'], right_on=['ques_id', 'mid'])
        return result['score']

    def extract_features(self, questions_dict, topics_df, lexicon_df, query_result):
        if 'ques_id' in query_result:
            qid = query_result['ques_id']
        else:
            qid = query_result['qid']
        # print(qid)
        question = questions_dict[qid]
        if 'pred_sequence' in query_result: sub1_relations = query_result['pred_sequence']
        else: sub1_relations = query_result['sub1_relation']
        rel_ct = len(sub1_relations)
        if 'pred_ans' in query_result:
            answers = query_result['pred_ans']
        else:
            answers = query_result['pred_entities']
        answer_ct = len(answers)
        if 'constraints_sequence' in query_result: constraints = query_result['constraints_sequence']
        else: constraints = query_result['sub1_constraints']
        constraint_ct = len(constraints)  # no. of constraints
        if 'topic_entity' in query_result:
            topic_entity = query_result['topic_entity']
        else:
            topic_entity = query_result['topic']
        if 'topic_scores' in query_result:
            topic_scores = query_result['topic_scores']
        else:
            topic_scores = self.get_topic_scores(topics_df, qid, [topic_entity])  # sum of topic scores
        entity_scores = query_result['entity_scores']
        ans_entity_linking_score = np.mean(entity_scores)
        # semantic_score = query_result["score"]  # score
        prior_match_score = self.get_rel_surface_match(question["tokens"], sub1_relations)
        type_constraint_ct = self.get_type_constraints_ct(constraints)  # type of constraint predictate
        constraints_match_score = self.get_constraint_surface_match(constraints)  # no. of constraint words in question
        co_occurence_score = self.get_co_occurence(question["tokens"], sub1_relations, lexicon_df)
        features = {
            "rel_ct": rel_ct,
            "answer_ct": answer_ct,
            "constraint_ct": constraint_ct,
            "topic_scores": topic_scores,
            # "semantic_score": semantic_score,
            "prior_match_score": prior_match_score,
            "type_constraint_ct": type_constraint_ct,
            "constraints_match_score": constraints_match_score,
            "co_occurence_score": co_occurence_score,
            "is_open": self.is_open(sub1_relations),
            "ans_score": ans_entity_linking_score
        }
        return features

    def is_open(self, rels):
        if len(rels) == 0:
            return 0
        for rel in rels:
            if len(rel.split('.')) > 2:
                return 0
        return 1

    def get_co_occurence_keywords(self, ques_tokens, relation_keywords, lexicon_df):
        score = 0.0
        if len(relation_keywords) == 0 or len(ques_tokens) == 0:
            return 0.0
        for r in relation_keywords:
            ques_lexicon_entries = lexicon_df[lexicon_df['rel_keyword'] == r]['ques_keyword'].tolist()
            for q in ques_lexicon_entries:
                if q in ques_tokens:
                    score += 1.0
        return score

    def get_co_occurence(self, ques_tokens, relations, lexicon_df):
        relation_keywords = self.relation_tokens(relations)
        return self.get_co_occurence_keywords(ques_tokens, relation_keywords, lexicon_df)

    def get_type_constraints_ct(self, constraints):
        type_ct = 0
        for c in constraints:
            if c['relation'] in ANS_CONSTRAINT_RELATIONS:
                type_ct += 1
        return type_ct

    def get_constraint_surface_match(self, constraints):
        score = 0
        for c in constraints:
            surface_form = set(c["surface_form"].lower().split(" "))
            name = set(c["name"].lower().split(" "))
            match = surface_form.intersection(name)
            score += float(len(match)) * 1.0 / float(len(name))
        return score

    def get_rel_surface_match_keywords(self, ques_tokens, relation_keywords):
        if len(relation_keywords) == 0:
            return 1.0
        keywords_in_ques = relation_keywords.intersection(ques_tokens)
        return float(len(keywords_in_ques)) * 1.0/ float(len(relation_keywords))


    def get_rel_surface_match(self, ques_tokens, relations):
        relation_keywords = self.relation_tokens(relations)
        return self.get_rel_surface_match_keywords(ques_tokens, relation_keywords)

    def relation_tokens(self, relations):
        results = []
        for relation in relations:
            tokens = relation.split('.')
            for token in tokens:
                results = results + token.split('_')
        results = [r.lower() for r in results if r not in self.stopwords]
        return set(results)

    def get_topic_scores(self, topics_df, qid, entities):
        ques_df = topics_df[topics_df["ques_id"] == qid]
        topic_scores = 0.0
        for e in entities:
            row = ques_df[ques_df["mid"] == e]
            if len(row) > 0:
                score = row['score'].iloc[0]
                topic_scores += score
        return topic_scores

    def add_prior_score(self, row):
        return row['score'] + row['features']['prior_match_score']


    def re_rank_logistic(self, output_path=None):
        train_df = self.read_features(self.dev_src, self.dev_topic_path, self.dev_predictions_path, self.dev_features_path, self.lexicon_src)
        train_df['label'] = 0

        top_k_header = train_df.columns.values
        results = []
        sample_size = 5
        threshold = 0.3
        for qid, group_df in train_df.groupby("ques_id"):
            # print(qid)
            group_df = group_df.sort_values(by=["f1"], ascending=False)
            group_df_sub = group_df.head(min(len(group_df), sample_size))
            max_f1 = group_df_sub["f1"].max()
            for i, row in group_df_sub.iterrows():
                scaled_f1_score = row['f1']
                if max_f1 > 0:
                    scaled_f1_score = float(row['f1']) * 1.0 / float(max_f1)
                if scaled_f1_score > threshold:
                    row['label'] = 1
                results.append(row.values)

        to_train = pd.DataFrame.from_records(results, columns=top_k_header)
        X_train = np.array(pd.DataFrame.from_records(to_train['features']))
        scaler = preprocessing.StandardScaler().fit(X_train)
        y_train = np.array(to_train['label'])

        reg = LogisticRegression(random_state=0, solver='liblinear', max_iter=5000).fit(X_train, y_train)
        print(reg.score(X_train, y_train))

        test_df = self.read_features(self.test_src, self.test_topic_path, self.test_predictions_path, self.test_features_path, self.lexicon_src)
        test_df['prior_score'] = test_df.apply(lambda x: x.features['prior_match_score'], axis=1)
        test_df['co_score'] = test_df.apply(lambda x: x.features['co_occurence_score'], axis=1)
        X_test = np.array(pd.DataFrame.from_records(test_df['features']))
        X_test = scaler.transform(X_test)
        y_pred = reg.predict(X_test)
        y_score = reg.predict_proba(X_test)

        all_f1s = []
        top1_preds = []
        test_df['predicted_label'] = y_pred
        test_df['predicted_score'] = np.max(y_score, axis=1)
        test_df["total_score"] = test_df['predicted_score'] + test_df["score"] + test_df["co_score"]
        for qid, group_df in test_df.groupby("ques_id"):
            #print(qid)
            group_df_pos = group_df[group_df['predicted_label'] == 1]
            if len(group_df_pos) == 0:
                group_df_pos = group_df[group_df['predicted_label'] == 0]
                group_df_pos = group_df_pos.sort_values(by=["total_score"], ascending=False)
            else:
                group_df_pos = group_df_pos.sort_values(by=["predicted_score"], ascending=False)
            group_df_sub = group_df_pos.head(min(len(group_df), 1))
            best_f1 = group_df_sub['f1'].max()
            all_f1s.append(best_f1)
            top1_preds.append(group_df_sub.to_dict('records')[0])

        macro_avg_f1 = np.mean(all_f1s)
        print(macro_avg_f1)

        # top1_preds_df = pd.DataFrame.from_records(top1_preds)
        # if output_path is not None:
        #     top1_preds_df.to_csv(output_path, index=None)

    def get_biased_score(self, row):
        if row['features']['is_open']:
            return row['score']
        else:
            return row['score'] + 0.05

    def get_new_features(self, row):
        features = row['features']
        features['semantic_score'] = self.get_biased_score(row)
        return features

    def re_rank_regression(self, output_path=None, features_to_drop=None, topk=1):
        train_df = self.read_features(self.dev_src, self.dev_topic_path, self.dev_predictions_path, self.dev_features_path, self.lexicon_src)
        train_df['label'] = 0
        results = []
        sample_size = 10

        '''sort by f1_score'''
        for qid, group_df in train_df.groupby("ques_id"):
            group_df = group_df.sort_values(by=["f1"], ascending=False).head(min(len(group_df), sample_size))
            results += group_df.to_dict('records')

        features_df = pd.DataFrame(results)
        if not 'score' in features_df:
            features_df['score'] = features_df['sub1_score']
            features_df['score'] = features_df.apply(lambda x: self.get_biased_score(x) , axis=1)
        features_df['features']=features_df.apply(lambda x: self.get_new_features(x), axis=1)

        print("found features")

        features_df = features_df.reset_index()
        X_train_df = pd.DataFrame.from_records(features_df['features'])

        if features_to_drop is not None:
            X_train_df = X_train_df.drop(columns=features_to_drop)
        X_train = np.array(X_train_df)
        scaler = preprocessing.StandardScaler().fit(X_train)
        y_train = np.array(features_df['f1'])

        reg = LinearRegression().fit(X_train, y_train)
        # reg = IsotonicRegression().fit(X_train, y_train)
        # print(reg.score(X_train, y_train))
        print("finished training")
        test_df = self.read_features(self.test_src, self.test_topic_path, self.test_predictions_path, self.test_features_path, self.lexicon_src)
        # test_df['bias_score'] = test_df.apply(lambda x: self.get_bias(x) , axis=1)
        test_df['prior_score'] = test_df.apply(lambda x: x.features['prior_match_score'], axis=1)
        # test_df['co_score'] = test_df.apply(lambda x: x.features['co_occurence_score'], axis=1)
        test_df['total_score'] = 0.0
        if not 'score' in test_df:
            test_df['score'] = test_df['sub1_score']
            test_df['score'] = test_df.apply(lambda x: self.get_biased_score(x) , axis=1)
        test_df['features'] = test_df.apply(lambda x: self.get_new_features(x), axis=1)

        X_test_df = pd.DataFrame.from_records(test_df['features'])
        if features_to_drop is not None:
            X_test_df = X_test_df.drop(columns=features_to_drop)
        X_test = np.array(X_test_df)
        X_test = scaler.transform(X_test)
        y_pred = reg.predict(X_test)
        # y_pred = scaler.fit_transform(y_pred)

        print("re-ranking test")
        correct_top1_ct = 0
        total_prec1 = 0
        all_f1s = []
        openie_ct = 0

        test_df['pred_score'] = y_pred
        # test_df["total_score"] = test_df["agg"] + test_df["pred_score"] + test_df["prior_score"]
        topk_preds = []


        for qid, group_df in test_df.groupby("ques_id"):
            total_prec1 +=1
            group_df = group_df.reset_index()
            scaler = MinMaxScaler()
            group_df["pred_score"] = scaler.fit_transform(np.array(group_df['pred_score'].values).reshape(-1, 1)).reshape(1,-1)[0]
            #  group_df["total_score"] = group_df['pred_score'] + (group_df["score"]) + group_df['co_score']
            group_df["total_score"] = (0.5 * group_df['pred_score']) + (group_df["score"])\
                                      #+ group_df['bias_score']

            f1_sorted_records = group_df.sort_values(by=["total_score"], ascending=False).to_dict('records')

            best_pred_f1_row = f1_sorted_records[0]
            best_f1 = float(best_pred_f1_row['f1'])
            best_f1_topic = best_pred_f1_row['topic']
            best_f1_preds = best_pred_f1_row['pred_entities']
            if best_pred_f1_row['features']['is_open']:
                openie_ct += 1
            all_f1s.append(best_f1)

            ground_answers = self.test_ground_ans_dict[qid].get(best_f1_topic, [])
            for pred in best_f1_preds:
                if pred in ground_answers:
                    correct_top1_ct += 1
                    break

            topk_records = f1_sorted_records[:min(topk, len(f1_sorted_records))]
            topk_preds += topk_records

        prec_at_1 = float(correct_top1_ct) / len(self.test_ground_ans_dict)
        #macro_avg_f1 = np.mean(all_f1s)
        macro_avg_f1 = float(np.sum(all_f1s)) / len(self.test_ground_ans_dict)
        print("macro f1: {}".format(macro_avg_f1))
        print("prec@1: {}".format(float(correct_top1_ct) / float(total_prec1)))

        top1_preds_df = pd.DataFrame.from_records(topk_preds)
        if output_path is not None:
            top1_preds_df.to_csv(output_path, index=None)

    def pairwise_rank(self, n_best=1):
        train_df = self.read_features(self.dev_src, self.dev_topic_path, self.dev_predictions_path, self.dev_features_path, self.lexicon_src)
        train_df['rank'] = train_df.groupby('ques_id')['f1'].rank(ascending=False, method='first')
        train_df['group_id'] = train_df.groupby('ques_id').ngroup().add(1)
        sample_size = 5

        '''sort by f1_score'''
        X_train = np.empty((0, len(train_df['features'].iloc[0])))
        y_train = np.empty(0,)
        for qid, group_df in train_df.groupby("ques_id", as_index=False):
            # print(qid)
            group_df = group_df.sort_values(by=["f1"], ascending=False).head(min(len(group_df), sample_size))
            X_train_group = np.array(pd.DataFrame(list(group_df['features'])))
            y_train_group = np.array(list(group_df['rank']))
            if X_train_group.shape[0] < 2:
                continue
            X_train_group_transform, y_train_group_transform = ranking.transform_pairwise(X_train_group, y_train_group)
            X_train = np.append(X_train, X_train_group_transform, axis=0)
            y_train = np.append(y_train, y_train_group_transform, axis=0)


        print("train examples: {}".format(X_train.shape))
        clf = svm.LinearSVC(random_state=0, tol=1e-4, max_iter=10000)
        scaler = preprocessing.StandardScaler().fit(X_train)
        clf = clf.fit(X_train, y_train)
        print("finished training")

        test_df = self.read_features(self.test_src, self.test_topic_path, self.test_predictions_path, self.test_features_path, self.lexicon_src)
        all_f1s = []
        for qid, group_df in test_df.groupby("ques_id", as_index=False):
            # print(qid)
            features_df = pd.DataFrame(list(group_df['features']))
            X_test = np.array(features_df)
            X_test = scaler.transform(X_test)
            if X_train_group.shape[0] < 2:
                y_test = np.array([0])
            else:
                y_scores = np.dot(X_test, clf.coef_.ravel())
                group_df['re_rank_score'] = y_scores
                group_df['total'] = group_df['re_rank_score'] + group_df['score']
                #print(y_scores)
                group_df_total_scores = np.array(group_df['total'].values)
                y_test = np.argsort(group_df_total_scores)
                #y_test = np.argsort(y_scores)
            best_row_index = np.where(y_test == 0)[0]
            group_data = group_df['f1'].values
            best_f1 = group_data[best_row_index]
            all_f1s.append(best_f1)
        macro_avg_f1 = np.mean(all_f1s)
        print(macro_avg_f1)

def re_rank(ques_data_dir, dev_data_dir, test_data_dir, ranking_algorithm="regression", features_to_drop=None, epoch=5, topk=1):
    print("{} {}".format(test_data_dir, ranking_algorithm))

    test_input_path = os.path.join(ques_data_dir, "data/WebQSP.test.json")
    test_topic_path = os.path.join(ques_data_dir, "topics/test.csv")
    dev_input_path = os.path.join(ques_data_dir, "data/WebQSP.train.json")
    dev_topic_path = os.path.join(ques_data_dir, "topics/train.csv")

    dev_preds_path = os.path.join(dev_data_dir, "final_results.csv".format(epoch))
    dev_features_path = os.path.join(dev_data_dir, "features_train.csv".format(epoch))

    test_preds_path = os.path.join(test_data_dir, "final_results.csv".format(epoch))
    test_features_path = os.path.join(test_data_dir, "features_test.csv".format(epoch))
    test_reranked_path = os.path.join(test_data_dir, "prediction_top{}_reranked.csv".format(epoch, topk))

    queryRanker = QueryRanker(dev_input_path, dev_topic_path, dev_preds_path, dev_features_path,
                              test_input_path, test_topic_path, test_preds_path,
                              test_features_path, "lexicon")
    print(dev_features_path)
    print(test_features_path)
    queryRanker.extract_all_features(dev_input_path, dev_topic_path, dev_preds_path, dev_features_path, "lexicon")
    queryRanker.extract_all_features(test_input_path, test_topic_path, test_preds_path, test_features_path, "lexicon")

    if ranking_algorithm =="regression":
        queryRanker.re_rank_regression(test_reranked_path, features_to_drop, topk)
    elif ranking_algorithm=="logistic":
        queryRanker.re_rank_logistic()
    elif ranking_algorithm=="pairwise":
        queryRanker.pairwise_rank()


if __name__ == '__main__':
    ques_data_dir = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/WebQSP-final"

    # '''no retrain'''
    # DEV_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/webqsp_wo_retrain_trainpred/train_result"
    #
    #
    # # ablation experiment configs:
    # # full_model
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/webqsp_wo_retrain/test_result"
    # re_rank(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, ranking_algorithm="pairwise", topk=1)
    # # no constraints
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/wo_retrain_ablation/wo_cons/E_5_test_results"
    # re_rank(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, ranking_algorithm="regression", topk=1, epoch=5, features_to_drop=["constraint_ct", "type_constraint_ct", "constraints_match_score"])
    #
    # # no priors
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/wo_retrain_ablation/wo_prior/E_5_test_results"
    # re_rank(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, ranking_algorithm="regression", topk=1, epoch=5, features_to_drop=["prior_match_score", "co_occurence_score"])
    #
    # # no attention
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/wo_retrain_ablation/wo_attn/E_5_test_results"
    # re_rank(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, ranking_algorithm="regression", topk=1, epoch=5)
    #


    '''retrain'''
    # DEV_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/webqsp_w_retrain/attnTrue_consTrue_lr0.0005/E_9_train_results"

    # ablation experiment configs:
    # full_model
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/webqsp_w_retrain/attnTrue_consTrue_lr0.0005/E_9_test_results"
    # re_rank(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, ranking_algorithm="logistic", epoch=9, topk=1)
    #
    # no constraints
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/webqsp_w_retrain_ablation/wo_cons/E_9_test_results"
    # re_rank(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, epoch=9, ranking_algorithm="regression", features_to_drop=["constraint_ct", "type_constraint_ct", "constraints_match_score"])

    # no priors
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/webqsp_w_retrain_ablation/wo_prior/E_9_test_results"
    # re_rank(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, epoch=9, ranking_algorithm="regression", features_to_drop=["prior_match_score", "co_occurence_score"])

    # no attention
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/webqsp_w_retrain_ablation/wo_attn/E_9_test_results"
    # re_rank(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, epoch=9, ranking_algorithm="regression")

    '''------------------------------------------------'''



    '''emnlp data'''
    # DEV_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/webqsp/analyze_results/test_results_100/train_set/"
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/webqsp/analyze_results/test_results_100/test_set/"
    # re_rank(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, epoch=9, ranking_algorithm="regression",topk=1)


    working_dir =  "/media/nbhutani/Data/textray_workspace/emnlp_data/webqsp"
    test_input_path = os.path.join(ques_data_dir, "data/WebQSP.test.json")
    test_topic_path = os.path.join(ques_data_dir, "topics/test.csv")
    dev_input_path = os.path.join(ques_data_dir, "data/WebQSP.train.json")
    dev_topic_path = os.path.join(ques_data_dir, "topics/train.csv")

    configs = {
        "train": {
            "data_dir": "full_model/train_set",
            "pred_file": "prediction_align.csv",
            "features_file": "train_features.csv"
        },
        "kb_only": {
            "data_dir": "kbonly",
            "pred_file": "prediction.csv",
            "features_file": "test_features.csv"
        },
        "align": {
            "data_dir": "full_model/test_set",
            "pred_file": "prediction_align.csv",
            "features_file": "test_features.csv"
        },
        "wo_constraints": {
            "data_dir": "wo_cons",
            "pred_file": "prediction_align.csv",
            "features_file": "test_features.csv"
        },
        "wo_prior": {
            "data_dir": "wo_prior",
            "pred_file": "prediction_align.csv",
            "features_file": "test_features.csv"
        },
        "wo_attn": {
            "data_dir": "wo_attn3",
            "pred_file": "prediction_align.csv",
            "features_file": "test_features.csv"
        },
        # "wo_attn": {
        #     "data_dir": "wo_attn",
        #     "pred_file": "prediction_align.csv",
        #     "features_file": "test_features.csv"
        # },
        "kb_0.9": {
            "data_dir": "kb_downsample/kb_0.9",
            "pred_file": "prediction_align_0.9.csv",
            "features_file": "test_features.csv"
        },
        "kb_0.75": {
            "data_dir": "kb_downsample/kb_0.75",
            "pred_file": "prediction_align_0.75.csv",
            "features_file": "test_features.csv"
        },
        "kb_0.5": {
            "data_dir": "kb_downsample/kb_0.5",
            "pred_file": "prediction_align_0.5.csv",
            "features_file": "test_features.csv"
        }
    }

    dev_config = configs["train"]
    dev_data_dir = os.path.join(working_dir, dev_config["data_dir"])
    dev_preds_path = os.path.join(dev_data_dir, dev_config["pred_file"])
    dev_features_path = os.path.join(dev_data_dir, dev_config["features_file"])

    # test_config = configs["kb_only"]
    # test_config = configs["align"]
    # test_config = configs["wo_constraints"]
    # test_config = configs["wo_prior"]
    test_config = configs["wo_attn"]
    # test_config = configs["kb_0.9"]
    # test_config = configs["kb_0.75"]
    # test_config = configs["kb_0.5"]
    topk=1

    test_data_dir = os.path.join(working_dir, test_config["data_dir"])

    test_preds_path = os.path.join(test_data_dir, test_config["pred_file"])
    test_features_path = os.path.join(test_data_dir, test_config["features_file"])
    test_reranked_path = os.path.join(test_data_dir, "prediction_top{}_reranked.csv".format(topk))

    queryRanker = QueryRanker(dev_input_path, dev_topic_path, dev_preds_path, dev_features_path,
                              test_input_path, test_topic_path, test_preds_path,
                              test_features_path, "lexicon")
    # print(dev_features_path)
    # print(test_features_path)
    # queryRanker.extract_all_features(dev_input_path, dev_topic_path, dev_preds_path, dev_features_path, "lexicon")
    # queryRanker.extract_all_features(test_input_path, test_topic_path, test_preds_path, test_features_path, "lexicon")

    print(test_data_dir)
    queryRanker.re_rank_regression(test_reranked_path, features_to_drop=None, topk=topk)
