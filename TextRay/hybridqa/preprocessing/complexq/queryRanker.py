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
from preprocessing import  metricUtils

ANS_CONSTRAINT_RELATIONS = ["people.person.gender", "common.topic.notable_types", "common.topic.notable_for"]

entity_pattern = re.compile(r'ns:([a-z]\.([a-zA-Z0-9_]+)) ')
variable_pattern = re.compile(r'\?(([a-zA-Z0-9]+)) ')

class QueryRanker(object):

    def __init__(self, dev_src, dev_topic_path, dev_predictions_path, dev_queries_dir, dev_features_path, test_src, test_topic_path, test_predictions_path, test_queries_dir, test_features_path, lexicon_src):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.dev_src = dev_src
        self.dev_topic_path = dev_topic_path
        self.dev_predictions_path = dev_predictions_path
        self.dev_queries_dir = dev_queries_dir
        self.dev_features_path = dev_features_path

        self.test_src = test_src
        self.test_topic_path = test_topic_path
        self.test_predictions_path = test_predictions_path
        self.test_queries_dir = test_queries_dir
        self.test_features_path = test_features_path

        self.lexicon_src = lexicon_src


    def read_features(self, ques_src, topics_path, predictions_path, queries_dir, features_path, lexicon_src):
        if not os.path.exists(features_path):
            self.extract_all_features(ques_src, topics_path, predictions_path, queries_dir, features_path, lexicon_src)

        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
        questions_dict = {}
        for q in questions:
            questions_dict[q["ID"]] = q

        df = pd.read_csv(features_path)
        df[['sub2_relation', 'sub2_constraints']] = df[['sub2_relation', 'sub2_constraints']].fillna(value='[]')
        df['sub2_index'] = df['sub2_index'].fillna(value=-1)
        df['sub1_relation'] = df['sub1_relation'].apply(lambda x: ast.literal_eval(x))
        df['sub2_relation'] = df['sub2_relation'].apply(lambda x: ast.literal_eval(x))
        df['sub1_constraints'] = df['sub1_constraints'].apply(lambda x: ast.literal_eval(x))
        df['sub2_constraints'] = df['sub2_constraints'].apply(lambda x: ast.literal_eval(x))
        df['features'] = df['features'].apply(lambda x: ast.literal_eval(x))
        df['pred_entities'] = df['pred_entities'].apply(lambda x: ast.literal_eval(x))
        # df['f1_score'] = df['f1_score'].apply(lambda x: float(x))
        df['f1_score'] = df.apply(lambda x: self.get_f1(questions_dict, x), axis=1)
        return df

    def get_f1(self, ques_dict, row):
        ques_id = row['qid']
        question = ques_dict[ques_id]
        ground_answers = question["Answers"]
        if len(ground_answers) == 0:
            recall, precision, f1 = 0.0, 0.0, 0.0
        else:
            recall, precision, f1 = metricUtils.compute_f1(ground_answers, row['pred_entities'])
        return f1

    def extract_all_features(self, ques_src, topics_path, predictions_path, queries_dir, features_path, lexicon_src):
        if os.path.exists(features_path):
            return
        questions = json.load(codecs.open(ques_src, 'r', encoding='utf-8'))
        questions_dict = {}
        for q in questions:
            ques_str = q["question"]
            ques_tokens = set(nltk.word_tokenize(ques_str.lower()))
            ques_tokens = set([self.lemmatizer.lemmatize(w) for w in ques_tokens])
            q["tokens"] = ques_tokens
            questions_dict[q["ID"]] = q

        topics_df = pd.read_csv(topics_path)
        lexicon_df = pd.read_csv(lexicon_src, names=['rel_keyword', 'ques_keyword'], sep='\t')
        print("ques_count {}".format(len(questions_dict)))
        df = pd.read_csv(predictions_path)
        print(df.shape)
        self.counter = 0
        df[['sub2_relation', 'sub2_constraints']] = df[['sub2_relation', 'sub2_constraints']].fillna(value='[]')
        df['sub2_index'] = df['sub2_index'].fillna(value=-1)
        df['sub1_relation'] = df['sub1_relation'].apply(lambda x: ast.literal_eval(x))
        df['sub2_relation'] = df['sub2_relation'].apply(lambda x: ast.literal_eval(x))
        df['sub1_constraints'] = df['sub1_constraints'].apply(lambda x: ast.literal_eval(x))
        df['sub2_constraints'] = df['sub2_constraints'].apply(lambda x: ast.literal_eval(x))
        df['pred_entities'] = df['pred_entities'].apply(lambda x: ast.literal_eval(x))

        df['features'] = df.apply(lambda x: self.extract_features(questions_dict, topics_df, queries_dir, lexicon_df, x), axis=1)
        df.to_csv(features_path, index=False)


    def extract_features(self, questions_dict, topics_df, queries_dir, lexicon_df, query_result):
        self.counter = self.counter + 1
        if self.counter % 5000 == 0:
            print self.counter
        qid = query_result['qid']
        #print(qid)
        question = questions_dict[qid]
        if "query" in query_result:
            query = query_result["query"]
        else:
            queries_path = os.path.join(queries_dir, "query_" + qid + ".json")
            if os.path.exists(queries_path):
                queries_df = pd.read_csv(queries_path)
                queries_df[['sub2_relation', 'sub2_constraints']] = queries_df[['sub2_relation', 'sub2_constraints']].fillna(value='[]')
                queries_df['sub2_index'] = queries_df['sub2_index'].fillna(value=-1)
                query = queries_df[(queries_df['sub1_index'] == query_result["sub1_index"]) &(queries_df['sub2_index'] == query_result["sub2_index"])]['query'].values
                if len(query) == 0:
                    print qid
                    print query_result["sub1_index"]
                    print query_result["sub2_index"]

        ques_entities = self.get_entities_from_sparql(query)
        variables = self.get_variables_from_sparql(query)
        sub1_relations = query_result["sub1_relation"]
        if sub1_relations is None:
            sub1_relations = []

        sub2_relations = query_result["sub2_relation"]
        if sub2_relations is None:
            sub2_relations = []
        relations = sub1_relations + sub2_relations
        sub1_constraints = query_result["sub1_constraints"]
        if sub1_constraints is None:
            sub1_constraints = []
        sub2_constraints = query_result["sub2_constraints"]
        if sub2_constraints is None:
            sub2_constraints = []
        constraints = sub1_constraints + sub2_constraints
        unique_entities = set(query_result["pred_entities"]).union(ques_entities)

        rel1_ct = len(sub1_relations)
        rel2_ct = len(sub2_relations)
        query_entities_ct = len(ques_entities) # no. of entities
        query_nodes_ct = query_entities_ct + len(variables)
        answer_ct = len(query_result["pred_entities"])
        rel_ct = len(relations) # no. of relations
        constraint_ct = len(constraints) # no. of constraints
        topic_scores = self.get_topic_scores(topics_df, qid, ques_entities, query) # sum of topic scores
        semantic_score = query_result["agg_score"] # score
        prior_match_score = self.get_rel_surface_match(question["tokens"], relations)
        type_constraint_ct = self.get_type_constraints_ct(constraints) # type of constraint predictate
        constraints_match_score = self.get_constraint_surface_match(constraints) # no. of constraint words in question
        rel_diversity = self.get_relations_diversity(sub1_relations, sub2_relations)
        co_occurence_score = self.get_co_occurence(question["tokens"], relations, lexicon_df)
        is_rel1_open = self.is_open(sub1_relations)
        is_rel2_open = self.is_open(sub2_relations)
        is_hybrid = 0
        if is_rel1_open > 0 or is_rel2_open > 0: is_hybrid = 1

        sub1_score = query_result["sub1_score"]
        if np.isnan(sub1_score):
            sub1_score = 0.0
        sub2_score = query_result["sub2_score"]
        if np.isnan(sub2_score):
            sub2_score = 0.0
        unique_entities_ct = len(unique_entities)
        return {
            "query_entities_ct": query_entities_ct,
            "query_nodes_ct": query_nodes_ct,
            "answer_ct": answer_ct,
            "rel_ct" : rel_ct,
            "constraint_ct": constraint_ct,
            "topic_scores": topic_scores,
            "semantic_score": semantic_score,
            "prior_match_score": prior_match_score,
            "co_occurence_score": co_occurence_score,
            "type_constraint_ct": type_constraint_ct,
            "constraints_match_score": constraints_match_score,
            "rel1_ct": rel1_ct,
            "rel2_ct": rel2_ct,
            "rel_diversity": rel_diversity,
            "sub1_score": sub1_score,
            "sub2_score": sub2_score,
            "unique_entities_ct": unique_entities_ct,
            "is_hybrid": is_hybrid,
            "is_rel1_open": is_rel1_open,
            "is_rel2_open": is_rel2_open
        }

    def is_open(self, rels):
        if len(rels) == 0:
            return 0
        for rel in rels:
            if len(rel.split('.')) > 2:
                return 0
        return 1


    def get_relations_diversity(self, rels1, rels2):
        rel1_keywords = self.relation_tokens(rels1)
        rel2_keywords = self.relation_tokens(rels2)
        intersect = len(rel1_keywords.intersection(rel2_keywords))
        union = len(rel1_keywords) + len(rel2_keywords) - intersect
        return 1.0 - (float(intersect) * 1.0 / float(union))


    def get_co_occurence(self, ques_tokens, relations, lexicon_df):
        score = 0.0
        relation_keywords = self.relation_tokens(relations)
        if len(relation_keywords) == 0 or len(ques_tokens) == 0:
            return 0.0
        for r in relation_keywords:
            ques_lexicon_entries = lexicon_df[lexicon_df['rel_keyword'] == r]['ques_keyword'].tolist()
            for q in ques_lexicon_entries:
                if q in ques_tokens:
                    score += 1.0
        return score

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


    def get_rel_surface_match(self, ques_tokens, relations):
        relation_keywords = self.relation_tokens(relations)
        if len(relation_keywords) == 0:
            return 1.0
        keywords_in_ques = relation_keywords.intersection(ques_tokens)
        return float(len(keywords_in_ques)) * 1.0/ float(len(relation_keywords))


    def relation_tokens(self, relations):
        results = []
        for relation in relations:
            tokens = relation.split('.')
            for token in tokens:
                results = results + token.split('_')
        results = [r.lower() for r in results if r not in self.stopwords]
        return set(results)

    def get_topic_scores(self, topics_df, qid, entities, query):
        ques_df = topics_df[topics_df["ques_id"] == qid]
        topic_scores = 0.0
        for e in entities:
            # print e
            row = ques_df[ques_df["mid"] == e]
            # print row
            if len(row) > 0:
                score = row['score'].iloc[0]
                topic_scores += score

        # if topic_scores == 0.0:
        #     print qid
        #     print entities
        #     print query
        return topic_scores


    def get_variables_from_sparql(self, query):
        variables = set()
        try:
            matches = re.findall(variable_pattern, str(query))
        except:
            matches = []
        for m in matches:
            if m[0] == "name":
                continue
            variables.add(m[0])
        return variables


    def get_entities_from_sparql(self, query):
        entities = set()
        try:
            matches = re.findall(entity_pattern, str(query))
        except:
            matches = []
        for m in matches:
            entities.add(m[0])
        return entities

    '''look at questions where model best is not the same as best f1'''
    def analyze_missclassifed(self, n_best=1):
        df = self.read_features(self.test_src, self.test_topic_path, self.test_predictions_path, self.test_queries_dir, self.test_features_path, self.lexicon_src)

        questions = json.load(codecs.open(self.test_src, 'r', encoding='utf-8'))
        questions_dict = {}
        for q in questions:
            questions_dict[q["ID"]] = q

        all_f1s = []
        for qid, group in df.groupby("qid"):
            group = group.reset_index()

            # features_df = pd.DataFrame(list(group['features']))
            # max_prior_match_score = features_df['prior_match_score'].max()
            # if max_prior_match_score == 0:
            #     max_prior_match_score = 1.0
            # group['scaled_prior'] = features_df['prior_match_score'].apply((lambda x: x * 1.0 / max_prior_match_score))
            # group['agg'] = group['agg'] + group['scaled_prior']
            group_df_scored = group.sort_values(by=["agg"], ascending=False)
            group_df_scored_sub = json.loads(group_df_scored.head(min(len(group_df_scored), n_best)).to_json(orient='records'))
            best_f1 = group_df_scored_sub[0]['f1_score']
            all_f1s.append(best_f1)
            group_df_f1_ranked = group.sort_values(by=["f1_score"], ascending=False)
            group_df_f1_ranked_sub = json.loads(group_df_f1_ranked.head(min(len(group_df_f1_ranked), n_best)).to_json(orient='records'))
            for i in range(n_best):
                row1 = group_df_scored_sub[i]
                row2 = group_df_f1_ranked_sub[i]
                if row1['sub1_index'] != row2['sub1_index'] and row1['sub2_index'] != row2['sub2_index']:
                    print(qid)
                    print(questions_dict[qid]["question"])
                    print("predicted best sub1 relation {} with sub2 relation {} has f1 score: {} and model score: {} and prior score: {}".format(row1['sub1_relation'],
                                                                                row1['sub2_relation'], row1['f1_score'],
                                                                                row1['agg_score'], row1['features']['prior_match_score']))
                    print("true best sub1 relation {} with sub2 relation {} has f1 score: {} and model score: {} and prior score: {}".format(row2['sub1_relation'],
                                                                                row2['sub2_relation'], row2['f1_score'],row2['agg_score'],
                                                                                row2['features']['prior_match_score']))
        print(np.mean(all_f1s))
        print(np.max(all_f1s))

    '''look at ranks of the best f1'''
    def analyze_ranks(self, n_best=1):
        df = self.read_features(self.test_src, self.test_topic_path, self.test_predictions_path, self.test_queries_dir, self.test_features_path, self.lexicon_src)
        all_ranks = []
        for qid, group in df.groupby("qid"):
            group = group.reset_index()
            group['agg_rank'] = group['agg'].rank(ascending=False)
            group['f1_rank'] = group['f1_score'].rank(ascending=False)
            group_df_f1_ranked = group.sort_values(by=["f1_score"], ascending=False)
            group_df_f1_ranked_sub = json.loads(group_df_f1_ranked.head(min(len(group_df_f1_ranked), n_best)).to_json(orient='records'))
            for i in range(n_best):
                row = group_df_f1_ranked_sub[i]
                agg_rank = row['agg_rank']
                all_ranks.append(agg_rank)
        print(np.mean(all_ranks))
        print(np.min(all_ranks))
        print(np.max(all_ranks))

    '''simply add prior score to re-rank'''
    def re_rank_no_learning(self, n_best=1):
        df = self.read_features(self.test_src, self.test_topic_path, self.test_predictions_path, self.test_queries_dir, self.test_features_path, self.lexicon_src)
        all_f1s = []

        df['agg_score'] = df.apply(lambda x: self.add_prior_score(x), axis=1)
        for qid, group in df.groupby("qid"):
            group_df = group.reset_index()
            group_df = group_df.sort_values(by=["agg_score"], ascending=False).head(min(len(group_df), n_best))
            best_f1 = group_df['f1_score'].max()
            all_f1s.append(best_f1)

        macro_avg_f1 = np.mean(all_f1s)
        print(macro_avg_f1)


    def add_prior_score(self, row):
        return row['agg'] + row['features']['prior_match_score']


    def re_rank_violation(self):
        train_df = self.read_features(self.dev_src, self.dev_topic_path, self.dev_predictions_path, self.dev_queries_dir, self.dev_features_path, self.lexicon_src)
        train_df['label'] = 0

        train_data = []
        differences = []
        for qid, group in train_df.groupby("qid"):
            best_model_pred = group.sort_values(by=["agg_score"], ascending=False).to_dict('records')[0]
            best_ground = group.sort_values(by=["f1_score"], ascending=False).to_dict('records')[0]
            best_ground_score = best_ground['agg_score']
            best_pred_score = best_model_pred['agg_score']
            difference = abs(best_pred_score - best_ground_score)
            differences.append(difference)
            # if difference > 0.01:
            if best_pred_score != best_ground_score:
                best_ground['label'] = 1
                train_data.append(best_ground)
                best_model_pred['label'] = 0
                train_data.append(best_model_pred)
        print(np.mean(differences))
        print(np.max(differences))

        to_train = pd.DataFrame(train_data)
        X_train = np.array(pd.DataFrame.from_records(to_train['features']))
        scaler = preprocessing.StandardScaler().fit(X_train)
        y_train = np.array(to_train['label'])

        reg = LogisticRegression(random_state=0, solver='liblinear', max_iter=5000).fit(X_train, y_train)
        print(reg.score(X_train, y_train))

        test_df = self.read_features(self.test_src, self.test_topic_path, self.test_predictions_path, self.test_queries_dir, self.test_features_path, self.lexicon_src)
        test_df['prior_score'] = test_df.apply(lambda x: x.features['prior_match_score'], axis=1)
        X_test = np.array(pd.DataFrame.from_records(test_df['features']))
        X_test = scaler.transform(X_test)
        y_pred = reg.predict(X_test)
        y_score = reg.predict_proba(X_test)

        all_f1s = []
        test_df['predicted_label'] = y_pred
        test_df['predicted_score'] = np.max(y_score, axis=1)
        test_df["total_score"] = test_df['predicted_score']  + test_df['agg_score']# test_df["agg"] + test_df["prior_score"]
        for qid, group_df in test_df.groupby("qid"):
            # print(qid)
            group_df_pos = group_df[group_df['predicted_label'] == 1]
            if len(group_df_pos) == 0:
                group_df_pos = group_df[group_df['predicted_label'] == 0]
                group_df_pos = group_df_pos.sort_values(by=["total_score"], ascending=False)
            else:
                group_df_pos = group_df_pos.sort_values(by=["total_score"], ascending=False)
            group_df_sub = group_df_pos.head(min(len(group_df), 1))
            best_f1 = group_df_sub['f1_score'].max()
            all_f1s.append(best_f1)

        macro_avg_f1 = np.mean(all_f1s)
        print(macro_avg_f1)


    def re_rank_logistic(self, output_path=None):
        train_df = self.read_features(self.dev_src, self.dev_topic_path, self.dev_predictions_path, self.dev_queries_dir, self.dev_features_path, self.lexicon_src)
        train_df['label'] = 0

        top_k_header = train_df.columns.values
        results = []
        sample_size = 5
        threshold = 0.3
        for qid, group_df in train_df.groupby("qid"):
            # print(qid)
            group_df = group_df.sort_values(by=["f1_score"], ascending=False)
            group_df_sub = group_df.head(min(len(group_df), sample_size))
            max_f1 = group_df_sub["f1_score"].max()
            for i, row in group_df_sub.iterrows():
                scaled_f1_score = row['f1_score']
                if max_f1 > 0:
                    scaled_f1_score = float(row['f1_score']) * 1.0 / float(max_f1)
                if scaled_f1_score > threshold:
                    row['label'] = 1
                results.append(row.values)

        to_train = pd.DataFrame.from_records(results, columns=top_k_header)
        X_train = np.array(pd.DataFrame.from_records(to_train['features']))
        scaler = preprocessing.StandardScaler().fit(X_train)
        y_train = np.array(to_train['label'])

        reg = LogisticRegression(random_state=0, solver='liblinear', max_iter=5000).fit(X_train, y_train)
        print(reg.score(X_train, y_train))

        test_df = self.read_features(self.test_src, self.test_topic_path, self.test_predictions_path, self.test_queries_dir, self.test_features_path, self.lexicon_src)
        test_df['prior_score'] = test_df.apply(lambda x: x.features['prior_match_score'], axis=1)
        X_test = np.array(pd.DataFrame.from_records(test_df['features']))
        X_test = scaler.transform(X_test)
        y_pred = reg.predict(X_test)
        y_score = reg.predict_proba(X_test)

        all_f1s = []
        top1_preds = []
        test_df['predicted_label'] = y_pred
        test_df['predicted_score'] = np.max(y_score, axis=1)
        test_df["total_score"] = test_df['predicted_score'] + test_df["agg_score"] + test_df["prior_score"]
        for qid, group_df in test_df.groupby("qid"):
            #print(qid)
            group_df_pos = group_df[group_df['predicted_label'] == 1]
            if len(group_df_pos) == 0:
                group_df_pos = group_df[group_df['predicted_label'] == 0]
                group_df_pos = group_df_pos.sort_values(by=["total_score"], ascending=False)
            else:
                group_df_pos = group_df_pos.sort_values(by=["predicted_score"], ascending=False)
            group_df_sub = group_df_pos.head(min(len(group_df), 1))
            best_f1 = group_df_sub['f1_score'].max()
            all_f1s.append(best_f1)
            top1_preds.append(group_df_sub.to_dict('records')[0])

        macro_avg_f1 = np.mean(all_f1s)
        print(macro_avg_f1)

        # top1_preds_df = pd.DataFrame.from_records(top1_preds)
        # if output_path is not None:
        #     top1_preds_df.to_csv(output_path, index=None)

    def re_rank_regression(self, output_path=None, features_to_drop=None, topk=1):
        train_df = self.read_features(self.dev_src, self.dev_topic_path, self.dev_predictions_path, self.dev_queries_dir, self.dev_features_path, self.lexicon_src)
        train_df['label'] = 0
        results = []
        sample_size = 10

        questions = json.load(codecs.open(self.test_src, 'r', encoding='utf-8'))
        questions_dict = {}
        for q in questions:
            if q["compositionality_type"] == "composition" or q["compositionality_type"] == "conjunction":
                questions_dict[q["ID"]] = q

        '''sort by f1_score'''
        for qid, group_df in train_df.groupby("qid"):
            group_df = group_df.sort_values(by=["f1_score"], ascending=False).head(min(len(group_df), sample_size))
            results += group_df.to_dict('records')

        features_df = pd.DataFrame(results)

        print("found features")

        features_df = features_df.reset_index()
        X_train_df = pd.DataFrame.from_records(features_df['features'])
        if features_to_drop is not None:
            X_train_df = X_train_df.drop(columns=features_to_drop)
        X_train = np.array(X_train_df)
        scaler = preprocessing.StandardScaler().fit(X_train)
        y_train = np.array(features_df['f1_score'])

        reg = LinearRegression().fit(X_train, y_train)
        # reg = IsotonicRegression().fit(X_train, y_train)
        # print(reg.score(X_train, y_train))
        print("finished training")
        test_df = self.read_features(self.test_src, self.test_topic_path, self.test_predictions_path, self.test_queries_dir, self.test_features_path, self.lexicon_src)
        test_df['prior_score'] = test_df.apply(lambda x: x.features['prior_match_score'], axis=1)
        test_df['total_score'] = 0.0
        X_test_df = pd.DataFrame.from_records(test_df['features'])
        if features_to_drop is not None:
            X_test_df = X_test_df.drop(columns=features_to_drop)
        X_test = np.array(X_test_df)
        X_test = scaler.transform(X_test)
        y_pred = reg.predict(X_test)
        #y_pred = scaler.fit_transform(y_pred)

        print("re-ranking test")
        correct_top1_ct = 0
        all_f1s = []
        test_df['pred_score'] = y_pred
        # test_df["total_score"] = test_df["agg"] + test_df["pred_score"] + test_df["prior_score"]
        topk_preds = []
        for qid, group_df in test_df.groupby("qid"):
            ground_answers = questions_dict[qid]["Answers"]
            group_df = group_df.reset_index()
            scaler = MinMaxScaler()
            group_df["pred_score"] = scaler.fit_transform(np.array(group_df['pred_score'].values).reshape(-1, 1)).reshape(1,-1)[0]

            group_df["total_score"] = (0.5 * group_df['pred_score']) + (group_df["agg_score"])

            sorted_records = group_df.sort_values(by=["total_score"], ascending=False).to_dict('records')

            best_row = sorted_records[0]
            best_f1 = float(best_row['f1_score'])
            best_pred_entites = best_row['pred_entities']

            for p in best_pred_entites:
                if p in ground_answers:
                    correct_top1_ct += 1
                    break
            all_f1s.append(best_f1)

            topk_records = sorted_records[:min(topk, len(sorted_records))]
            topk_preds += topk_records

        sum = np.sum(all_f1s)
        total = len(questions_dict)
        print("macro_f1: {}".format(float(sum) * 1.0 / total))

        prec1= float(correct_top1_ct) * 1.0 / float(total)
        print ("precision@1: {}".format(prec1))

        top1_preds_df = pd.DataFrame.from_records(topk_preds)
        if output_path is not None:
            top1_preds_df.to_csv(output_path, index=None)

    def pairwise_rank(self, n_best=1):
        train_df = self.read_features(self.dev_src, self.dev_topic_path, self.dev_predictions_path, self.dev_queries_dir, self.dev_features_path, self.lexicon_src)
        train_df['rank'] = train_df.groupby('qid')['f1_score'].rank(ascending=False, method='first')
        train_df['group_id'] = train_df.groupby('qid').ngroup().add(1)
        sample_size = 5

        '''sort by f1_score'''
        X_train = np.empty((0, len(train_df['features'].iloc[0])))
        y_train = np.empty(0,)
        for qid, group_df in train_df.groupby("qid", as_index=False):
            # print(qid)
            group_df = group_df.sort_values(by=["f1_score"], ascending=False).head(min(len(group_df), sample_size))
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

        test_df = self.read_features(self.test_src, self.test_topic_path, self.test_predictions_path, self.test_queries_dir, self.test_features_path, self.lexicon_src)
        all_f1s = []
        for qid, group_df in test_df.groupby("qid", as_index=False):
            # print(qid)
            features_df = pd.DataFrame(list(group_df['features']))
            X_test = np.array(features_df)
            X_test = scaler.transform(X_test)
            if X_train_group.shape[0] < 2:
                y_test = np.array([0])
            else:
                y_scores = np.dot(X_test, clf.coef_.ravel())
                group_df['re_rank_score'] = y_scores
                group_df['total'] = group_df['re_rank_score'] + group_df['agg_score']
                #print(y_scores)
                group_df_total_scores = np.array(group_df['total'].values)
                y_test = np.argsort(group_df_total_scores)
                #y_test = np.argsort(y_scores)
            best_row_index = np.where(y_test == 0)[0]
            group_data = group_df['f1_score'].values
            best_f1 = group_data[best_row_index]
            all_f1s.append(best_f1)
        macro_avg_f1 = np.mean(all_f1s)
        print(macro_avg_f1)

def evaluate_compq_prec(ques_data_dir, dev_data_dir, test_data_dir, ranking_algorithm="regression", frac=None):
    test_input_path = os.path.join(ques_data_dir, "annotated/test.json")
    if frac:
        test_reranked_path = os.path.join(test_data_dir, "test_top1_reranked_v2_" + str(frac) + ".csv")
    else:
        test_reranked_path = os.path.join(test_data_dir, "test_top1_reranked_v2.csv")

    questions = json.load(codecs.open(test_input_path, 'r', encoding='utf-8'))
    questions_dict = {}
    for q in questions:
        if q["compositionality_type"] == "composition" or q["compositionality_type"] == "conjunction":
            questions_dict[q["ID"]] = q["Answers"]


    correct_count = 0
    total_count = len(questions_dict)
    top1_preds = pd.read_csv(test_reranked_path)
    for index, row in top1_preds.iterrows():
        ground_answers = questions_dict[row['qid']]
        predicted_entities = ast.literal_eval(row['pred_entities'])
        # print("{}, {}".format(ground_answers, predicted_entities))
        for p in predicted_entities:
            # total_count += 1
            if p in ground_answers:
                correct_count += 1
                break
    print ("precision@1: {}".format(float(correct_count) * 1.0 / float(total_count)))


def evaluate_compq_f1(ques_data_dir, dev_data_dir, test_data_dir, ranking_algorithm="regression", features_to_drop=None, topk=1):
    print("{} {}".format(test_data_dir, ranking_algorithm))

    test_input_path = os.path.join(ques_data_dir, "annotated/test.json")
    test_topic_path = os.path.join(ques_data_dir, "topic_entities/main/test_topic.csv")
    dev_input_path = os.path.join(ques_data_dir, "annotated/dev.json")
    dev_topic_path = os.path.join(ques_data_dir, "topic_entities/main/dev_topic.csv")

    dev_queries_dir = os.path.join(dev_data_dir, "queries")
    dev_preds_path = os.path.join(dev_data_dir, "9_prediction_v2.csv")
    dev_features_path = os.path.join(dev_data_dir, "dev_features_v2.csv")

    test_queries_dir = os.path.join(test_data_dir, "queries")
    test_preds_path = os.path.join(test_data_dir, "9_prediction_v2.csv")
    test_features_path = os.path.join(test_data_dir, "test_features_v2.csv")
    test_reranked_path = os.path.join(test_data_dir, "test_top{}_reranked_v2.csv".format(topk))

    queryRanker = QueryRanker(dev_input_path, dev_topic_path, dev_preds_path, dev_queries_dir, dev_features_path,
                              test_input_path, test_topic_path, test_preds_path, test_queries_dir,
                              test_features_path, "lexicon")

    if ranking_algorithm =="regression":
        queryRanker.re_rank_regression(test_reranked_path, features_to_drop, topk)
    elif ranking_algorithm=="logistic":
        queryRanker.re_rank_logistic()
    elif ranking_algorithm=="pairwise":
        queryRanker.pairwise_rank()

    evaluate_compq_prec(ques_data_dir, dev_data_dir, test_data_dir, ranking_algorithm=ranking_algorithm)


if __name__ == '__main__':
    ques_data_dir = "/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess"
    # dev_data_dir = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_dev"
    # ablation experiment configs:
    # full_model
    working_dir = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq"
    # test_data_dir = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5"
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_wo_constraints"
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_wo_attention"
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_wo_prior"
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/a5_dev"
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_downsample/kb_0.9"
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_downsample/kb_0.75"
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_downsample/kb_0.5"
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/emnlp_data/compq/kb_only"

    configs = {
        "dev": {
            "data_dir": "a5_dev",
            "pred_file": "9_prediction_align.csv",
            "queries_dirname": "queries",
            "features_file": "dev_features.csv"
        },
        "align": {
            "data_dir": "a5",
            "pred_file": "9_prediction_align.csv",
            "queries_dirname": "queries",
            "features_file": "test_features.csv"
            },
        "kb_only": {
            "data_dir": "kb_only",
            "pred_file": "5_prediction.csv",
            "queries_dirname": "queries",
            "features_file": "test_features.csv"
            },
        "wo_constraints": {
            "data_dir": "a5_wo_constraints",
            "pred_file": "9_prediction_v2.csv",
            "queries_dirname": "queries",
            "features_file": "test_features.csv"
        },
        "wo_attention": {
            "data_dir": "a5_wo_attention",
            "pred_file": "9_prediction_v2.csv",
            "queries_dirname": "queries",
            "features_file": "test_features.csv"
        },
        "wo_prior": {
            "data_dir": "a5_wo_prior",
            "pred_file": "9_prediction_v2.csv",
            "queries_dirname": "queries",
            "features_file": "test_features.csv"
        },
        "kb_0.9": {
            "data_dir": "kb_downsample/kb_0.9",
            "pred_file": "9_prediction_v2_0.9.csv",
            "queries_dirname": "queries_0.9",
            "features_file": "test_features.csv"
        },
        "kb_0.75": {
            "data_dir": "kb_downsample/kb_0.75",
            "pred_file": "9_prediction_v2_0.75.csv",
            "queries_dirname": "queries_0.75",
            "features_file": "test_features.csv"
        },
        "kb_0.5": {
            "data_dir": "kb_downsample/kb_0.5",
            "pred_file": "9_prediction_v2_0.5.csv",
            "queries_dirname": "queries_0.5",
            "features_file": "test_features.csv"
        },
    }


    test_input_path = os.path.join(ques_data_dir, "annotated/test.json")
    test_topic_path = os.path.join(ques_data_dir, "topic_entities/main/test_topic.csv")
    dev_input_path = os.path.join(ques_data_dir, "annotated/dev.json")
    dev_topic_path = os.path.join(ques_data_dir, "topic_entities/main/dev_topic.csv")

    dev_config = configs["dev"]

    dev_data_dir = os.path.join(working_dir, dev_config["data_dir"])
    dev_queries_dir = os.path.join(dev_data_dir, dev_config["queries_dirname"])
    dev_preds_path = os.path.join(dev_data_dir, dev_config["pred_file"])
    dev_features_path = os.path.join(dev_data_dir, dev_config["features_file"])

    # test_config = configs["kb_only"]
    test_config = configs["align"]
    # test_config = configs["wo_constraints"]
    # test_config = configs["wo_attention"]
    # test_config = configs["wo_prior"]
    # test_config = configs["kb_0.9"]
    # test_config = configs["kb_0.75"]
    # test_config = configs["kb_0.5"]
    topk = 10

    test_data_dir = os.path.join(working_dir, test_config["data_dir"])
    print test_data_dir
    test_queries_dir = os.path.join(test_data_dir, test_config["queries_dirname"])
    test_preds_path = os.path.join(test_data_dir, test_config["pred_file"])
    test_features_path = os.path.join(test_data_dir, test_config["features_file"])
    test_reranked_path = os.path.join(test_data_dir, "test_top{}_reranked.csv".format(topk))


    queryRanker = QueryRanker(dev_input_path, dev_topic_path, dev_preds_path, dev_queries_dir, dev_features_path,
                              test_input_path, test_topic_path, test_preds_path, test_queries_dir,
                              test_features_path, "lexicon")
    queryRanker.re_rank_regression(test_reranked_path, features_to_drop=[ "sub1_score", "sub2_score"], topk=topk)


    # evaluate_compq_f1(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, features_to_drop=[ "sub1_score", "sub2_score"], ranking_algorithm="regression", topk=1)
    '------------'
    # no constraints
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/ablation_wo_cons/CONSFalse_OPTadam_LR0.0005_GA0.5_ATTNTrue500_DO0.0_PRFalse/5"
    # evaluate_compq_f1(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, ranking_algorithm="regression", features_to_drop=["constraint_ct", "type_constraint_ct", "constraints_match_score"])

    # no priors
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/ablation_test_wo_prior/CONSTrue_OPTadam_LR0.0005_GA0.5_ATTNTrue500_DO0.0_PRFalse/5"
    # evaluate_compq_f1(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, ranking_algorithm="regression", features_to_drop=["prior_match_score", "co_occurence_score"])

    # no attention
    # TEST_RUNNING_DIR = "/media/nbhutani/Data/textray_workspace/TextRay/results/wo_attn/CONSTrue_OPTadam_LR0.0005_GA0.5_ATTNFalse500_DO0.0_PRFalse/5"
    # evaluate_compq_f1(DATA_PREFIX, DEV_RUNNING_DIR, TEST_RUNNING_DIR, ranking_algorithm="logistic")

    # evaluate_compq_prec(DATA_PREFIX, TEST_RUNNING_DIR, DEV_RUNNING_DIR, ranking_algorithm="regression")




