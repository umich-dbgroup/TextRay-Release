import os
import pandas as pd
import ast
import json
import codecs

def read_results(src):
    if not os.path.exists(src):
        print src
        return None
    print src
    df = pd.read_csv(src)
    df[['sub2_relation', 'sub2_constraints']] = df[['sub2_relation', 'sub2_constraints']].fillna(value='[]')
    df['sub1_relation'] = df['sub1_relation'].apply(lambda x: ast.literal_eval(x))
    df['sub2_relation'] = df['sub2_relation'].apply(lambda x: ast.literal_eval(x))
    df['sub1_constraints'] = df['sub1_constraints'].apply(lambda x: ast.literal_eval(x))
    df['sub2_constraints'] = df['sub2_constraints'].apply(lambda x: ast.literal_eval(x))
    df['f1_score'] = df['f1_score'].apply(lambda x: float(x))
    df['agg_score'] = df['agg_score'].apply(lambda x: float(x))
    print df.shape
    return df

def analyze_incorrect(src1, src2, n_best=1):
    df1 = read_results(src1)
    df2 = read_results(src2)
    for qid, group in df1.groupby("qid"):
        group_df1_sorted = group.sort_values(by=["agg_score"], ascending=False)
        group_df1_sorted_sub = json.loads(group_df1_sorted.head(min(len(group_df1_sorted), n_best)).to_json(orient='records'))

        group_df_2 = df2[df2["qid"] == qid]
        group_df2_sorted = group_df_2.sort_values(by=["agg_score"], ascending=False)
        group_df2_sorted_sub = json.loads(group_df2_sorted.head(min(len(group_df2_sorted), n_best)).to_json(orient='records'))

        for i in range(n_best):
            row1 = group_df1_sorted_sub[i]
            row2 = group_df2_sorted_sub[i]
            # sub1_key = '{}_{}_{}_{}'.format(row1['sub1_relation'], row1['sub1_constraints'], row1['sub2_relation'], row1['sub1_constraints'])
            # sub2_key = '{}_{}_{}_{}'.format(row2['sub1_relation'], row2['sub1_constraints'], row2['sub2_relation'],
            #                                 row2['sub1_constraints'])
            sub1_key = '{}_{}'.format(row1['sub1_relation'], row1['sub1_constraints'])
            sub2_key = '{}_{}'.format(row2['sub1_relation'], row2['sub1_constraints'])
            if sub1_key != sub2_key and row1['f1_score'] > row2['f1_score']:

                print qid
                print row1
                print row2
                print
                print

def analyze_incorrect_train(src1, sub1_flat_file_path, cand_dir):
    files = os.listdir(src1)
    files = [f for f in files if f.startswith('sub1_pred')]
    test_index1 = pd.read_csv(sub1_flat_file_path, sep=',')
    test_index1['index'] = test_index1['index'].astype(int)

    for f in files:
        qid = f.replace("sub1_pred_", "").replace(".json", "")
        path_dict = __get_sub1_paths__(cand_dir, qid)

        preds_1 = json.load(open(os.path.join(src1, f)))
        if len(preds_1) > 0:
            pred1 = preds_1[0]
            topic1, rel1_data = __get_sub1_paths_data__(pred1["index"], path_dict, test_index1)
            if rel1_data["derived_label"] == 0:
                print f
                print rel1_data
                print pred1
                print
                print



def analyze_incorrect_sub1(src1, src2, sub1_flat_file_path, sub2_flat_file_path, cand_dir):
    files = os.listdir(src1)
    files = [f for f in files if f.startswith('sub1_pred')]
    test_index1 = pd.read_csv(sub1_flat_file_path, sep=',')
    test_index1['index'] = test_index1['index'].astype(int)


    test_index2 = pd.read_csv(sub2_flat_file_path, sep=',')
    test_index2['index'] = test_index2['index'].astype(int)

    for f in files:
        qid = f.replace("sub1_pred_", "").replace(".json", "")
        path_dict = __get_sub1_paths__(cand_dir, qid)

        preds_1 = json.load(open(os.path.join(src1, f)))
        preds_2 = json.load(open(os.path.join(src2, f)))

        if len(preds_1) > 0 and len(preds_2) > 0:
            pred1 = preds_1[0]
            pred2 = preds_2[0]
            topic1, rel1_data = __get_sub1_paths_data__(pred1["index"], path_dict, test_index1)
            topic2, rel2_data = __get_sub1_paths_data__(pred2["index"], path_dict, test_index2)

            sub1_key = '{}_{}'.format(rel1_data['relations'], rel1_data['constraints'])
            sub2_key = '{}_{}'.format(rel2_data['relations'], rel2_data['constraints'])

            if sub1_key != sub2_key:
                print f
                print rel1_data
                print rel2_data
                print
                print

def __get_sub1_paths__(sub1_cands_dir, qid):
    # print('reading query graph 1 for ' + qid)
    path = os.path.join(sub1_cands_dir, qid + ".json") # read sub1 json
    path_dict = {}
    if not os.path.exists(path):
        print('path does not exist for qid ' + qid)
        return path_dict
    sub1_paths = json.load(codecs.open(path, 'r', encoding='utf-8'))
    for topic in sub1_paths.keys():
        for path in sub1_paths[topic]:
            path["topic"] = topic
            path["is_openie"] = False
            key = __get_path_key__(topic, path)
            path_dict[key] = path

    return path_dict

def __get_sub1_paths_data__(cand_index, path_dict, test_index):
    # print 'Cand Index: {}'.format(cand_index)
    #rel_data_keys = self.test_index[self.test_index["index"] == cand_index].to_records(index=False)
    rel_data_keys = test_index[test_index["index"] == cand_index].to_dict('records')
    if len(rel_data_keys) == 0:
        print "Key not found"
        return None, None
    rel_data_key = rel_data_keys[0]
    topic = rel_data_key["topic"]
    if "is_openie" in rel_data_key and rel_data_key["is_openie"]:
        lookup_key = topic + "_" + str(rel_data_key["openie"]) + "_" + str(())
    else:
        look_up_key = topic + "_" + str(rel_data_key["relations"]) + "_" + str(rel_data_key.get("constraints", ()))
    # if look_up_key not in path_dict:
    #     print look_up_key
    #     print path_dict
    rel_data = path_dict.get(look_up_key, None)
    return topic, rel_data


def __get_path_key__(topic, path):
    rels = tuple([p for p in path['relations']])
    constraints = []
    if 'constraints' in path:
        constraints = [constraint['relation'] for constraint in path['constraints']]
    constraints = tuple(constraints)
    key = topic + "_" + str(rels) + "_" + str(constraints)
    return key



if __name__ == '__main__':
    # src1 = '/media/nbhutani/Data/textray_workspace/model_predictions/0428_kbonly_predictions/E_14/14_prediction.csv'
    # src2 ='/media/nbhutani/Data/textray_workspace/vldb_data/full_model_test/CONSTrue_OPTadam_LR0.0005_GA0.5_ATTNTrue500_DO0.0_PRFalse/5/5_prediction.csv'
    #
    # analyze_incorrect(src1, src2)

    # src1 = '/media/nbhutani/Data/textray_workspace/model_predictions/debug_0429_kbonly_predictions/CONSTrue_ATTNTrue_KBTrue_OIEFalse_LR0.0002_LRG0.5_ADO0.0_LDO0.0_APOOLFalse_PTHR2/5'
    # src2 = '/media/nbhutani/Data/textray_workspace/vldb_data/full_model_test/CONSTrue_OPTadam_LR0.0005_GA0.5_ATTNTrue500_DO0.0_PRFalse/5'
    #
    # look_up1 = os.path.join(src1, 'sub1_lookup.csv')
    # look_up2 = os.path.join(src2, 'sub1_lookup.csv')
    #
    # cands_dir = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess/rewards/test/sub1'
    #
    # analyze_incorrect_sub1(src1, src2, look_up1, look_up2, cands_dir)


    train_src = "/media/nbhutani/Data/textray_workspace/model_predictions/debug_0429_kbonly_predictions/train_pred/0429_kbonly_train_test/CONSTrue_ATTNTrue_KBTrue_OIEFalse_LR0.0002_LRG0.5_ADO0.0_LDO0.0_APOOLFalse_PTHR2/5"
    train_look_up = os.path.join(train_src, 'sub1_lookup.csv')
    cands_dir = '/media/nbhutani/Data/textray_workspace/TextRay/datasets/ComplexWebQuestions_preprocess/rewards/rescaled_max_priors_derived_0.5/train'

    analyze_incorrect_train(train_src, train_look_up, cands_dir)