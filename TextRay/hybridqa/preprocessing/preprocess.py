import argparse
import pandas as pd
import numpy as np
import re
import ast
import codecs
import json
from stringUtils import sim_match_entities


def annotate_split_part1(row, mids_df):
    return annotate(row.ID, row.ID + ":" + "split_part1", row.split_part1_ques, mids_df)


def annotate_split_part2(row, mids_df):
    return annotate(row.ID, row.ID + ":" + "split_part2", row.split_part2_ques, mids_df)


def annotate_ques(row, mids_df):
    return match_entities(mids_df, row.ID, row.question)


def annotate(ques_id, sub_ques_id, sub_ques, mids_df):
    if sub_ques is None:
        return {"sub_ID": sub_ques_id}
    sub_ques = str(sub_ques)
    matched = match_entities(mids_df, ques_id, sub_ques)
    return {"sub_ID": sub_ques_id, "sub_ques": sub_ques, "entities": matched}


def match_entities(mids_df, ques_id, ques_str):
    mid_df_sub = mids_df[mids_df['ID'] == ques_id]
    mid_df_sub = mid_df_sub.drop_duplicates('mid')  # keep only unique mids
    matched = []
    for index, mid_row in mid_df_sub.iterrows():
        mid = mid_row.mid
        if mid is None:
            continue
        has_exact_match = False
        for name in mid_row.names:
            m = re.search(re.escape(name), ques_str, re.IGNORECASE)
            if m:
                matched.append({"mid": mid, "mention": name})
                has_exact_match = True
                break  # only one name match for each mid
        if not has_exact_match:
            matched_seq, score = sim_match_entities(ques_str, mid_row.names)
            if matched_seq:
                matched.append({"mid": mid, "mention": matched_seq})
    return matched


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='joining dataset')
    parser.add_argument('-I', '--input',
                        type=str,
                        help='Input')
    parser.add_argument('-T', '--split_input',
                        type=str,
                        help='S plit Input')
    parser.add_argument('-M', '--mids',
                        type=str,
                        help='entity ids')
    parser.add_argument('-O', '--output',
                        type=str,
                        help='Output')
    parser.add_argument('-S', '--name',
                        type=str,
                        help='Type of split')

    args = parser.parse_args()

    questions = pd.read_json((args.input))
    split_points = pd.read_json(args.split_input)
    mids = pd.read_csv(args.mids, encoding='utf-8')
    mids['names'] = mids['names'].fillna('[]')
    mids['names'] = mids['names'].apply(lambda x: str(x).replace('][', ','))
    mids['names'] = mids['names'].apply(ast.literal_eval)

    print('finished reading mids')
    joined = questions.merge(split_points, how='outer', on='ID')
    joined = joined.rename(columns={'split_part1': 'split_part1_ques', 'split_part2': 'split_part2_ques', 'question_x': 'question'})

    joined['entities'] = joined.apply(annotate_ques, args=(mids,), axis=1)

    print('finished joining question entities')
    joined['split_part1'] = joined.apply(annotate_split_part1, args=(mids,), axis=1)
    print('finished finding entities for sub question 1')
    joined['split_part2'] = joined.apply(annotate_split_part2, args=(mids,), axis=1)
    print('finished finding entities for sub question 2')

    if 'answers' in joined.columns:
        joined = joined[['ID', 'compositionality_type', 'machine_question', 'question', 'sparql', 'webqsp_ID', 'webqsp_question', 'comp', 'split_part1', 'split_part2', 'entities', 'answers']]
    else:
        joined = joined[
            ['ID', 'compositionality_type', 'machine_question', 'question', 'sparql', 'webqsp_ID', 'webqsp_question','comp', 'split_part1', 'split_part2', 'entities']]
    with open(args.output, 'w') as f:
        json.dump(joined.to_dict(orient='records'), f, indent=4)