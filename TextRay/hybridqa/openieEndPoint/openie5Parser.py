import pandas as pd
import argparse
import os
import io
from nltk.tokenize import sent_tokenize


def snippets_to_text(input, output_dir, name):
    df = pd.read_json(input, encoding="utf-8")
    df['line_id'] = range(1, len(df.index)+1)
    sids = io.open(os.path.join(output_dir,"sid-{}.txt".format(name)), "w")
    snippet_texts = io.open(os.path.join(output_dir,"snippet-{}.txt".format(name)), "w")
    snippet_ques_ids = []
    snippet_ids = []
    snippet_text = []
    for index, row in df.iterrows():
        if "ptrnet split" not in row.split_source:
            continue
        unique_questionId = str(row.question_ID) + ":" + str(row.split_type)
        for snippet_idx, snippet in enumerate(row.web_snippets):
            snippet_ques_ids.append(unique_questionId)
            snippet_ids.append(snippet_idx)
            snippet_text.append(snippet["snippet"])
    snippet_df = pd.DataFrame(data={'snippet_ques_id': snippet_ques_ids,'snippet_id': snippet_ids, 'text': snippet_text})
    snippet_df = snippet_df.sort_values(by=['snippet_id','snippet_ques_id'])
    for index, row in snippet_df.iterrows():
        sids.write((row.snippet_ques_id + ":" + str(row.snippet_id) + "\n").encode('utf-8').decode('utf-8'))
        snippet_texts.write((row.text + "\n").encode('utf-8').decode('utf-8'))
    sids.close()
    snippet_texts.close()


def snippets_to_sentences(output_dir, name):
    sid_filepath = os.path.join(output_dir,"sid-{}.txt".format(name))
    snippet_filepath = os.path.join(output_dir,"snippet-{}.txt".format(name))

    sids = pd.DataFrame([line.rstrip('\n') for line in open(sid_filepath)], columns=['sid'])
    snippets_data = []
    for line in io.open(snippet_filepath, encoding="utf-8", mode="r"):
        snippets_data.append(line.rstrip('\n'))
    snippets = pd.DataFrame(snippets_data, columns=['snippet'])
    df = sids.join(snippets)
    df['sentences'] = df['snippet'].apply(sent_tokenize)

    sid_out = io.open(os.path.join(output_dir,"sid-sentence-{}.txt".format(name)), "w", encoding="utf-8")
    snippet_out = io.open(os.path.join(output_dir,"snippet-sentence-{}.txt".format(name)), "w", encoding="utf-8")

    for index, row in df.iterrows():
        snippet_id = row.sid
        for sentence_index, sentence in enumerate(row.sentences):
            sentence_id = str(snippet_id) + ":" + str(sentence_index)
            sid_out.write((sentence_id + "\n").encode('utf-8').decode('utf-8'))
            snippet_out.write((sentence + "\n").encode('utf-8').decode('utf-8'))
    sid_out.close()
    snippet_out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Triple Coverage script')
    parser.add_argument('-I', '--input',
                        type=str,
                        help='Input file')
    parser.add_argument('-O', '--output_dir',
                        type=str,
                        help='Output dir file')
    parser.add_argument('-S', '--name',
                        type=str,
                        help='Type of split')
    args = parser.parse_args()

    snippets_to_text(args.input, args.output_dir, args.name)
    snippets_to_sentences(args.output_dir, args.name)




