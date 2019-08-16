import argparse
import pandas as pd
import ast


def filter(row, threshold):
    return [t for t in row.triples if t["confidence"] > threshold]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Triple Candidates')
    parser.add_argument('-I', '--input',
                        type=str,
                        help='Input triples file')
    parser.add_argument('-O', '--output',
                        type=str,
                        help='Output triples file')
    parser.add_argument('-T', '--threshold',
                        type=int)
    args = parser.parse_args()
    triples = pd.read_csv(args.input)
    triples['triples'] = triples['triples'].apply(lambda triples: ast.literal_eval(triples))
    triples['filtered_triples'] = triples.apply(lambda x: filter(x, args.threshold), axis = 1)

    triples = triples[["id", "sentence", "filtered_triples"]]
    triples.to_csv(args.output, index=None)
