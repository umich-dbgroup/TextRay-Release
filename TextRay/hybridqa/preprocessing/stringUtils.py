import string
import nltk
import fuzzywuzzy.process as fuzz_process
from metricUtils import jaccard_ch


def match_entities(text, names, encoding='utf-8'):
    '''
    :param text: text to match entities
    :param names: list of names and alias
    :return: matched seq, if no such seq, return None
    '''
    text_tokens = nltk.word_tokenize(text.lower())
    entity_token_count = {}
    for name in names:
        name_tokens = nltk.word_tokenize(name.lower())
        for token in name_tokens:
            if token.lower() not in entity_token_count.keys():
                entity_token_count[token.lower()] = 1
            else:
                entity_token_count[token.lower()] += 1

    sorted_entity_tokens = sorted(entity_token_count.keys(), key=lambda x:entity_token_count[x], reverse=True)
    # find the most frequent word that matches:
    text_idx_start = 0
    text_idx_end = 0
    entity_idx = 0
    while entity_idx < len(sorted_entity_tokens):
        if sorted_entity_tokens[entity_idx] in text_tokens:
            text_idx_start = text_idx_end = text_tokens.index(sorted_entity_tokens[entity_idx])
            break
        entity_idx += 1

    # no match
    if entity_idx == len(sorted_entity_tokens):
        return None

    # search right in original text
    while text_idx_end <= len(text_tokens) - 2:
        if text_tokens[text_idx_end + 1] in set(sorted_entity_tokens):
            text_idx_end += 1
        else:
            break

    # search left in original text
    while text_idx_start >= 1:
        if text_tokens[text_idx_start - 1] in set(sorted_entity_tokens):
            text_idx_start -= 1
        else:
            break

    matched_seq = ' '.join(text_tokens[text_idx_start:text_idx_end + 1])
    return matched_seq


def fuzzy_match_entities(text, names,encoding='utf-8'):
    tokens = nltk.word_tokenize(text.lower())
    candidate_sets = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            candidate = ' '.join(tokens[i:j])
            candidate_sets.append(candidate)

    matched_seq2score = {}
    for name in names:
        matched = fuzz_process.extractOne(name.lower(), candidate_sets, score_cutoff=70)
        if matched:
            matched_seq2score[matched[0]] = matched[1]
    if len(matched_seq2score) == 0:
        return None
    #print matched_seq2score
    sorted_seq = sorted(matched_seq2score.keys(), reverse=True, cmp=lambda a, b: 1 if matched_seq2score[a] > matched_seq2score[b] else \
                                                                -1 if matched_seq2score[a] < matched_seq2score[b] else \
                                                                1 if len(a) > len(b) else\
                                                                -1 if len(a) < len(b) else 0)
    return sorted_seq[0]


def sim_match_entities(text, names, metric='jaccard', encoding='utf-8', threshold=0.4):
    tokens = nltk.word_tokenize(text.lower())
    candidate_sets = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            candidate = ' '.join(tokens[i:j])
            candidate_sets.append(candidate)
    print candidate_sets
    max_score = 0
    max_seq = ""
    for name in names:
        for candidate in candidate_sets:
            curr_score = jaccard_ch(candidate, name)
            if curr_score > threshold:
                if max_score < curr_score:
                    max_seq = candidate
                    max_score = curr_score
                elif max_score == curr_score and len(candidate) > len(max_seq):
                    max_seq = candidate
    if max_score == 0:
        return None, 0
    #print max_seq, max_score
    return max_seq, max_score

def match_entities(text, names, metric='jaccard', encoding='utf-8', threshold=0.4):
    tokens = nltk.word_tokenize(text.lower())
    candidate_sets = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            candidate = ' '.join(tokens[i:j])
            candidate_sets.append((candidate, i, j))
    max_score = 0
    max_seq = None
    for name in names:
        for candidate in candidate_sets:
            curr_score = jaccard_ch(candidate[0].lower(), name.lower())
            if curr_score > threshold:
                if max_score < curr_score:
                    max_seq = candidate
                    max_score = curr_score
                elif max_score == curr_score and len(candidate) > len(max_seq):
                    max_seq = candidate
    if max_score == 0:
        return None, 0
    return max_seq, max_score

def match_names_to_mention(text, names, metric='jaccard', encoding='utf-8', threshold=0.4):
    spans = get_spans(text)
    candidate_sets = []
    for i in range(len(spans)):
        for j in range(i, len(spans)):
            span_start = spans[i][1]
            span_end = spans[j][1] + spans[j][2]
            candidate_name = text[span_start: span_end]
            candidate_sets.append((candidate_name, span_start, span_end))
    max_score = 0
    max_seq = None
    for name in names:
        for candidate in candidate_sets:
            curr_score = jaccard_ch(candidate[0].lower(), name.lower())
            if curr_score > threshold:
                if max_score < curr_score:
                    max_seq = candidate
                    max_score = curr_score
                elif max_score == curr_score and len(candidate) > len(max_seq):
                    max_seq = candidate
    if max_score == 0:
        return None
    return max_seq

def get_spans(text):
    tokens = nltk.word_tokenize(text)
    offset = 0
    spans = []
    for token in tokens:
        offset = text.find(token, offset)
        spans.append((token, offset, len(token)))
        offset += len(token)
    return spans


if __name__ == '__main__':
    #find_mentions('Who is Barack Obama?'.lower(), ['Barack'.lower(), 'Obama'.lower(), 'B. Obama'.lower()])
    entities = match_entities('What is the mascot of the school where Thomas R. Ford is a grad student?', [u'Thomas R. Ford', u'Thomas Robert Ford'])
    #seq = sim_match_entities('Who is Barack Obama?'.lower(), ['Barack'.lower(), 'Obama'.lower(), 'B. Obama'.lower()])
    print entities

