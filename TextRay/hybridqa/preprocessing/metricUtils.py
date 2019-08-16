
def jaccard_ch(a, b, gram_min=2, gram_max=4):
    a_sign = set()
    b_sign = set()
    gram_num = gram_min
    try:
        while gram_num <= gram_max:
            for i in range(len(a) - gram_num + 1):
                a_sign.add(a[i:i + gram_num])
            for i in range(len(b) - gram_num + 1):
                b_sign.add(b[i:i + gram_num])
            gram_num += 1
        inter = a_sign.intersection(b_sign)
        un = a_sign | b_sign
        return float(len(inter)) / float(len(un))
    except Exception:
        return 0

"""return a tuple with recall, precision, and f1 for one example"""
def compute_f1(goldList, predictedList):
    if len(goldList)==0:
        raise Exception("gold list may not be empty")
    if len(predictedList)==0: #If we return an empty list recall is zero and precision is one
        return (0,1,0)
    precision = 0
    for entity in predictedList:
        if entity in goldList:
            precision += 1
    precision = float(precision) / len(predictedList)

    recall=0
    for entity in goldList:
        if entity in predictedList:
            recall+=1
    recall = float(recall) / len(goldList)

    f1 = 0
    if precision+recall > 0:
        f1 = 2 * recall * precision / (precision + recall)

    return recall, precision, f1

"""return a tuple with recall, precision, and f1 for one example"""
def compute_micro_precision_recall(goldList, predictedList):
    if len(goldList)==0:
        return 1.0, 0.0
    if len(predictedList)==0: #If we return an empty list recall is zero and precision is one
        return 0.0, 0.0
    intersect = set(predictedList).intersection(goldList)
    if len(intersect) > 0:
        return 1.0, 1.0
    return 0.0, 0.0