import numpy as np

def precision_k(score_label, k):
    p, i = 0, 0
    for s in score_label:
        if i < k:
            if s[1]>3:
                p += 1
            i += 1
    return p/k

def dcg_k(score_label, k):
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            dcg += (2**s[1]-1) / np.log2(2+i)
            i += 1
    return dcg

def ndcg_k(score_label, k):
    score_label_ = sorted(score_label, key=lambda d:d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            norm += (2**s[1]-1) / np.log2(2+i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm

def auc(score_label):
    fp1, tp1, fp2, tp2, auc = 0.0, 0.0, 0.0, 0.0, 0.0
    for s in score_label:
        fp2 += (1-s[1]) # noclick
        tp2 += s[1] # click
        auc += (tp2 - tp1) * (fp2 + fp1) / 2
        fp1, tp1 = fp2, tp2
    try:
        return 1- auc / (tp2 * fp2)
    except:
        return 0.5

def mae(score_label):
    n = 0
    error = 0
    for s in score_label:
        error += abs(s[1] - s[0])
        n += 1
    return error / n

def mrse(score_label):
    n = 0
    error = 0
    for s in score_label:
        error += (s[1] - s[0]) ** 2
        n += 1
    return np.sqrt(error/n)
