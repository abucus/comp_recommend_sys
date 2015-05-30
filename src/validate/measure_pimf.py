'''
Created on May 25, 2015

@author: tengmf
'''
import numpy as np

def precision_recall(retrived, relevant):
    retrived = retrived
    relevant = relevant
    intersect_num = len(np.intersect1d(retrived, relevant))
    #print retrived, relevant
    return (1.*intersect_num / len(retrived), 1.*intersect_num / len(relevant))

def normalize_scores(scores):
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + np.spacing(0))
    
def ndcg(scores):
    scores = normalize_scores(scores)
    length = scores.shape[0]
    #print length
    if length == 1:
        return 0
    else:
        iscores = np.sort(scores)[::-1]
        dcg = scores[0]
        idcg = iscores[0]
        for i in xrange(1, length):
            #print dcg, scores[i] / np.log2(i + 1)
            dcg += scores[i] / np.log2(i + 1)
            idcg += iscores[i] / np.log2(i + 1)
        return dcg / (idcg + np.spacing(0))
    
if __name__ == "__main__":
    print ndcg(np.array([3, 2, 3, 0, 1, 2]))
