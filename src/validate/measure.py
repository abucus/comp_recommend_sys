'''
Created on Feb 19, 2015

@author: tengmf
'''
import numpy as np
import os.path as op
from numpy.linalg import norm
import json
    
    
class Measure:
    def __init__(self, R_hat, R_test):
        self.R_hat = R_hat 
        self.R_test = R_test
        self.I = np.where(R_test > 0, 1, 0)
        self.non_zero_count = self.I.sum()
        self.diff = (R_hat - R_test) * self.I
        
        self.I_non_zero_row_idx = []
        for i in range(self.I.shape[0]):
            if np.all(self.I[i] == 0):
                continue
            self.I_non_zero_row_idx.append(i)

        
    def reset(self, R_hat):
        self.R_hat = R_hat
        self.diff = (R_hat - self.R_test) * self.I
        
    def mae(self):
        return norm(self.diff, 1) / self.non_zero_count
    
    def rmse(self):
        return np.sqrt(norm(self.diff) ** 2 / self.non_zero_count)
    
    def precision_recall(self, k):
        self.precision_recall_k = k
        unit_len = int(np.sqrt(self.R_test.shape[1]))
        count = 0
        total_precision = total_recall = 0
        for i in self.I_non_zero_row_idx:
            # print "cal precision recall row",i," time begin:",datetime.datetime.now()
            for j in range(0, self.R_test.shape[1] - unit_len + 1, unit_len):
                tmp_rlt = self.__cal_precision_recall(self.R_hat[i, j:(j + unit_len)],
                                                      self.R_test[i, j:(j + unit_len)])
                if tmp_rlt:
                    total_precision += tmp_rlt[0]
                    total_recall += tmp_rlt[1]
                    count += 1
        return [total_precision / count, total_recall / count]
    
    def __cal_precision_recall(self, train_score, true_score):
        # print "in __cal_precision_recall",train_score,true_score
        if np.all(true_score == 0):
            return None
        train_score_sorted_idx = np.argsort(train_score)[::-1]
        true_score_sorted_idx = np.where(true_score > 0)[0]
        k = min(self.precision_recall_k, len(train_score_sorted_idx))
        
        intersect_num = len(np.intersect1d(train_score_sorted_idx[0:k], true_score_sorted_idx, assume_unique=True))
        # print intersect_num
        return (intersect_num * 1. / k, intersect_num * 1. / len(true_score_sorted_idx))
        
    def ndcg(self, k):
        self.ndcg_k = k
        unit_len = int(np.sqrt(self.R_test.shape[1]))
        count = 0
        total = 0
        for i in self.I_non_zero_row_idx:
            # print "cal ndcg row",i," time begin:",datetime.datetime.now()
            for j in range(0, self.R_test.shape[1] - unit_len + 1, unit_len):
                tmp_rlt = self.__cal_ndcg(self.R_hat[i, j:(j + unit_len)],
                                          self.R_test[i, j:(j + unit_len)])
                if tmp_rlt:
                    total += tmp_rlt
                    count += 1
        return total / count
                
    def __cal_ndcg(self, train_score, true_score):
        if np.all(true_score == 0):
            return None
        
        pack0 = list(zip(train_score, true_score))
        pack = [x for x in pack0 if x[1] > 0]
        if len(pack) < self.ndcg_k:
            return None
        
        pack.sort(key=lambda l:l[0], reverse=True)
        
        score = self.__normalize(np.array([x[1] for x in pack]))
        
        dcg = self.__cal_dcg(score) 
        idcg = self.__cal_dcg(sorted(score, reverse=True))
        
        return dcg / (idcg + np.spacing(0))
            
    def __cal_dcg(self, score):
        result = score[0]
        for i in range(0, min(self.ndcg_k, len(score))):
            result += (2.**(score[i]) - 1) / np.log2(i + 2)
        return result
    
    def __normalize(self, scores):
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + np.spacing(0))

def flattern_fpmc_matrix(a):
    user_num, item_num = a.shape[0], a.shape[1]
    m = np.zeros((user_num, item_num ** 2))
    for u in range(user_num):
        for i in range(item_num):
            m[u, i * item_num:(i + 1) * item_num] = a[u, i]
    return m

if __name__ == "__main__":
    R_hat = np.loadtxt(op.join("..", "..", "output", "data2", "validate", "nmf3", "129", "WH.txt"))
    R_test = np.loadtxt(op.join("..", "..", "output", "data2", "validate", "test", "pure_matrix.csv"), delimiter=',')
    m = Measure(R_hat, R_test)
    print('ndcg@{}:{},{}:{},{}:{}'.format(3,m.ndcg(3),5,m.ndcg(5),10,m.ndcg(10)))
#     a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
#     print a.shape
#     print flattern_fpmc_matrix(a)
# m = Measure(None,None)
# print m.__cal_ndcg(array([2,3,1,4]), array([5,6,8,3]))
# print m.__cal_dcg(array([3,6,5,8]))/m.__cal_dcg(array([8,6,5,3]))
# R_hat = np.array([[1, 2,  3, 13,  9, 13, 28,  4, 14], [4,  6,  8, 17, 16, 13, 16, 11, 10]])
# R_test = np.array([[16, 17, 24, 24, 28,  7, 23, 18,  7], [10,  6, 20, 29,  3,  6,  5,  8, 10]])
# m = Measure(R_hat, R_test)
# print m.precision_recall(2)
