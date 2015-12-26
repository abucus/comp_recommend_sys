'''
Created on Dec 25, 2015

@author: tengmf
'''
import numpy as np
import os
import os.path as op
from pandas import DataFrame;
from src.decomposition.nmf_4 import NMF4
from src.validate.measure import Measure

ks = np.arange(10, 70, 10);
_lambdas = [10 ** i for i in np.arange(-4, 4)]
lambda_as = [10 ** i for i in np.arange(-4, 4)]
lambda_bs = [10 ** i for i in np.arange(-4, 4)]
base_path = op.join("..", "..", "output")

for data in ["data", "data2"]:    
    V = np.loadtxt(op.join(base_path, data, "validate", "training", "pure_matrix.csv"), delimiter=",")
    C = np.loadtxt(op.join(base_path, data, "validate", "training", "company_matrix"))
    R_test = np.loadtxt(op.join(base_path, data, "validate", "test", "pure_matrix.csv"), delimiter=",")
    
    result = DataFrame(columns=["Data Set", "lambda", "lambda A", "lambda B", "k"] + [i + str(j) for i in ["NDCG@", "Precision@", "Recall@"] for j in [3, 5, 10]])
    output_path = op.join(base_path, data, "validate", "nmf4")
    if not op.exists(output_path):
        os.makedirs(output_path)
        
    idx = 0;
    for k in ks:
        for _lambda in _lambdas:
            for lambda_a in lambda_as:
                for lambda_b in lambda_bs:
                    
                    nmf = NMF4()
                    
                    WInit = np.random.uniform(1, 2, (V.shape[0], k))
                    HInit = np.random.uniform(1, 2, (k, V.shape[1]))
                    W, H = nmf.factorize(V, C, k, _lambda, lambda_a, lambda_b, WInit=WInit, HInit=HInit)
                    WH = W.dot(H)

                    np.savetxt(op.join(output_path, "WH"+str(idx)+".txt"), WH)
                    
                    m = Measure(WH, R_test)
                    precision_recall = zip(*[m.precision_recall(i) for i in [3, 5, 10]])
                    result.loc[idx] = [data, _lambda, lambda_a, lambda_b, k, m.ndcg(3), m.ndcg(5), m.ndcg(10)] + list(precision_recall[0]) + list(precision_recall[1])
                    
                    idx += 1
                    del WInit,HInit, W,H,nmf,m
                    
    result.to_csv(op.join(output_path, "result.csv"), index=False)
                
                # Measure m = Meas
