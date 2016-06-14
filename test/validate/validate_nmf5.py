'''
Created on Dec 25, 2015

@author: tengmf
'''
import logging
#logging.disable(logging.CRITICAL)

import numpy as np
import os
import os.path as op
import src.util.log as mylog
from pandas import DataFrame;
from src.decomposition.nmf_5 import NMF5
from src.validate.measure import Measure

logger = mylog.getLogger('validate_nmf5','DEBUG')
logger.addHandler(mylog.getLogHandler('INFO', 'CONSOLE'))
logger.addHandler(mylog.getLogHandler('DEBUG', 'FILE'))

ks = [10]#np.arange(10,70,20)
_lambdas = [10 ** i for i in np.arange(-5, -2)]
sigma_as = [1e3]##[10 ** i for i in np.arange(-3, 3)]
sigma_bs = [1e3]#[10 ** i for i in np.arange(-3, 3)]
etas = [10]#[10 ** i for i in np.arange(-3, 3)]

thetas = [0.001]##[10 ** i for i in np.arange(-3, 3)]

total = len(ks)*len(_lambdas)*len(sigma_as)*len(sigma_bs)*len(etas)*len(thetas)

base_path = op.join("..", "..", "output")

for data in ["data2"]:#, "data3", "data"]:
    V = np.loadtxt(op.join(base_path, data, "validate", "training", "pure_matrix.csv"), delimiter=",")
    C = np.loadtxt(op.join(base_path, data, "validate", "training", "company_matrix"))
    R_test = np.loadtxt(op.join(base_path, data, "validate", "test", "pure_matrix.csv"), delimiter=",")
    
    result = DataFrame(columns=["Data Set", "lambda", "sigma A", "sigma B", "eta", "theta", "k"] + [i + str(j) for i in ["NDCG@", "Precision@", "Recall@"] for j in [3, 5, 10]])
    output_path = op.join(base_path, data, "validate", "nmf5")
    if not op.exists(output_path):
        os.makedirs(output_path)
        
    idx = 0;
    for k in ks:
        WInit = np.random.uniform(8, 20, (V.shape[0], k))
        HInit = np.random.uniform(8, 20, (k, V.shape[1]))

        for _lambda in _lambdas:
            for sigma_a in sigma_as:
                for sigma_b in sigma_bs:
                    for eta in etas:
                        for theta in thetas:

                            

                            nmf = NMF5()
                            W, H = nmf.factorize(V, C, k, _lambda, sigma_a, sigma_b, eta, theta, WInit=WInit, HInit=HInit)
                            WH = W.dot(H)                            
                            m = Measure(WH, R_test)
                            precision_recall = list(zip(*[m.precision_recall(i) for i in [3, 5, 10]]))
                            result.loc[idx] = [data, _lambda, sigma_a, sigma_b, eta, theta, k, m.ndcg(3), m.ndcg(5), m.ndcg(10)] + list(precision_recall[0]) + list(precision_recall[1])
                            idx += 1
                            msg = "###### progress %f on %s #####"%(100.*idx/total, data)
                            logger.info(msg)
                            #print(msg)
                            del W,H,nmf,m
                            #del WInit,HInit

        #del WInit,HInit
    result.to_csv(op.join(output_path, "result.csv"), index=False)                      
                # Measure m = Meas
