from src.decomposition.nmf_3_2 import NMF3
from src.decomposition.nmf_4 import NMF4
import os.path as op
import numpy as np
import logging

logging.basicConfig(filename='time2.log',level=logging.DEBUG)


def time_cost(f):
    def wrap(*args):
        import time
        start = time.time()
        f(*args)
        end = time.time()
        logging.info ("time elapsed %r"%(end-start))
    return wrap

def loadData(data):
    base_path = op.join("..", "..", "output", data, "validate", "training")
    V = np.loadtxt(op.join(base_path, "pure_matrix.csv"), delimiter=",")
    C = np.loadtxt(op.join(base_path, "company_matrix"))
    return (V,C)

@time_cost
def testNMF3(V, C, W_init, H_init, args):
    nmf = NMF3()
    nmf.factorize(V, C, WInit=W_init, HInit=H_init, _lambda = args['_lambda'])

@time_cost
def testNMF4(V, C, W_init, H_init, args):
    nmf = NMF4()
    nmf.factorize(V, C, _lambda=args['_lambda'], lambda_a=args['lambda_a'], lambda_b=args['lambda_b'], WInit=W_init, HInit=H_init)


datas = ["data", "data2", "data3"]
parameters = [{'k':10, '_lambda':10, 'lambda_a':1, 'lambda_b':10},\
{'k':50, '_lambda':1, 'lambda_a':1, 'lambda_b':1},\
{'k':10, '_lambda':0.0001, 'lambda_a':0.001, 'lambda_b':0.1}]

for i in [0]:
    data = datas[i]
    parameter = parameters[i]
    (V,C) = loadData(data)
    W_init = np.random.uniform(1, 2, (V.shape[0], parameter['k']))
    H_init = np.random.uniform(1, 2, (parameter['k'], V.shape[1]))
    logging.info("nmf3 on %s"%data) 
    testNMF3(V, C, W_init, H_init, parameter)
    logging.info("nmf4 on %s"%data) 
    testNMF4(V, C, W_init, H_init, parameter)

    


        