'''
Created on May 25, 2015

@author: tengmf
'''
import pickle
import unittest
import csv

import numpy as np
import os.path as op
from src.decomposition.pimf import PIMF
from src.validate.measure_pimf import precision_recall
from src.validate.measure import Measure
from src.log import get_logger


class Test(unittest.TestCase):


    def testPIMF(self):
        base_path = op.join('..', '..', 'output', 'data2', 'validate', 'pimf')
        pimf = PIMF(base_path, k=50, mu=2)
        test_data = pickle.load(open(op.join(base_path, 'test', 'table'),'r'))
        event_map = pickle.load(open(op.join(base_path, 'test', 'event_map'),'r'))
        user_map = pickle.load(open(op.join(base_path, 'test', 'user_map'),'r'))
        
        total_precision = []
        total_recall = []
        
        for k in [3,5,10]:
            precision = []
            recall = []
            for u,u_data in test_data.iteritems():
                if len(u_data['events'])>0:
                    for e in u_data['events']:
                        rlt = pimf.predict(u, e[0], k)
                        score = precision_recall(rlt, np.full((1,), event_map[e[1]]))
                        precision.append(score[0])
                        recall.append(score[1])
            total_precision.append((k,np.mean(precision)))
            total_recall.append((k,np.mean(recall))) 
        
        
        logger = get_logger(__name__)
        logger.info('precision:\n{}'.format(total_precision))
        logger.info('recall:\n{}'.format(total_recall))
        
        measure = Measure(R_hat = pimf.utility, R_test=np.loadtxt(op.join(base_path, 'test', 'utility')))
        
        with open(op.join(base_path, 'result.csv'), 'wb') as f:
            writer = csv.writer(f)
            for p in total_precision:
                writer.writerow(['precision@{}'.format(p[0]), p[1]]) 
            for r in total_recall:
                writer.writerow(['recall@{}'.format(r[0]), r[1]])
            writer.writerow(['RMSE', measure.rmse()])
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testPIMF']
    unittest.main()
