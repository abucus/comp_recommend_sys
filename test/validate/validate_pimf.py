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
from src.validate.measure_pimf import precision_recall, ndcg
from src.validate.measure import Measure
from src.log import get_logger


class Test(unittest.TestCase):


    def testPIMF(self):
        base_path = op.join('..', '..', 'output', 'data2', 'validate', 'pimf')
        pimf = PIMF(op.join(base_path,'training'), k=50, mu=2)
        test_data = pickle.load(open(op.join(base_path, 'test', 'table'), 'r'))
        event_map = pickle.load(open(op.join(base_path, 'test', 'event_map'), 'r'))
        
        total_precision = []
        total_recall = []
        total_ndcg = []
        
        validate_data_path = op.join(base_path,'test')
        pimf_true = PIMF(validate_data_path, k=50, mu=2)
        
        for k in [3, 5, 10]:
            precisions = []
            recalls = []
            ndcgs = []
            for u, u_data in test_data.iteritems():
                if len(u_data['events']) > 0:
                    relevant = np.array([event_map[e[1]] for e in u_data['events']])[:k]
                    t = u_data['events'][0][0]
                    
                    recommend = pimf.predict(u, t, k)
                    if relevant.shape[0]<k or recommend.shape[0]<k:
                        continue
                    pr = precision_recall(recommend, relevant)
                    precisions.append(pr[0])
                    recalls.append(pr[1])
                    
                    scores = np.array([pimf_true.purchae_prob(u, t, i) for i in recommend])
                    ndcgs.append(ndcg(scores))
            total_precision.append((k, np.mean(precisions)))
            total_recall.append((k, np.mean(recalls))) 
            total_ndcg.append((k, np.mean(ndcgs)))
        
        
        logger = get_logger(__name__)
        logger.info('precisions:\n{}'.format(total_precision))
        logger.info('recalls:\n{}'.format(total_recall))
        logger.info('ndcgs:\n{}'.format(total_ndcg))
        
        measure = Measure(R_hat=pimf.utility, R_test=np.loadtxt(op.join(base_path, 'test', 'utility')))
        
        with open(op.join(base_path, 'test_result.csv'), 'wb') as f:
            writer = csv.writer(f)
            for p in total_precision:
                writer.writerow(['precisions@{}'.format(p[0]), p[1]]) 
            for r in total_recall:
                writer.writerow(['recalls@{}'.format(r[0]), r[1]])
            for c in total_ndcg:
                writer.writerow(['ndcgs@{}'.format(c[0]), c[1]])
            writer.writerow(['RMSE', measure.rmse()])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testPIMF']
    unittest.main()
