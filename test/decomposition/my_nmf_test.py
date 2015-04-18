'''
Created on Feb 7, 2015

@author: tengmf
'''
import unittest,datetime
import numpy as np
import os.path as op
from src.decomposition.nmf_3 import NMF3 


class Test(unittest.TestCase):


    def setUp(self):
        row_num = 25
        col_num = 125
        r = 5
        #self.V = np.random.random_sample((row_num, col_num))
        self.V = np.loadtxt(op.join("..","..","output","data2","pure_matrix.csv"), delimiter=',')
        self.C = np.loadtxt(op.join("..","..","output","data2","company_matrix"))
        #self.init_W = np.loadtxt("nmf_init_W.txt")#np.random.random_sample((row_num, r))
        #self.init_H = np.loadtxt("nmf_init_H.txt")#np.random.random_sample((r, col_num))
        #np.savetxt("my_nmf_V.txt", self.V)


    def tearDown(self):
        pass


    def test_my_nmf(self):
        V, C = self.V, self.C
        my_nmf = NMF3()
        time_cost = []
        for k in [10,20]:
            start = datetime.datetime.now()
            W_init = np.random.normal(0,1,(V.shape[0],k))
            H_init = np.random.normal(0,1,(k,V.shape[1]))
            (W,H) = my_nmf.factorize(V, C, W_init, H_init)
            time_cost.append({'k':k,'time':(datetime.datetime.now()-start).total_seconds()})
        print time_cost

        #(W, H) = my_nmf.factorize(V, self.init_W, self.init_H, 100)
        #np.savetxt("my_nmf_W.txt", W)
        #np.savetxt("my_nmf_H.txt", H)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
