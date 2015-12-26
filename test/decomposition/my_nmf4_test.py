'''
Created on Feb 7, 2015

@author: tengmf
'''
import unittest, datetime
import numpy as np
import os.path as op
from src.decomposition.nmf_4 import NMF4


class Test(unittest.TestCase):


    def setUp(self):
        row_num = 25
        col_num = 125
        r = 5
        # self.V = np.random.random_sample((row_num, col_num))
        self.V = np.loadtxt(op.join("..", "..", "output", "data2", "pure_matrix.csv"), delimiter=',')
        self.C = np.loadtxt(op.join("..", "..", "output", "data2", "company_matrix"))
        # self.init_W = np.loadtxt("nmf_init_W.txt")#np.random.random_sample((row_num, r))
        # self.init_H = np.loadtxt("nmf_init_H.txt")#np.random.random_sample((r, col_num))
        # np.savetxt("my_nmf_V.txt", self.V)


    def tearDown(self):
        pass


    def test_my_nmf(self):
        V, C = self.V, self.C
        my_nmf = NMF4()
        time_cost = []
        start = datetime.datetime.now()
        W_init = np.random.uniform(1, 2, (V.shape[0], 20))
        H_init = np.random.uniform(1, 2, (20, V.shape[1]))
        (W, H) = my_nmf.factorize(V, C, 20)
        time_cost.append({'k':20, 'time':(datetime.datetime.now() - start).total_seconds()})
        print time_cost

        # (W, H) = my_nmf.factorize(V, self.init_W, self.init_H, 100)
        np.savetxt("my_nmf4_W.txt", W)
        np.savetxt("my_nmf4_H.txt", H)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
