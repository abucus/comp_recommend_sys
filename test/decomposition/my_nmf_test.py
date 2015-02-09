'''
Created on Feb 7, 2015

@author: tengmf
'''
import unittest
import numpy as np
from src.decomposition.nmf import NMF 


class Test(unittest.TestCase):


    def setUp(self):
        row_num = 25
        col_num = 125
        r = 5
        #self.V = np.random.random_sample((row_num, col_num))
        self.V = np.loadtxt("nmf_V.txt")
        self.init_W = np.loadtxt("nmf_init_W.txt")#np.random.random_sample((row_num, r))
        self.init_H = np.loadtxt("nmf_init_H.txt")#np.random.random_sample((r, col_num))
        #np.savetxt("my_nmf_V.txt", self.V)


    def tearDown(self):
        pass


    def test_my_nmf(self):
        V = self.V
        my_nmf = NMF()
        (W, H) = my_nmf.factorize(V, self.init_W, self.init_H, 100)
        np.savetxt("my_nmf_W.txt", W)
        np.savetxt("my_nmf_H.txt", H)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
