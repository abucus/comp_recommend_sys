'''
Created on Feb 6, 2015

@author: tengmf
'''
import unittest
import numpy as np
from src.lib.nmf import nmf
from src.decomposition.nmf import NMF

class Test(unittest.TestCase):

    def prepare(self):
        pass
    
    def setUp(self):
        row_num = 25
        col_num = 125
        r = 5
        self.V = np.random.random_sample((row_num, col_num))
        self.init_W = np.random.random_sample((row_num, r))
        self.init_H = np.random.random_sample((r, col_num))
        np.savetxt("nmf_V.txt", self.V)
        pass


    def tearDown(self):
        pass


    def testLibNMF(self):
        (W,H) = nmf(self.V, self.init_W, self.init_H, 10e-3, 10000, 5000)
        np.savetxt("nmf_W.txt", W)
        np.savetxt("nmf_H.txt", H)
    '''    
    def testMyNMF(self):
        V = np.loadtxt("nmf_V.txt")
        my_nmf = NMF()
        (W,H) = my_nmf.factorize(V, self.init_W, self.init_H, 100)
        np.savetxt("my_nmf_W.txt", W)
        np.savetxt("my_nmf_H.txt", H)
    '''
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()