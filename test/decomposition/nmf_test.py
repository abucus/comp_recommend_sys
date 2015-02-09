'''
Created on Feb 6, 2015

@author: tengmf
'''
import unittest
import numpy as np
from src.lib.nmf import nmf

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


    def tearDown(self):
        pass


    def testLibNMF(self):
        (W, H) = nmf(self.V, self.init_W, self.init_H, 10e-3, 10000, 5000)
        np.savetxt("nmf_init_W.txt", self.init_W)
        np.savetxt("nmf_init_H.txt", self.init_H)
        np.savetxt("nmf_W.txt", W)
        np.savetxt("nmf_H.txt", H)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
