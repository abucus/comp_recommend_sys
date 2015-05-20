'''
Created on Feb 9, 2015

@author: tengmf
'''
import unittest
import numpy as np
import os.path as op
import os
from src.decomposition.nmf_2 import NMF2


class Test(unittest.TestCase):


    def setUp(self):
        self.input_path = base_path = op.join("..","..","output", "nmf2")
        if not op.exists(base_path):
            os.makedirs(base_path)
        self.log_path = op.join(base_path, "nmf2.log")
        if op.exists(self.log_path):
            os.remove(self.log_path)
            
        #row_num = 25
        #col_num = 125
        #r = 5
        #self.V = np.random.random_sample((row_num, col_num))
        self.V = np.loadtxt(op.join(base_path, "nmf_V.txt"))
        self.init_W = np.loadtxt(op.join(base_path, "nmf_init_W.txt"))#np.random.random_sample((row_num, r))
        self.init_H = np.loadtxt(op.join(base_path, "nmf_init_H.txt"))#np.random.random_sample((r, col_num))
        #np.savetxt("my_nmf_V.txt", self.V)


    def tearDown(self):
        pass


    def test_my_nmf(self):
        my_nmf = NMF2()
        (W, H) = my_nmf.factorize(self.V, self.init_W, self.init_H)
        np.savetxt(op.join(self.input_path, "my_nmf_2_W.txt"), W)
        np.savetxt(op.join(self.input_path, "my_nmf_2_H.txt"), H)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()