'''
Created on Feb 19, 2015

@author: tengmf
'''
import unittest
from src.decomposition.nmf_2 import NMF2
from src.validate.validate import Validator
import os.path as op
from src.decomposition.nmf_0 import NMF0
import os
import numpy as np
from sklearn.decomposition.nmf import NMF
class Test(unittest.TestCase):


    def setUp(self):
        self.input_path = op.join("..","..","output","data","validate")
        self.output_path = op.join("..","..","output","data","validate","nmf_ski")
        if not op.exists(self.output_path):
            os.makedirs(self.output_path)
        self.V = np.loadtxt(op.join(self.input_path, "training", "pure_matrix.csv"), delimiter = ",")
        self.V_test = np.loadtxt(op.join(self.input_path, "test", "pure_matrix.csv"), delimiter = ",")


    def tearDown(self):
        pass


    def testDecomposition(self):
        model = NMF()
        model.fit(self.V)
        np.savetxt(op.join(self.output_path, "WH.txt"), model.components_)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDecomposition']
    unittest.main()