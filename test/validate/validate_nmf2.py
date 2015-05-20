'''
Created on Feb 19, 2015

@author: tengmf
'''
import unittest
from src.decomposition.nmf_2 import NMF2
from src.validate.validate import Validator
import os.path as op

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testDecomposition(self):
        nmf = NMF2()
        validator = Validator(nmf, input_path = op.join("..","..","output","data2","validate"), 
                              output_path= op.join("..","..","output","data2","validate","no_regularization"))
        #for r in range(100,300,100):
        validator.validate(100)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDecomposition']
    unittest.main()