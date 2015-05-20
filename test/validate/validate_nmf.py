'''
Created on Feb 19, 2015

@author: tengmf
'''
import unittest
from src.validate.validate import Validator
import os.path as op
from src.decomposition.nmf import NMF

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testDecomposition(self):
        nmf = NMF()
        validator = Validator(nmf, input_path = op.join("..","..","output","data","validate"), 
                              output_path= op.join("..","..","output","data","validate","nmf"))
        #for r in range(100,300,100):
        for i in range(3):
            validator.validate(100)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDecomposition']
    unittest.main()