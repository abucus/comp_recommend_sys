'''
Created on Feb 19, 2015

@author: tengmf
'''
import unittest
from src.decomposition.nmf_2 import NMF2
from src.validate.validate import Validator
import os.path as op
from src.decomposition.nmf_0 import NMF0

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testDecomposition(self):
        nmf = NMF0()
        validator = Validator(nmf, input_path = op.join("..","..","output","data","validate"), 
                              output_path= op.join("..","..","output","data","validate","nmf0"))
        #for r in range(100,300,100):
        for i in range(3):
            validator.validate(100)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDecomposition']
    unittest.main()