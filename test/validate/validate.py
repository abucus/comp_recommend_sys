'''
Created on Feb 13, 2015

@author: tengmf
'''
import unittest
from src.decomposition.nmf_3 import NMF3
from src.validate.validate import Validator


class Test(unittest.TestCase):

    def testValidator(self):
        nmf = NMF3()
        validator = Validator(nmf)
        validator.validate(r = 400, _lambda = 1)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()