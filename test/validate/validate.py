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
        for r in range(100,500,100):
            for _lambda in [10**(i-3) for i in range(5)]:
                validator.validate(r, _lambda)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()