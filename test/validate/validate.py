'''
Created on Feb 13, 2015

@author: tengmf
'''
import unittest
import os.path as op
from src.decomposition.nmf_3 import NMF3
from src.validate.validate import Validator


class Test(unittest.TestCase):

    def testValidator(self):
        nmf = NMF3()
        validator = Validator(nmf,input_path = op.join("..","..","output","data","validate"),
                              output_path= op.join("..","..","output","data","validate","nmf3"))
        r_list = [10,50,300]
        lambda_list = [[10**(i-3) for i in range(5)],[10**(i-3) for i in range(5)],[10]]
        for i in range(3):
            r = r_list[i]
            for _lambda in lambda_list[i]:
                validator.validate(r, _lambda)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()