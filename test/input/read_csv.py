'''
Created on Feb 12, 2015

@author: tengmf
'''
import unittest
from src.input.read_csv import generate_file,read_in
import os.path as op
class Test(unittest.TestCase):


    def testReadCSV(self):
        generate_file(read_in(source_path=op.join("..", "..", "output", "data2", "original", 'simpleB.csv')),
                      base_file_path=op.join("..", "..", "output", "data2"))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testReadCSV']
    unittest.main()