'''
Created on Feb 12, 2015

@author: tengmf
'''
import unittest
from src.input.read_csv import generate_file,read_in

class Test(unittest.TestCase):


    def testReadCSV(self):
        generate_file(read_in())
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testReadCSV']
    unittest.main()