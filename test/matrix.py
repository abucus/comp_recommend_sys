'''
Created on Feb 11, 2015

@author: tengmf
'''
import unittest
import numpy as np
import datetime
class Test(unittest.TestCase):

    def setUp(self):
        self.a = np.random.random_sample((3000,10000))
    def tearDown(self):
        pass
    def testMaxtrixAccess(self):
        time_start = datetime.datetime.now()
        for i in range(self.a.shape[1]):
            b = self.a[:,i]
        time_end = datetime.datetime.now()
        print (time_end-time_start).total_seconds()
        
        time_start = datetime.datetime.now()
        for i in range(self.a.shape[1]):
            b = self.a.T[i]
        time_end = datetime.datetime.now()
        print (time_end-time_start).total_seconds()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testMaxtrixAccess']
    unittest.main()