'''
Created on Feb 15, 2015

@author: tengmf
'''
import unittest,os,csv
import os.path as op
import numpy as np
from src.lib.nmf import nmf
from numpy.linalg import norm

class Test(unittest.TestCase):


    def setUp(self):
        self.input_path = op.join("..","..","output","data","validate")
        self.output_path = op.join("..","..","output","data","validate","nmf")
        if not op.exists(self.output_path):
            os.makedirs(self.output_path)
        self.V = np.loadtxt(op.join(self.input_path, "training", "pure_matrix.csv"), delimiter = ",")
        self.V_test = np.loadtxt(op.join(self.input_path, "test", "pure_matrix.csv"), delimiter = ",")
        self.I = np.where(self.V_test>0, 1, 0)
        self.non_zero_count = self.I.sum()
        

    def tearDown(self):
        pass


    def testNMF(self):
        results = []
        for r in range(100,400,100):
            print "r=",r," calculating..."
            init_W = np.random.random_sample((self.V.shape[0], r))
            init_H = np.random.random_sample((r, self.V.shape[1]))
            (W, H) = nmf(self.V, init_W, init_H, 10e-5, 100000, 10)
            print np.dot(W,H) 
            diff = (np.dot(W,H)-self.V_test)*self.I
            results.append([r,np.sqrt(norm(diff)**2/self.non_zero_count), norm(diff,1)/self.non_zero_count])
                
        with open(op.join(self.output_path, "stat.txt"), "w") as f:
            writer = csv.writer(f, delimiter = ",")
            writer.writerow(["r", "RMSE", "MAE"])
            writer.writerows(results)
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testNMF']
    unittest.main()