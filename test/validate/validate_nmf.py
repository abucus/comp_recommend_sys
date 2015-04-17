'''
Created on Feb 15, 2015

@author: tengmf
'''
import unittest,os,csv,json
import os.path as op
import numpy as np
from src.lib.nmf import nmf
from numpy.linalg import norm

class Test(unittest.TestCase):


    def setUp(self):
        self.input_path = op.join("..","..","output","data2","validate")
        self.output_path = op.join("..","..","output","data2","validate","nmf")
        if not op.exists(self.output_path):
            os.makedirs(self.output_path)
        
        if op.exists(op.join(self.output_path, "outputconfig.txt")):
            with open(op.join(self.output_path, "outputconfig.txt")) as cf:
                config = json.load(cf)
                self.output_count = config['count']
        else:
            self.output_count = 1
             
        self.V = np.loadtxt(op.join(self.input_path, "training", "pure_matrix.csv"), delimiter = ",")
        self.V_test = np.loadtxt(op.join(self.input_path, "test", "pure_matrix.csv"), delimiter = ",")
        self.I = np.where(self.V_test>0, 1, 0)
        self.non_zero_count = self.I.sum()
        

    def tearDown(self):
        pass


    def testNMF(self):
        for r in range(100,400,100):
            print "r=",r," calculating..."
            init_W = np.random.random_sample((self.V.shape[0], r))
            init_H = np.random.random_sample((r, self.V.shape[1]))
            (W, H) = nmf(self.V, init_W, init_H, 0.1, 100000, 2)
            WH =  np.dot(W,H) 
            cur_path = op.join(self.output_path, str(self.output_count))
            if not op.exists(cur_path):
                os.makedirs(cur_path)
            np.savetxt(op.join(cur_path,"W.txt"), W)
            np.savetxt(op.join(cur_path,"H.txt"), H)
            np.savetxt(op.join(cur_path,"WH.txt"), WH)
                
            with open(op.join(cur_path, "result.txt"), "w") as f:
                json.dump({'r':r,'lambda':None}, f)
            
            self.output_count += 1
        with open(op.join(self.output_path, "outputconfig.txt"), "w") as f:
            json.dump({'count':self.output_count}, f)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testNMF']
    unittest.main()