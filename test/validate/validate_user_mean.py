'''
Created on Feb 19, 2015

@author: tengmf
'''
import unittest
import unittest,os,csv
import os.path as op
import numpy as np

class Test(unittest.TestCase):


    def setUp(self):
        self.input_path = op.join("..","..","output","data","validate")
        self.output_path = op.join("..","..","output","data","validate","user_mean")
        if not op.exists(self.output_path):
            os.makedirs(self.output_path)
        self.V = np.loadtxt(op.join(self.input_path, "training", "pure_matrix.csv"), delimiter = ",")
        self.V_test = np.loadtxt(op.join(self.input_path, "test", "pure_matrix.csv"), delimiter = ",")

    def tearDown(self):
        pass


    def testUserMean(self):
        user_mean = self.V.mean(axis = 1)
        abs_sum = square_sum = 0.
        non_zero_idxes = zip(*np.where(self.V_test>0))
        non_zero_count = len(non_zero_idxes)
        for i,j in non_zero_idxes:
            v = np.abs(self.V_test[i,j] - user_mean[i]) 
            abs_sum += v
            square_sum += v**2
        
        with open(op.join(self.output_path, "result.txt"), "w") as f:
            writer = csv.writer(f, delimiter = ",")
            writer.writerow(["RMSE", "MAE"])
            writer.writerow([np.sqrt(square_sum/non_zero_count), abs_sum/non_zero_count])
        
        
            


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testUserMean']
    unittest.main()