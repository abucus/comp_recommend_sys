'''
Created on Feb 19, 2015

@author: tengmf
'''
import unittest, os, csv, datetime
import os.path as op
import numpy as np
from src.validate.measure import Measure

class Test(unittest.TestCase):


    def setUp(self):
        self.input_path = op.join("..", "..", "output", "data2", "validate")
        self.output_path = op.join("..", "..", "output", "data2", "validate", "user_mean")
        if not op.exists(self.output_path):
            os.makedirs(self.output_path)
        self.V = np.loadtxt(op.join(self.input_path, "training", "pure_matrix.csv"), delimiter=",")
        self.V_test = np.loadtxt(op.join(self.input_path, "test", "pure_matrix.csv"), delimiter=",")

    def tearDown(self):
        pass


    def testUserMean(self):
#        user_mean = self.V.mean(axis = 1)
#        abs_sum = square_sum = 0.
#        non_zero_idxes = zip(*np.where(self.V_test>0))
#        non_zero_count = len(non_zero_idxes)
#         for i,j in non_zero_idxes:
#             v = np.abs(self.V_test[i,j] - user_mean[i]) 
#             abs_sum += v
#             square_sum += v**2
        if op.exists(op.join(self.output_path, "R_hat.txt")):
            print "load R_hat"
            R_hat = np.loadtxt(op.join(self.output_path, "R_hat.txt"))
        else:
            R_hat = np.zeros(self.V_test.shape)
            for i in range(0, self.V.shape[0]):
                print "constructing row", i, " starting time", datetime.datetime.now()
                row = self.V[i]
                if np.any(row != 0):
                    # avg of all row elem or avg of all postive elem
                    R_hat[i] = row[row > 0].mean()
            np.savetxt(op.join(self.output_path, "R_hat.txt"), R_hat)
        '''
        m = Measure(R_hat, self.V_test)
        k_list = [3,5,10]
        with open(op.join(self.output_path, "measures.txt"), "w") as f:
            writer = csv.writer(f, delimiter = ",")
            writer.writerow(["MAE", "RMSE"]+[i+str(j) for i in ["Precision", "Recall", "NCDG"] for j in k_list])
            result_1 = [m.mae(), m.rmse()]
            #writer.writerow([np.sqrt(square_sum/non_zero_count), abs_sum/non_zero_count])
            precision_relt = []
            recall_relt = []
            ncdg_relt = []
            for j in k_list:
                pr = m.precision_recall(j)
                precision_relt.append(pr[0])
                recall_relt.append(pr[1])
                ncdg_relt.append(m.ndcg(j))

            writer.writerow(result_1 + precision_relt + recall_relt + ncdg_relt)
        '''
            


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testUserMean']
    unittest.main()
