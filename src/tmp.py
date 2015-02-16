'''
Created on Feb 15, 2015

@author: tengmf
'''
import os.path as op
import numpy as np
import json,csv
from numpy.linalg import norm

def recalerror():
    path = op.join("..", "output","data","validate")
    V_test = np.loadtxt(op.join(path, "test", "pure_matrix.csv"),delimiter=",")
    I = np.where(V_test>0, 1, 0)
    non_zero_count = I.sum()
    results = []
    for i in range(2,17):
        WH = np.loadtxt(op.join(path, str(i), "WH.txt"))
        with open(op.join(path, str(i), "result.txt"), "r") as f:
            result = json.load(f)
        diff = (WH - V_test)*I
        mae = norm(diff,1)/non_zero_count
        rmse = np.sqrt(norm(diff)**2/non_zero_count)
        results.append([result['r'],rmse,mae])
        
        del result['norm']
        result['mae'],result['rmse'] = mae,rmse
        with open(op.join(path, str(i), "result.txt"), "w") as f:
            json.dump(result, f)
            
    with open(op.join(path, "total_stat.txt"), "w") as f:
            writer = csv.writer(f, delimiter = ",")
            writer.writerow(["r", "RMSE", "MAE"])
            writer.writerows(results)

def cal_rmse():
    path = op.join("..", "output","data","validate")
    V = np.loadtxt(op.join(path, "training", "pure_matrix.csv"),delimiter=",") 
    
    
recalerror()    