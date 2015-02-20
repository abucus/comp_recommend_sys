'''
Created on Feb 19, 2015

@author: tengmf
'''
import os.path as op
import numpy as np
import csv,json,datetime
from src.validate.measure import Measure

base_path = op.join("..", "..","output","data","validate")
k_list = [3,5,10]
R_test = np.loadtxt(op.join(base_path, "test","pure_matrix.csv"),delimiter=",")
m = None
with open(op.join("..", "..","output","data","validate","measures.csv"), "wb") as f:
    writer = csv.writer(f)
    writer.writerow(["r", "lambda","MAE", "RMSE"]+[i+str(j) for i in ["Precision", "Recall", "NCDG"] for j in k_list])
    for i in range(2,18):
        print "calculating folder",i,"starting time,", datetime.datetime.now()
        R_hat = np.loadtxt(op.join(base_path, str(i), "WH.txt"))
        if not m:
            m = Measure(R_hat, R_test)
        else:
            m.reset(R_hat)
            
        with open(op.join(base_path, str(i), "result.txt"), "r") as f:
            par = json.load(f)
        result_1 = [par['r'],par['lambda'],m.mae(),m.rmse()]
        
        precision_relt = []
        recall_relt = []
        ncdg_relt = []
        for j in k_list:
            pr = m.precision_recall(j)
            precision_relt.append(pr[0])
            recall_relt.append(pr[1])
            ncdg_relt.append(m.ndcg(j))

        writer.writerow(result_1 + precision_relt + recall_relt + ncdg_relt)
    
    
    
    