'''
Created on Feb 19, 2015

@author: tengmf
'''
import os.path as op
import numpy as np
import csv,json,datetime,pickle
from src.validate.measure import Measure
from src.validate.measure import flattern_fpmc_matrix

def cal_measure():
    input_path = op.join("..", "..","output","data2","validate")
    output_path = op.join("..", "..","output","data2","validate", "nmf0")
    folders = range(2,7)
    
    k_list = [3,5,10]
    R_test = np.loadtxt(op.join(input_path, "test","pure_matrix.csv"),delimiter=",")
    m = None
    with open(op.join(output_path,"measures.csv"), "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["r", "lambda","MAE", "RMSE"]+[i+str(j) for i in ["Precision", "Recall", "NCDG"] for j in k_list])
        for i in folders:
            print "calculating folder",i,"starting time,", datetime.datetime.now()
            R_hat = np.loadtxt(op.join(output_path, str(i), "WH.txt"))
            if not m:
                m = Measure(R_hat, R_test)
            else:
                m.reset(R_hat)
                
            with open(op.join(output_path, str(i), "result.txt"), "r") as f:
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
    
def cal2():
    input_path = op.join("..", "..","output","data2","validate")
    R_test = np.loadtxt(op.join(input_path, "test","pure_matrix.csv"),delimiter=",")
    R_hat1 = np.loadtxt(op.join(input_path, "user_mean", "R_hat.txt"))  
    R_hat2 = np.zeros(R_test.shape)
    print (R_hat2-R_hat1).mean()
    m = Measure(R_hat1, R_test)
    m2 = Measure(R_hat2, R_test)
    for i in [3,5,10]:
        pr = m.precision_recall(i)
        pr2 = m2.precision_recall(i)
        print "precision@",i,pr[0]==pr2[0]
        print "recall@",i,pr[1]==pr2[1]
        print "ndcg@",i,m.ndcg(i)==m2.ndcg(i)

def cal_fpmc():
    base_path = op.join('..','..','output','fpmc_data')
    a_train = flattern_fpmc_matrix(pickle.load(open(op.join(base_path, 'training','a'))))
    a_test = flattern_fpmc_matrix(pickle.load(open(op.join(base_path, 'test', 'full_tensor'))))
    print a_train.shape, a_test.shape
    m = Measure(a_train, a_test)
    print 'MAE:',m.mae()
    print 'RMSE:',m.rmse()
    k_list = [3,5,10]
    precision_relt = []
    recall_relt = []
    ncdg_relt = []
    for j in k_list:
        pr = m.precision_recall(j)
        print 'precision@'+str(j),pr[0]
        print 'recall@'+str(j),pr[1]
        print 'ncdg@'+str(j),m.ndcg(j)

if __name__ == '__main__':
    # for nmf
    # cal_measure()
    
    # for fpmc
    # cal_fpmc()    
    