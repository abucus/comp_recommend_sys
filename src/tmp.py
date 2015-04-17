'''
Created on Feb 15, 2015

@author: tengmf
'''
import os.path as op
import numpy as np
import json,csv
import matplotlib.pyplot as pt
from numpy.linalg import norm
from src.input.read_csv import read_in

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

def cal_events_length():
    data = read_in(source_path=op.join("..", "output", "data", "original", 'simpleA2.csv'))
    #events = []
    time_interval_avg = []
    all_ = []
    for v in data['table'].values():
        #events.append(len(v['events']))
        c_intervals = []
        for i in range(1, len(v['events'])):
            interval = (v['events'][i][0] - v['events'][i-1][0]).total_seconds()/3600./24.
            if interval <= 90:
                c_intervals.append(interval)
                all_.append(interval)
        if len(c_intervals)>0:
            time_interval_avg.append(np.mean(c_intervals))
    with open(op.join("..","output","data","avg_time_interval_by_user.csv"), "wb") as f:
        writer = csv.writer(f)
        for row in time_interval_avg:
            writer.writerow([row])
    with open(op.join("..","output","data","avg_time_interval.csv"), "wb") as f:
        writer = csv.writer(f)
        for row in all_:
            writer.writerow([row])
    print time_interval_avg
    with open(op.join("..","output","data","avg_time_interval_mean.txt"), "wb") as f:
        json.dump({'avg_by_user':np.mean(time_interval_avg), 'avg':np.mean(all_)}, f)
def cal_company_size():
    data = read_in(source_path=op.join("..", "output", "data", "original", 'simpleA2.csv'))
    company = {}
    count = 0
    for v in data['table'].values(): 
        count += 1
        if v['company'] not in company:
            company[v['company']] = 1
        else:
            company[v['company']] += 1
    cs = company.values()
    print "mean",np.mean(cs),"max",np.max(cs),"min",np.min(cs),"mid",np.median(cs)
    print "#people",count,"#company",len(company),"avg",count*1./len(company)
    bins = [i+0.5 for i in range(10)]
    pt.hist(cs,bins)
    pt.show()
#     print np.mean(all_)    
#     print "mean:",np.mean(events)
#     print "max:",np.max(events)
#     print "min:",np.min(events)
cal_company_size()