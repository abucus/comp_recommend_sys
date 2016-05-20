'''
Created on May 29, 2015

@author: tengmf
'''
import os.path as op
import json

for i in range(1, 150):
    path = op.join("..", "output", "data2", "validate", "nmf3", str(i), 'result.txt')
    if op.exists(path):
        result = json.load(open(path, 'r'))
        if result['r'] == 100 and result['lambda'] == 0.001:
            print(i,result['r'],result['lambda'],result['rmse'])
