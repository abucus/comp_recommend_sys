'''
Created on Feb 12, 2015

@author: tengmf
'''
import os.path as op
import numpy as np
import json,os
from numpy.linalg import norm
class Validator(object):
    '''
    classdocs
    '''

    def __init__(self, nmf, input_path = op.join("..","..","output","data","validate"), 
                 output_path= op.join("..","..","output","data","validate")):
        '''
        Constructor
        '''
        self.nmf = nmf
        #training_stat =open(op.join(input_path, "training","stat"))
        #test_stat =open(op.join(input_path, "test","stat"))
        self.input_path = input_path
        self.output_path = output_path
        
        if not op.exists(self.input_path):
            os.makedirs(self.input_path)
        if not op.exists(self.output_path):
            os.makedirs(self.output_path)
        output_config_path = op.join(output_path, "output_config.txt")
        
        if not op.exists(output_config_path):
            self.output_count = 1
        else:
            output_config_file = open(output_config_path, "r")
            output_config = json.load(output_config_file)
            self.output_count = output_config['count']
            
        self.r = None
        self.training_V = np.loadtxt(op.join(self.input_path, "training","pure_matrix.csv"), delimiter=",")
        self.C = np.loadtxt(op.join(self.input_path, "training","company_matrix"))
        self.test_V = np.loadtxt(op.join(self.input_path, "test","pure_matrix.csv"), delimiter=",")
        self.I = np.where(self.test_V>0, 1, 0)
        self.non_zero_count = self.I.sum()
        print "****** validator init done ******" \
        "\noutput_count:", self.output_count, \
        "\ninput_path:",self.input_path
    
    def validate(self, r, _lambda):
        
        if not self.r or self.r != r:
            self.r = r
            initW = self.initW = np.random.random_sample((self.training_V.shape[0], r))
            initH = self.initH = np.random.random_sample((r, self.training_V.shape[1]))
        else:
            initW = self.initW
            initH = self.initH
            
        print "****** init W,H done ******" \
        "\nV shape:", self.training_V.shape,\
        "\ninit W shape:", initW.shape,\
        "\ninit H shape:", initH.shape
        
        (W,H) = self.nmf.factorize(self.training_V, self.C, initW, initH, _lambda)
        
        WH = np.dot(W,H)
        
        output_path = op.join(self.output_path, str(self.output_count))
        if not op.exists(output_path):
            os.makedirs(output_path)
        np.savetxt(op.join(output_path,"W.csv"), W)
        np.savetxt(op.join(output_path,"H.csv"), H)
        np.savetxt(op.join(output_path, "WH.txt"), WH)
        del W,H
        
        
        diff = (WH-self.test_V)*self.I
        with open(op.join(output_path, "result.txt"), 'w') as f:
            json.dump({'rmse': np.sqrt(norm(diff)**2/self.non_zero_count), 'mae': norm(diff,1)/self.non_zero_count, 'r':r, 'lambda':_lambda}, f)
        
        del WH
        
        self.output_count += 1
        with open(op.join(self.output_path, "output_config.txt"), 'w') as f:
            json.dump({'count':self.output_count}, f)
