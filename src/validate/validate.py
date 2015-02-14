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
        print "****** validator init done ******" \
        "\noutput_count:", self.output_count, \
        "\ninput_path:",self.input_path
    
    def validate(self, r, _lambda):
        training_V = np.loadtxt(op.join(self.input_path, "training","pure_matrix.csv"), delimiter=",")
        C = np.loadtxt(op.join(self.input_path, "training","company_matrix"))
        initW = np.random.random_sample((training_V.shape[0], r))
        initH = np.random.random_sample((r, training_V.shape[1]))
        print "****** init W,H done ******" \
        "\nV shape:", training_V.shape,\
        "\ninit W shape:", initW.shape,\
        "\ninit H shape:", initH.shape
        
        (W,H) = self.nmf.factorize(training_V, C, initW, initH)
        del training_V
        
        test_V = np.loadtxt(op.join(self.input_path, "test","pure_matrix.csv"))
        WH = np.dot(W,H)
        
        output_path = op.join(self.output_path, str(self.output_count))
        if not op.exists(output_path):
            os.makedirs(output_path)
        np.savetxt(op.join(output_path,"W.csv"), W)
        np.savetxt(op.join(output_path,"H.csv"), H)
        np.savetxt(op.join(output_path, "WH.txt"), WH)
        np.savetxt(op.join(output_path, "error.txt"), norm(WH-test_V))
        
        del W,H,WH,test_V
        
        with open(op.join(self.output_path, "output_config.txt"), 'w') as f:
            json.dump({'count':self.output_count+1}, f)
        
        
        
        
        