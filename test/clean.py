'''
Created on Feb 8, 2015

@author: tengmf
'''
import os.path as op
import glob
import os
CLEAN_PATH = (op.join(".","decomposition","*.txt"), op.join(".","decomposition","*.log"), op.join("..","log","*.log"))
for p in CLEAN_PATH:
    for fl in glob.glob(p):
        os.remove(fl)
