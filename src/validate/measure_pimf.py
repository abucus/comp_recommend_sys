'''
Created on May 25, 2015

@author: tengmf
'''
import numpy as np

def precision_recall(retrived, relevant):
    retrived = retrived
    relevant = relevant
    intersect_num = len(np.intersect1d(retrived, relevant))
    print retrived,relevant
    return (1.*intersect_num/len(retrived), 1.*intersect_num/len(relevant))

