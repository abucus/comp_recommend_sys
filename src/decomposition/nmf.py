'''
Created on Feb 2, 2015

@author: tengmf
'''
import numpy as np
import logging
from numpy.linalg import norm

class NMF(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        logging.basicConfig(filename='./nmf_log.log',level=logging.DEBUG)
        
        
    def factorize(self, V, WInit = None, HInit = None, max_iter = None):
        '''
        Factorize a non-negative matrix V(nxm) into the product of W(nxr) and H(rxm) 
        '''
        self.V = V
        W = WInit
        H = HInit
        max_iter = 100
        for iter_count in range(100):
            W = self.__compute_argmin_matrix_for_f_wh__(H = H)
            H = self.__compute_argmin_matrix_for_f_wh__(W = W)
        return (W,H)
        
    def __compute_argmin_matrix_for_f_wh__(self, W = None, H = None):
        '''
        Fix W(or H), compute matrix H(or W) which minimize f(W,H) := 1/2 ||V-WH||^2
        '''
        if H is not None:
            V = self.V.T
            M = H.T
            logging.info("fix H, compute W")
        else:
            V = self.V
            M = W 
            logging.info("fix W, compute H")
        cols = []
        for i in range(V.shape[1]):
            v = V[:,i]
            cols.append(self.__compute_argmin_x_for_fx__(v, M))
        return np.column_stack(cols)
       
       
    def __compute_argmin_x_for_fx__(self, v, M, beta = .01, 
                        sigma = .01, alpha_init = 1, max_iter = 100):
        '''
        Find x which minimize f(x) := ||v - Mx||
        '''
        logging.info("computing x...")
        x_old = np.full(M.shape[1], 1)
        alpha = alpha_init
        grad_f_old = self.__grad_f__(v, M, x_old)
        x_new = self.__p__(x_old - alpha * grad_f_old)
        if self.__decrease_condition__(v, M, x_old, x_new, grad_f_old, sigma):
            # increase step loop
            while self.__decrease_condition__(v, M, x_old, x_new, grad_f_old, sigma):
                logging.info("increasing step, previous alpha is {alpha}".format(alpha=alpha))
                alpha = alpha/beta
                x_old = x_new
                grad_f_old = self.__grad_f__(v, M, x_old)
                x_new = self.__p__(x_old - alpha*grad_f_old)
        else:
            # decrease step loop
            while not self.__decrease_condition__(v, M, x_old, x_new, grad_f_old, sigma):
                logging.info("increasing step, previous alpha is {alpha}".format(alpha=alpha))
                alpha = alpha/beta
                x_old = x_new
                grad_f_old = self.__grad_f__(v, M, x_old)
                x_new = self.__p__(x_old - alpha*grad_f_old)
        logging.info("the latest x is {0}".format(x_old))
        return x_old
        
    def __decrease_condition__(self, v, M, x_old, x_new, grad_f_old,sigma): 
        f_x_old = self.__f__(v, M, x_old)       
        f_x_new = self.__f__(v, M, x_new)       
        return f_x_new-f_x_old-sigma*np.dot(grad_f_old, x_new - x_old) <= 0
        
    def __p__(self,x):
        l = np.zeros(x.shape)
        u = np.full(x.shape, 1., dtype=np.float)
        rlt = np.full(x.shape, 0., dtype=np.float)
        for i in range(x.shape[0]):
            if l[i]<x[i] and x[i]<u[i] :
                rlt[i] = x[i]
            elif x[i]>u[i]:
                rlt[i] = u[i]
            elif x[i]<l[i]:
                rlt[i] = l[i]
        return rlt
    
    def __f__(self, v, M, x):
        return norm(v - np.dot(M, x))**2
    
    def __grad_f__(self, v, M, x):
        '''
        Compute the gradient of f(x) := 1/2 ||v - Mx||^2
        '''
        print "v.shape {vshape} \n M.shape {mshape} \n x.shape {xshape}".format(vshape=v.shape, mshape=M.shape, xshape=x.shape)
        return np.dot(M.T, np.dot(M,x)-v) 
            
        
        
   