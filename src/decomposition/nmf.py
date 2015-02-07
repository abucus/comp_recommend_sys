'''
Created on Feb 2, 2015

@author: tengmf
'''
import numpy as np
import os
import os.path as op
import logging
from numpy.linalg import norm

class NMF(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        
        log_path = op.join("..","..","log","my_npm.log")
        if(op.exists(log_path)):
            os.remove(log_path)
        logging.basicConfig(filename = log_path,level=logging.ERROR)
        '''
        
    def factorize(self, V, WInit = None, HInit = None, max_iter = None):
        '''
        Factorize a non-negative matrix V(nxm) into the product of W(nxr) and H(rxm) 
        '''
        self.V = V
        W = WInit
        H = HInit
        max_iter = 100
        for iter_count in range(max_iter):
            #logging.info("iter count in factorization:{0}".format(iter_count))
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
            #logging.info("fix H, compute W")
        else:
            V = self.V
            M = W 
            #logging.info("fix W, compute H")
        cols = []
        for i in range(V.shape[1]):
            v = V[:,i]
            cols.append(self.__compute_argmin_x_for_fx__(v, M))
        if H is not None:
            return np.row_stack(cols)
        else:
            return np.column_stack(cols)
    
       
       
    def __compute_argmin_x_for_fx__(self, v, M, beta = .1, 
                        sigma = .01, alpha_init = 1, max_iter = 100):
        '''
        Find x which minimize f(x) := 1/2 ||v - Mx||^2
        '''
        #logging.info("computing x...")
        x_old = np.full(M.shape[1], 1)
        alpha = alpha_init
        grad_f_old = self.__grad_f__(v, M, x_old)
        x_new = self.__p__(x_old - alpha * grad_f_old)
        iter_count, max_iter = 0, 15
        
        if self.__decrease_condition__(v, M, x_old, x_new, grad_f_old, sigma):
            # increase step loop
            while iter_count < max_iter and self.__decrease_condition__(v, M, x_old, x_new, grad_f_old, sigma):
                #logging.debug("increasing step, previous alpha is {alpha}".format(alpha=alpha))
                alpha = alpha/beta
                x_old = x_new
                grad_f_old = self.__grad_f__(v, M, x_old)
                x_new = self.__p__(x_old - alpha*grad_f_old)
                iter_count += 1
        else:
            # decrease step loop
            while iter_count < max_iter and not self.__decrease_condition__(v, M, x_old, x_new, grad_f_old, sigma):
                #logging.debug("decreasing step, previous alpha is {alpha}".format(alpha=alpha))
                alpha = alpha*beta
                x_old = x_new
                grad_f_old = self.__grad_f__(v, M, x_old)
                x_new = self.__p__(x_old - alpha*grad_f_old)
                iter_count += 1
        #logging.info("has reached the max_iter:{0}".format(iter_count == max_iter))
        #logging.info("the latest x is {0}, the latest alpha is {1}".format(x_old, alpha))
        return x_old
        
    def __decrease_condition__(self, v, M, x_old, x_new, grad_f_old,sigma): 
        f_x_old = self.__f__(v, M, x_old)       
        f_x_new = self.__f__(v, M, x_new)
        condition_value = f_x_new-f_x_old-sigma*np.dot(grad_f_old, x_new - x_old)
        #logging.info(" xk={0} \n xk+1={1}".format(x_old, x_new))
        #logging.info(" f_xk={0} \n f_xk+1={1} \n grad_f_xk={2} \n decrease condition value:{3}".format(f_x_old, f_x_new, grad_f_old, condition_value))       
        return condition_value <= 0
        
    def __p__(self,x):        
        return np.where(x<0, 0, x)
    
    def __f__(self, v, M, x):
        '''
        Definition of f(x) = 1/2 ||v-Mx||^2
        '''
        return .5*norm(v - np.dot(M, x))**2
    
    def __grad_f__(self, v, M, x):
        '''
        Compute the gradient of f(x) := 1/2 ||v - Mx||^2
        '''
        #print "v.shape {vshape} \n M.shape {mshape} \n x.shape {xshape}".format(vshape=v.shape, mshape=M.shape, xshape=x.shape)
        return np.dot(M.T, np.dot(M,x)-v)
            
        
        
   