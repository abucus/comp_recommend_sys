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
    Original NMF
    '''
    def __init__(self):
        '''
        Constructor
        '''
        log_path = op.join("..","..","log","my_npm.log")
        if(op.exists(log_path)):
            os.remove(log_path)
        logging.basicConfig(filename = log_path,level=logging.DEBUG)
        
        
    def factorize(self, V, C, WInit=None, HInit=None, max_iter=5):
        '''
        Factorize a non-negative matrix V(nxm) into the product of W(nxr) and H(rxm) 
        '''
        self.V = V
        W = WInit
        H = HInit
        max_iter = 20
        for iter_count in range(max_iter):
            # logging.info("iter count in factorization:{0}".format(iter_count))
            W = self.__compute_argmin_matrix_for_f_wh(W, H, True)
            H = self.__compute_argmin_matrix_for_f_wh(W, H, False)
        return (W, H)
        
    def __compute_argmin_matrix_for_f_wh(self, W, H, transpose = False):
        '''
        Fix W(or H), compute matrix H(or W) which minimize f(W,H) := 1/2 ||V-WH||^2
        '''
        if transpose:
            V = self.V.T
            M = H.T
            logging.info("fix H, compute W")
        else:
            V = self.V
            M = W 
            logging.info("fix W, compute H")
        cols = []
        for i in range(V.shape[1]):
            v = V[:, i]
            if transpose:
                x_init = W[i,:]
            else:
                x_init = H[:,i]
            cols.append(self.__compute_argmin_x_for_fx(v, M, x_init))
        if transpose:
            return np.row_stack(cols)
        else:
            return np.column_stack(cols)
    
       
       
    def __compute_argmin_x_for_fx(self, v, M, x_init, beta=.1,
                                    alpha_init=.1, max_iter=100):
        '''
        Find x which minimize f(x) := 1/2 ||v - Mx||^2
        '''
        #logging.info("v:"+str(v))
        x_old = x_init#np.random.random_sample(M.shape[1])
        alpha = alpha_init
        grad_f_old = self.__grad_f(v, M, x_old)
        x_new = self.__p(x_old - alpha * grad_f_old)
        logging.info("==============\nx_old_init:{0}\nx_new_init:{1}\ngrad_f_old:{2}==================".format(x_old, x_new, grad_f_old))
        iter_count, max_iter = 0, 15
        
        if self.__decrease_condition(v, M, x_old, x_new, grad_f_old):
            # increase step loop
            while True:
                # logging.debug("increasing step, previous alpha is {alpha}".format(alpha=alpha))
                alpha = alpha / beta
                logging.info("---------------------\n x_old before update:{0}".format(x_old))
                x_old = x_new
                grad_f_old = self.__grad_f(v, M, x_old)
                x_new = self.__p(x_old - alpha * grad_f_old)
                need_continue = self.__decrease_condition(v, M, x_old, x_new, grad_f_old)
                logging.info("increasing loop alpha:{0},\nx_new:{1}\n,grad_f_old{2}\ncondition:{3}".format(alpha, x_new, grad_f_old, need_continue))
                iter_count += 1
                if iter_count == max_iter or not need_continue:
                    logging.info("log x_old before break:{0}".format(x_old))
                    break
            logging.info("the latest x is {0}, the latest alpha is {1}\n---------------------\n ".format(x_old, alpha*beta))
            return x_old
        else:
            # decrease step loop
            while iter_count < max_iter and not self.__decrease_condition(v, M, x_old, x_new, grad_f_old):
                # logging.debug("decreasing step, previous alpha is {alpha}".format(alpha=alpha))
                alpha = alpha * beta
                logging.info("---------------------\n x_old before update:{0}".format(x_old))
                x_old = x_new
                grad_f_old = self.__grad_f(v, M, x_old)
                x_new = self.__p(x_old - alpha * grad_f_old)
                need_continue = self.__decrease_condition(v, M, x_old, x_new, grad_f_old)
                logging.info("decreasing loop alpha:{0},\nx_new:{1}\n,grad_f_old{2}\ncondition:{3}".format(alpha, x_new, grad_f_old, need_continue))
                iter_count += 1
                if iter_count == max_iter or need_continue:
                    logging.info("log x_old before break:{0}".format(x_old))
                    break
            logging.info("the latest x is {0}, the latest alpha is {1}\n---------------------\n".format(x_new, alpha))
            return x_new
        # logging.info("has reached the max_iter:{0}".format(iter_count == max_iter))
        
    def __decrease_condition(self, v, M, x_old, x_new, grad_f_old):
        sigma = .01 
        f_x_old = self.__f(v, M, x_old)       
        f_x_new = self.__f(v, M, x_new)
        condition_value = f_x_new - f_x_old - sigma * np.dot(grad_f_old, x_new - x_old)
        # logging.info(" xk={0} \n xk+1={1}".format(x_old, x_new))
        # logging.info(" f_xk={0} \n f_xk+1={1} \n grad_f_xk={2} \n decrease condition value:{3}".format(f_x_old, f_x_new, grad_f_old, condition_value))       
        return condition_value <= 0
        
    def __p(self, x):        
        return np.where(x < 0, 0, x)
    
    def __f(self, v, M, x):
        '''
        Definition of f(x) = 1/2 ||v-Mx||^2
        '''
        return .5 * norm(v - np.dot(M, x)) ** 2
    
    def __grad_f(self, v, M, x):
        '''
        Compute the gradient of f(x) := 1/2 ||v - Mx||^2
        '''
        # print "v.shape {vshape} \n M.shape {mshape} \n x.shape {xshape}".format(vshape=v.shape, mshape=M.shape, xshape=x.shape)
        return np.dot(M.T, np.dot(M, x) - v)
