'''
Created on Feb 11, 2015

@author: tengmf
'''
from src.decomposition.nmf import NMF
from numpy.linalg import norm
import numpy as np
import os.path as op
import logging,os

class NMF3(object):
    '''
    NMF with regularization
    '''


    def __init__(self, log_path = op.join(".","my_nmf.log")):
        '''
        Constructor
        '''
        if(op.exists(log_path)):
            os.remove(log_path)
        logging.basicConfig(filename = log_path,level=logging.DEBUG)
        
        
    def factorize(self, V, C, WInit=None, HInit=None, max_iter=20):
        '''
        Factorize a non-negative matrix V(nxm) into the product of W(nxr) and H(rxm) 
        
        V the matrix to be factorized
        C the company matrix
        '''
        self.V = V
        self.C = C
        self.I = np.where(V>0, 1, 0)
        W = WInit
        H = HInit
        for iter_count in range(max_iter):
            # logging.info("iter count in factorization:{0}".format(iter_count))
            self.computing_W = True
            W = self.__compute_argmin_matrix_for_f_wh(W, H)
            self.computing_W = False
            H = self.__compute_argmin_matrix_for_f_wh(W, H)
        return (W, H)
        
    def __compute_argmin_matrix_for_f_wh(self, W, H):
        '''
        Fix W(or H), compute matrix H(or W) which minimize f(W,H) := 1/2 ||V-WH||^2
        '''
        if self.computing_W:
            X_old = W
        else:
            X_old = H
        grad_f_old = self.__grad_f(W, H)
        alpha = .1
        beta=.1
        X_new = self.__p(X_old - alpha * grad_f_old)
        
        iter_count, max_iter = 0, 15
        
        if self.__decrease_condition(W, H, X_new, grad_f_old):
            # increase step loop
            while True:
                # logging.debug("increasing step, previous alpha is {alpha}".format(alpha=alpha))
                alpha = alpha / beta
                X_old = X_new
                grad_f_old = self.__grad_f(v, M, x_old, nni)
                x_new = self.__p(x_old - alpha * grad_f_old)
                need_continue = self.__decrease_condition(v, M, x_old, x_new, grad_f_old, nni)
                logging.info("increasing loop alpha:{0},\nx_new:{1}\n,grad_f_old{2}\ncondition:{3}".format(alpha, x_new, grad_f_old, need_continue))
                iter_count += 1
                if iter_count == max_iter or not need_continue:
                    logging.info("log x_old before break:{0}".format(x_old))
                    break
            logging.info("the latest x is {0}, the latest alpha is {1}\n---------------------\n ".format(x_old, alpha*beta))
            return x_old
        else:
            # decrease step loop
            while True:
                # logging.debug("decreasing step, previous alpha is {alpha}".format(alpha=alpha))
                alpha = alpha * beta
                logging.info("---------------------\n x_old before update:{0}".format(x_old))
                x_old = x_new
                grad_f_old = self.__grad_f(v, M, x_old, nni)
                x_new = self.__p(x_old - alpha * grad_f_old)
                need_continue = self.__decrease_condition(v, M, x_old, x_new, grad_f_old, nni)
                logging.info("decreasing loop alpha:{0},\nx_new:{1}\n,grad_f_old{2}\ncondition:{3}".format(alpha, x_new, grad_f_old, need_continue))
                iter_count += 1
                if iter_count == max_iter or need_continue:
                    logging.info("log x_old before break:{0}".format(x_old))
                    break
            logging.info("the latest x is {0}, the latest alpha is {1}\n---------------------\n".format(x_new, alpha))
            return x_new
       
       
        
    def __decrease_condition(self, W, H, X_new, grad_f_old):
        sigma = .01 
        f_x_old = self.__f(W, H)
        if self.computing_W:
            f_x_new = self.__f(X_new, H)
        else:
            f_x_new = self.__f(W, X_new)
        
        condition_value = f_x_new - f_x_old - sigma * np.dot(grad_f_old, x_new - x_old)
        # logging.info(" xk={0} \n xk+1={1}".format(x_old, x_new))
        # logging.info(" f_xk={0} \n f_xk+1={1} \n grad_f_xk={2} \n decrease condition value:{3}".format(f_x_old, f_x_new, grad_f_old, condition_value))       
        return condition_value <= 0
        
    def __p(self, x):        
        return np.where(x < 0, 0, x)
    
    def __f(self, W, H):
        '''
        Definition of f(x) = 1/2 ||(v-Mx)*nni||^2, (v1,v2)*(v3,v4)=(v1*v2,v3*v4)
        '''
        return .5 * norm((self.V-np.dot(W,H))*self.I)**2 + np.dot(np.dot(W.T,self.C),W) 
    
    def __grad_f(self, W, H):
        '''
        Compute the gradient of f(x) := 1/2 ||(v-Mx)*nni||^2, (v1,v2)*(v3,v4)=(v1*v2,v3*v4)
        '''
        V = self.V
        I = self.I
        C = self.C
        if self.computing_W:
            grad = np.zeros(W.shape)
            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    # calculate grad_i_j
                    for k in range(V.shape[1]):
                        grad[i,j] += (np.dot(W[i],H[:,k]) - V[i,k])*H[j,k]*I[i,k]
                    grad[i,j] += np.dot(W[:,j],C[i]) + np.dot(W[:,j],C[:,i])
        else:
            grad = np.zeros(H.shape)
            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    # calculate grad_i_j 
                    for k in range(V.shape[0]):
                        grad[i,j] += (np.dot(W[k],H[:,j]) - V[k,j])*W[k,i]*I[k,j]
        return grad