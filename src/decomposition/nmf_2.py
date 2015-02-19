'''
Created on Feb 9, 2015

@author: tengmf
'''
from src.decomposition.nmf import NMF
from numpy.linalg import norm
import numpy as np
import os.path as op
import logging,os, datetime
class NMF2(object):
    '''
    NMF with regularization
    '''


    def __init__(self, log_path = op.join(".","my_nmf.log")):
        '''
        Constructor
        '''
        if(op.exists(log_path)):
            os.remove(log_path)
        
        
    def factorize(self, V, C, WInit=None, HInit=None, max_iter=10):
        '''
        Factorize a non-negative matrix V(nxm) into the product of W(nxr) and H(rxm) 
        
        V the matrix to be factorized
        C the company matrix
        '''
        self.V = V
        self.C = C
        self.I = np.where(V>0, 1, 0)
        
        self.__prepare_non_zero_idx()
        
        W = WInit
        H = HInit
        for iter_count in range(max_iter):
            print "in iter :",iter_count,"time begin",datetime.datetime.now()
            self.computing_W = True
            W = self.__compute_argmin_matrix_for_f_wh(W, H)
            self.computing_W = False
            H = self.__compute_argmin_matrix_for_f_wh(W, H)
        return (W, H)
    
    def __prepare_non_zero_idx(self):
        self.I_non_zero_row_idx = []
        self.I_non_zero_col_by_row = []
        for i in range(self.I.shape[0]):
            if np.all(self.I[i] == 0):
                continue
            self.I_non_zero_row_idx.append(i)
            self.I_non_zero_col_by_row.append(np.where(self.I[i])[0])
         
        self.I_non_zero_col_idx = []
        self.I_non_zero_row_by_col = []
        for i in range(self.I.shape[1]):
            if np.all(self.I[:,i] == 0):
                continue
            self.I_non_zero_col_idx.append(i)
            self.I_non_zero_row_by_col.append(np.where(self.I[:,i])[0])
        
    def __compute_argmin_matrix_for_f_wh(self, W, H):
        '''
        Fix W(or H), compute matrix H(or W) which minimize f(W,H) := 1/2 ||V-WH||^2
        '''
        if self.computing_W:
            print "*** in computing W ***"
            X_old = W
        else:
            print "*** in computing H ***"
            X_old = H
        grad_f_old = self.__grad_f(W, H)
        print "grad_f_old mean:",grad_f_old.mean(),"norm:",norm(grad_f_old)
        f_x_old = self.__f(W, H)
        print "f(x_old)=",f_x_old
        alpha = 1
        beta=.1
        X_new = self.__p(X_old - alpha * grad_f_old)
        
        iter_count, max_iter = 0, 5
        
        if f_x_old > self.__f(*((X_new, H) if self.computing_W else (W, X_new))):
            # increase step loop
            print "*** in increasing loop ***"
            while True:
                alpha = alpha / beta
                X_new_tmp = X_new
                X_new = self.__p(X_old - alpha * grad_f_old)
                f_x_new = self.__f(*((X_new, H) if self.computing_W else (W, X_new)))
                print "f(x) new:",f_x_new
                need_continue = f_x_old > f_x_new
                iter_count += 1
                if not need_continue:
                    print " alpha:",alpha," iter_count:",iter_count,\
                    "\nX_new_mean:",X_new.mean(),", diff:",norm(X_new - X_old),\
                    "\n*** exit step increasing loop ***"
                    return X_new_tmp
                elif iter_count == max_iter:
                    print " alpha:",alpha," iter_count:",iter_count,\
                    "\nX_new_mean:",X_new.mean(),", diff:",norm(X_new - X_old),\
                    "\n*** exit step increasing loop ***"
                    return X_new
                    break
            return X_new_tmp
        else:
            # decrease step loop
            print "*** in decreasing loop ***"
            while True:
                alpha = alpha * beta
                X_new = self.__p(X_old - alpha * grad_f_old)
                f_x_new = self.__f(*((X_new, H) if self.computing_W else (W, X_new)))
                print "f(x) new:",f_x_new
                need_continue = f_x_old > f_x_new
                iter_count += 1
                if need_continue:
                    print " alpha:",alpha," iter_count:",iter_count, \
                    "\nX_new_mean:",X_new.mean(),", diff:",norm(X_new - X_old),\
                    "\n*** exit step decreasing loop,","computing W ***" if self.computing_W else "computingH ***",
                    return X_new
                elif iter_count == max_iter:
                    print " keep ","W" if self.computing_W else "H","the same.","\n*** exit step decreasing loop ***"
                    return X_old
        
    def __p(self, x):        
        return np.where(x < 0, 0, x)
    
    def __f(self, W, H):
        '''
        Definition of f(x) = 1/2 ||(v-Mx)*nni||^2, (v1,v2)*(v3,v4)=(v1*v2,v3*v4)
        '''
        return .5 * norm((self.V-np.dot(W,H))*self.I)**2
    
    def __grad_f(self, W, H):
        '''
        Compute the gradient of f(x) := 1/2 ||(v-Mx)*nni||^2, (v1,v2)*(v3,v4)=(v1*v2,v3*v4)
        '''
        timebegin = datetime.datetime.now()
        print "cal grad time begin:",timebegin
        V = self.V
        I = self.I
        C = self.C
        if self.computing_W:
            grad = np.zeros(W.shape)
            for i,cols in zip(self.I_non_zero_row_idx,self.I_non_zero_col_by_row):
                for j in range(grad.shape[1]):
                    grad[i,j] = ((np.dot(W[i],H[:,cols]) - V[i,cols])*H[j,cols]*I[i,cols]).sum()
        else:
            grad = np.zeros(H.shape)
            for j,rows in zip(self.I_non_zero_col_idx, self.I_non_zero_row_by_col):
                for i in range(grad.shape[0]):
                    grad[i,j] = ((np.dot(W[rows,:],H[:,j])-V[rows,j])*W[rows,i]*I[rows,j]).sum()
        timeend = datetime.datetime.now()
        print "cal grad time end:",timeend
        print "cal grad time cost:",(timeend-timebegin).total_seconds()/60.," min"
        return grad
        
        