'''
Created on Feb 11, 2015

@author: tengmf
'''
from numpy.linalg import norm
import numpy as np
import os.path as op
import datetime
import os

class NMF3(object):
    '''
    NMF with regularization
    '''


    def __init__(self, log_path=op.join(".", "my_nmf.log")):
        '''
        Constructor
        '''
        if(op.exists(log_path)):
            os.remove(log_path)
        
        
    def factorize(self, V, C, WInit=None, HInit=None, _lambda=1, max_iter=5):
        '''
        Factorize a non-negative matrix V(nxm) into the product of W(nxr) and H(rxm) 
        
        V the matrix to be factorized
        C the company matrix
        '''
        self.V = V
        self.C = C
        self.I = np.where(V > 0, 1, 0)
        self._lambda = _lambda
                
        W = WInit
        H = HInit
        for iter_count in range(max_iter):
            print "in iter :", iter_count, "time begin", datetime.datetime.now()
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
            print "*** in computing W ***"
            X_old = W
        else:
            print "*** in computing H ***"
            X_old = H
        grad_f_old = self.__grad_f(W, H)
        print "grad_f_old mean:", grad_f_old.mean(), "norm:", norm(grad_f_old)
        f_x_old = self.__f(W, H)
        print "f(x_old)=", f_x_old
        alpha = 1
        beta = .1
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
                print "f(x) new:", f_x_new
                need_continue = f_x_old > f_x_new
                iter_count += 1
                if not need_continue:
                    print " alpha:", alpha, " iter_count:", iter_count, \
                    "\nX_new_mean:", X_new.mean(), ", diff:", norm(X_new - X_old), \
                    "\n*** exit step increasing loop ***"
                    return X_new_tmp
                elif iter_count == max_iter:
                    print " alpha:", alpha, " iter_count:", iter_count, \
                    "\nX_new_mean:", X_new.mean(), ", diff:", norm(X_new - X_old), \
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
                print "f(x) new:", f_x_new
                need_continue = f_x_old > f_x_new
                iter_count += 1
                if need_continue:
                    print " alpha:", alpha, " iter_count:", iter_count, \
                    "\nX_new_mean:", X_new.mean(), ", diff:", norm(X_new - X_old), \
                    "\n*** exit step decreasing loop,", "computing W ***" if self.computing_W else "computingH ***",
                    return X_new
                elif iter_count == max_iter:
                    print " keep ", "W" if self.computing_W else "H", "the same.", "\n*** exit step decreasing loop ***"
                    return X_old
        
    def __p(self, x):        
        return np.where(x < 0, 0, x)
    
    def __f(self, W, H):
        '''
        Definition of f(x) = 1/2 ||(v-Mx)*nni||^2, (v1,v2)*(v3,v4)=(v1*v2,v3*v4)
        '''
        return .5 * norm((self.V - np.dot(W, H)) * self.I) ** 2 + \
            self._lambda * np.dot(np.dot(W.T, self.C), W).trace() 
    
    def __grad_f(self, W, H):
        '''
        Compute the gradient of f(x) := 1/2 ||(v-Mx)*nni||^2, (v1,v2)*(v3,v4)=(v1*v2,v3*v4)
        '''
        timebegin = datetime.datetime.now()
#         print "cal grad time begin:", timebegin
        V = self.V
        I = self.I
        C = self.C
        _lambda = self._lambda
        WH = np.dot(W,H)

        if self.computing_W:            
            grad = np.dot((V-WH)*I, -H.T) + _lambda*np.dot(C+C.T, W)
        else:
            grad = -W.T.dot((V-WH)*I)
#         timeend = datetime.datetime.now()
        
#         print "cal grad time end:", timeend
#         print "cal grad time cost:", (timeend - timebegin).total_seconds() / 60., " min"
        print "grad norm", norm(grad)
        return grad
    
