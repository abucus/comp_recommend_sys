'''
Created on Feb 11, 2015

@author: tengmf
'''
from numpy.linalg import norm
import numpy as np
import os.path as op
import datetime
import os
import warnings
import src.util.log as mylog

logger = mylog.getLogger(__name__, 'DEBUG')
logger.addHandler(mylog.getLogHandler('INFO', 'CONSOLE'))
logger.addHandler(mylog.getLogHandler('DEBUG', 'FILE'))

class NMF5(object):
    '''
    NMF with regularization
    '''


    def __init__(self, log_path=op.join(".", "my_nmf.log")):
        '''
        Constructor
        '''
        if(op.exists(log_path)):
            os.remove(log_path)
        self.logger = logger


    def factorize(self, V, C, k=10, _lambda=1, sigma_a=1e-2, sigma_b=1e-2,  eta = 1, theta = 1,max_iter=5, WInit=None, HInit=None):
        '''
        Factorize a non-negative matrix V(nxm) into the product of W(nxr) and H(rxm)

        V the matrix to be factorized
        C the company matrix
        '''
        warnings.filterwarnings('error')
        self.V = V
        self.C = C
        self.I = np.where(V > 0, 1, 0)
        self._lambda = _lambda
        self.sigma = np.std(V[np.nonzero(V)])
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.eta = eta
        self.theta = theta

        r = int(np.sqrt(V.shape[1]))
        self.diagnol_col_idxes = [i*r + i for i in range(r)]
        self.non_diagnol_col_idxes = np.setdiff1d(range(r**2), self.diagnol_col_idxes)

        W = WInit if WInit is not None else np.random.uniform(1, 2, (V.shape[0], k))
        H = HInit if HInit is not None else np.random.uniform(1, 2, (k, V.shape[1]))

        for iter_count in range(max_iter):
            self.logger.debug("in iter : %d begin at %r"%(iter_count, datetime.datetime.now()))
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
            self.logger.debug("*** in computing W ***")
            X_old = W
        else:
            self.logger.debug("*** in computing H ***")
            X_old = H
        grad_f_old = self.__grad_f(W, H)
        f_x_old = self.__f(W, H)
        self.logger.debug("f(x_old)=%f"%f_x_old)
        alpha = 1
        beta = .5
        X_new = self.__p(X_old - alpha * grad_f_old)

        iter_count, max_iter = 0, 5

        if f_x_old > self.__f(*((X_new, H) if self.computing_W else (W, X_new))):
            # increase step loop
            self.logger.debug("*** in increasing loop ***")
            while True:
                alpha = alpha / beta
                X_new_tmp = X_new
                X_new = self.__p(X_old - alpha * grad_f_old)
                f_x_new = self.__f(*((X_new, H) if self.computing_W else (W, X_new)))
                self.logger.debug('norm(X_new_tmp):%f, norm(X_new):%f, f_x_new:%f'%(norm(X_new_tmp), norm(X_new), f_x_new))
                need_continue = f_x_old > f_x_new
                iter_count += 1
                if not need_continue:
                    self.logger.debug("alpha %f iter_count %d X_new_mean %f diff %f \n *** exit step increasing loop ***" \
                    %(alpha, iter_count, X_new.mean(), norm(X_new - X_old)))
                    return X_new_tmp
                elif iter_count == max_iter:
                    self.logger.debug("alpha %f iter_count %d X_new_mean %f diff %f \n *** exit step increasing loop ***" \
                    %(alpha, iter_count, X_new.mean(), norm(X_new - X_old)))
                    return X_new

        else:
            # decrease step loop
            self.logger.debug("*** in decreasing loop ***")
            while True:
                alpha = alpha * beta
                X_new = self.__p(X_old - alpha * grad_f_old)
                f_x_new = self.__f(*((X_new, H) if self.computing_W else (W, X_new)))
                self.logger.debug('norm(X_new):%f, f_x_new:%f'%(norm(X_new), f_x_new))
                iter_count += 1
                if f_x_old > f_x_new:
                    self.logger.debug("alpha %f iter_count %d X_new_mean %f diff %f \n *** exit step decreasing loop ***" \
                    %(alpha, iter_count, X_new.mean(), norm(X_new - X_old)))
                    return X_new
                elif iter_count == max_iter:
                    self.logger.debug("keep %s the same. \n*** exit step decreasing loop ***"%("W" if self.computing_W else "H"))
                    return X_old

    def __p(self, x):
        default_value_4_zero = 1e-4
        return np.where(x <= 0, default_value_4_zero, x)

    def __f(self, W, H):
        '''
        Definition of f(x) = 1/2 ||(v-Mx)*nni||^2, (v1,v2)*(v3,v4)=(v1*v2,v3*v4)
        '''
        diagnol_col_idxes = self.diagnol_col_idxes
        non_diagnol_col_idxes = self.non_diagnol_col_idxes

        WH_diagnol = np.dot(W,H[:,self.diagnol_col_idxes])
        self.logger.debug("norm W %f, norm H %f"%(norm(W),norm(H)))
        return .5 * norm((self.V[:,self.non_diagnol_col_idxes] - np.dot(W, H[:,self.non_diagnol_col_idxes])) * self.I[:,self.non_diagnol_col_idxes]) ** 2 /self.sigma**2 \
        - np.sum((self.V[:,self.diagnol_col_idxes]*np.log(WH_diagnol)-WH_diagnol)*self.I[:,self.diagnol_col_idxes]) \
        + norm(W)**2/self.sigma_a**2 + norm(H[:,non_diagnol_col_idxes])**2/self.sigma_b**2 \
        - np.sum((self.eta-1)*np.log(H[:,diagnol_col_idxes])-self.theta*H[:,diagnol_col_idxes])
        + self._lambda * np.dot(np.dot(W.T, self.C), W).trace()
        

    def __grad_f(self, W, H):
        '''
        Compute the gradient of f(x) := 1/2 ||(v-Mx)*nni||^2, (v1,v2)*(v3,v4)=(v1*v2,v3*v4)
        '''
        timebegin = datetime.datetime.now()
        V = self.V
        I = self.I
        C = self.C
        _lambda = self._lambda

        diagnol_col_idxes = self.diagnol_col_idxes
        non_diagnol_col_idxes = self.non_diagnol_col_idxes
        WH_diagnol = np.dot(W,H[:,diagnol_col_idxes])

        if self.computing_W:
            grad = np.dot((V[:,non_diagnol_col_idxes]-np.dot(W,H[:,non_diagnol_col_idxes]))*I[:,non_diagnol_col_idxes], -H[:,non_diagnol_col_idxes].T)/self.sigma**2 \
            + ((V[:,diagnol_col_idxes]*1./WH_diagnol-1)*I[:,diagnol_col_idxes]).dot(H[:,diagnol_col_idxes].T) \
            + _lambda*np.dot(C+C.T, W) \
            + 1/self.sigma_a**2*W
        else:
            grad = np.zeros(H.shape)
            grad[:,diagnol_col_idxes] = W.T.dot((V[:,diagnol_col_idxes]/WH_diagnol-1)*I[:,diagnol_col_idxes]) \
            - np.sum((self.eta-1)/H[:,diagnol_col_idxes]-self.theta)

            grad[:,non_diagnol_col_idxes] = -W.T.dot((V[:,non_diagnol_col_idxes]-np.dot(W,H[:,non_diagnol_col_idxes]))*I[:,non_diagnol_col_idxes])/self.sigma**2 + 1/self.sigma_b**2*H[:,non_diagnol_col_idxes]

        self.logger.debug("grad norm %f"%norm(grad))
        return grad

