'''
Created on Feb 11, 2015

@author: tengmf
'''
# from builtins import print
from numpy.linalg import norm
import numpy as np
from collections import namedtuple
from scipy.optimize import minimize
from numpy.linalg import norm
from src.validate.measure import Measure
from pandas import DataFrame
from sys import exit

# logger = mylog.getLogger(__name__, 'DEBUG')
# logger.addHandler(mylog.getLogHandler('INFO', 'CONSOLE'))
# logger.addHandler(mylog.getLogHandler('DEBUG', 'FILE'))

_P = namedtuple('_Parameters',
                'V C k lambda_ sigma sigma_a sigma_b eta theta I diagnol_col_idxes non_diagnol_col_idxes W_shape H_shape')


def _reshape(W, H, parameters):
    return W.reshape(parameters.W_shape), H.reshape(parameters.H_shape)


def _f(W, H, parameters):
    '''
    W
    H
    P: parameters
    '''
    W, H = _reshape(W, H, parameters)
    P = parameters
    WH_diagnol = np.dot(W, H[:, P.diagnol_col_idxes])
    # logger.debug("norm W %f, norm H %f" % (norm(W), norm(H)))
    return .5 * norm(
        (P.V[:, P.non_diagnol_col_idxes] - np.dot(W, H[:, P.non_diagnol_col_idxes])) * P.I[:,
                                                                                       P.non_diagnol_col_idxes]) ** 2 / P.sigma ** 2 - np.sum(
        -(P.V[:, P.diagnol_col_idxes] * np.log(WH_diagnol) - WH_diagnol) * P.I[:, P.diagnol_col_idxes]) + norm(
        W) ** 2 / P.sigma_a ** 2 + norm(H[:, P.non_diagnol_col_idxes]) ** 2 / P.sigma_b ** 2 - np.sum(
        (P.eta - 1) * np.log(H[:, P.diagnol_col_idxes]) - P.theta * H[:, P.diagnol_col_idxes]) + P.lambda_ * np.dot(
        np.dot(W.T, P.C), W).trace()


def __grad_f_W(W, H, parameters):
    W, H = _reshape(W, H, parameters)
    P = parameters
    WH_diagnol = np.dot(W, H[:, P.diagnol_col_idxes])
    grad = np.dot(
        (P.V[:, P.non_diagnol_col_idxes] - np.dot(W, H[:, P.non_diagnol_col_idxes])) * P.I[:, P.non_diagnol_col_idxes],
        -H[:, P.non_diagnol_col_idxes].T) / P.sigma ** 2 \
           - ((P.V[:, P.diagnol_col_idxes] * 1. / WH_diagnol - 1) * P.I[:, P.diagnol_col_idxes]).dot(
        H[:, P.diagnol_col_idxes].T) \
           + P.lambda_ * np.dot(P.C + P.C.T, W) \
           + 1 / P.sigma_a ** 2 * W
    return grad.reshape(np.product(grad.shape))


def __grad_f_H(W, H, parameters):
    W, H = _reshape(W, H, parameters)
    P = parameters
    WH_diagnol = np.dot(W, H[:, P.diagnol_col_idxes])
    grad = np.zeros(H.shape)
    grad[:, P.diagnol_col_idxes] = - W.T.dot(
        (P.V[:, P.diagnol_col_idxes] / WH_diagnol - 1) * P.I[:, P.diagnol_col_idxes]) \
                                   - (P.eta - 1) / H[:, P.diagnol_col_idxes] + P.theta

    grad[:, P.non_diagnol_col_idxes] = -W.T.dot(
        (P.V[:, P.non_diagnol_col_idxes] - np.dot(W, H[:, P.non_diagnol_col_idxes])) * P.I[:,
                                                                                       P.non_diagnol_col_idxes]) / P.sigma ** 2 + 1 / P.sigma_b ** 2 * H[
                                                                                                                                                       :,
                                                                                                                                                       P.non_diagnol_col_idxes]
    return grad.reshape(np.product(grad.shape))


def nmf6(V, C, k=10, lambda_=1, sigma_a=1e-2, sigma_b=1e-2, eta=1, theta=1, max_iter=1, WInit=None,
         HInit=None):
    # initialize
    W = WInit if WInit is not None else np.random.uniform(1e-3, 1, (V.shape[0], k))
    H = HInit if HInit is not None else np.random.uniform(1e-3, 1, (k, V.shape[1]))

    r = int(np.sqrt(V.shape[1]))
    diagnol_col_idxes = [i * r + i for i in range(r)]
    non_diagnol_col_idxes = np.setdiff1d(range(r ** 2), diagnol_col_idxes)
    sigma = np.std(V[np.nonzero(V)])
    parameters = _P(V, C, k, lambda_, sigma, sigma_a, sigma_b, eta, theta, np.where(V > 0, 1, 0), diagnol_col_idxes,
                    non_diagnol_col_idxes, W.shape, H.shape)

    print('W shape (%d,%d)' % W.shape)
    print('H shape (%d,%d)' % H.shape)

    n_W = np.product(W.shape)
    n_H = np.product(H.shape)
    W = W.reshape(n_W)
    H = H.reshape(n_H)
    bounds_W = [(1e-5, 9999)] * (n_W)
    bounds_H = [(1e-5, 9999)] * (n_H)

    for iter_count in range(max_iter):
        # logger.debug("in iter : %d begin at %r" % (iter_count, datetime.datetime.now()))
        print('*** in iter %d ***' % iter_count)
        result1 = minimize(lambda X: _f(X, H, parameters), W, jac=lambda X: __grad_f_W(X, H, parameters),
                          bounds=bounds_W, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-4, 'disp': True})
        # result = minimize(lambda X: _f(X, H, parameters), W, jac=False,
        #                  bounds=bounds_W, method='L-BFGS-B')

        W = result1.x
        print('solve W : %s %s\nF=%f' % ("OK" if result1.success else "Fail", result1.message, result1.fun))

        result2 = minimize(lambda X: _f(W, X, parameters), H, jac=lambda X: __grad_f_H(W, X, parameters),
                          bounds=bounds_H, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-4, 'disp': True})
        H = result2.x
        print('solve H : %s %s\nF=%f' % ("OK" if result2.success else "Fail", result2.message, result2.fun))
        #W, H = W, H_tmp
    #return (W, H, result.fun, norm(result.jac))
    return list(_reshape(W, H, parameters))+[result1.success, result2.success, result2.fun, norm(result2.jac)]



if __name__ == '__main__':
    import os.path as op
    from time import time

    base_path = op.join("..", "..", "output")
    data = 'data2'
    V = np.loadtxt(op.join(base_path, data, "validate", "training", "pure_matrix.csv"), delimiter=",")
    C = np.loadtxt(op.join(base_path, data, "validate", "training", "company_matrix"))
    ks = np.arange(10,60,10)#5
    lambda_s = [10 ** i for i in np.arange(-5, -2)]#3
    sigma_as = [10 ** i for i in np.arange(-3, 3)]#6
    sigma_bs = [10 ** i for i in np.arange(-3, 3)]#6
    etas = [2 ** i for i in np.arange(0, 6)]#6
    thetas = [10 ** i for i in np.arange(-3, 3)]#6

    R_test = np.loadtxt(op.join(base_path, data, "validate", "test", "pure_matrix.csv"), delimiter=",")
    result = DataFrame(columns=["Data Set", "lambda", "sigma A", "sigma B", "eta", "theta", "k", "W converge", "H converge", "W Max", "H Max","f", "jac norm"] \
                               + [i + str(j) for i in ["NDCG@", "Precision@", "Recall@"] for j in [3, 5, 10]])
    idx = 0

    start = time()
    for k in ks:
        WInit = np.random.uniform(1, 2, (V.shape[0], k))
        HInit = np.random.uniform(1, 2, (k, V.shape[1]))
        for lambda_ in lambda_s:
            for sigma_a in sigma_as:
                for sigma_b in sigma_bs:
                    for eta in etas:
                        for theta in thetas:
                            
                            W, H, W_success, H_success,obj_val,jac_norm = nmf6(V, C, 10, lambda_, sigma_a, sigma_b, eta, theta, WInit=WInit, HInit=HInit)

                            WH = W.dot(H)
                            m = Measure(WH, R_test)
                            precision_recall = list(zip(*[m.precision_recall(i) for i in [3, 5, 10]]))
                            result.loc[idx] = [data, lambda_, sigma_a, sigma_b, eta, theta, k, W_success, H_success, W.max(), H.max(), obj_val, jac_norm,m.ndcg(3), m.ndcg(5),
                                               m.ndcg(10)] + list(precision_recall[0]) + list(precision_recall[1])
                            
                            
                            print('W max %.2f min   %.2f norm %.2f' % (W.max(), W.min(), norm(W)))
                            print('H max %.2f min %.2f norm %.2f' % (H.max(), H.min(), norm(H)))
                            np.save('comp_recommend_sys/output/nmf6measure/W%d'%idx,W)
                            np.save('comp_recommend_sys/output/nmf6measure/H%d'%idx,H)                                
                            idx += 1
                            result.to_csv('comp_recommend_sys/output/nmf6measure/nmf_measure_result.csv')
    print('\n%.2f seconds cost\n' % (time() - start))