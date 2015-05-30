'''
Created on May 25, 2015

@author: tengmf
'''
import pickle
import os.path as op
import numpy as np
from src.log import get_logger
from src.lib.nmf import nmf
from src.input.read_csv import cal_interval_in_days

class PIMF(object):
    '''
    classdocs
    '''
    def __init__(self, base_path, k=None, mu=2):
        '''
        Constructor
        '''
        self.logger = get_logger(__name__)
        self.base_path = base_path 
        
        self.dp = pickle.load(open(op.join(base_path, 'dp'), 'r'))
        self.logger.info('loaded dp ')
        
        self.user_map = pickle.load(open(op.join(base_path, 'user_map'), 'r'))
        self.user_num = len(self.user_map)
        self.logger.info('loaded user map ')
        
        self.event_map = pickle.load(open(op.join(base_path, 'event_map'), 'r'))
        self.event_num = len(self.event_map)
        self.logger.info('loaded event map ')
        
        init_utility = np.loadtxt(op.join(base_path, 'utility'))
        self.logger.info('loaded init utility matrix map ')
        
        self.table = pickle.load(open(op.join(base_path, 'table'), 'r'))
        self.logger.info('loaded table')
        
        if not k:
            k = int(min(*(init_utility.shape)) / 5)
        if op.exists(op.join(base_path, 'utility_approximation')):
            self.utility = np.loadtxt(op.join(base_path, 'utility_approximation'))
            self.logger.info('utility approximation loaded')
        else:
            init_W = np.random.random_sample((init_utility.shape[0], k))
            init_H = np.random.random_sample((k, init_utility.shape[1]))
            (W, H) = nmf(init_utility, init_W, init_H, 1e-5, 1800, 50)
            self.utility = W.dot(H)
            np.savetxt(op.join(base_path, 'utility_approximation'), self.utility)
            del init_W, init_H, W, H
            self.logger.info('utility approximation computed')
        
        self.mu = mu
    
    def predict(self, u, t, k):
        '''
        return the recommendation item for user u at time t
        u the user id which is a string
        t the time stamp to predict
        k the number of predictions
        '''
        self.logger.debug('predict for u{} t{} event_num{}'.format(u, t, self.event_num))
        probs = np.zeros((self.event_num,))
        for i in xrange(self.event_num):
            probs[i] = self.__cal_purchae_prob(u, i, t)
        # return probs[-k:][::-1]
        return np.argsort(probs)[-k:][::-1]
    
    def purchae_prob(self, u, t, i):
        if type(i) == str:
            i = self.event_map[i]
        return self.__cal_purchae_prob(u, i, t)
    
    def __cal_us(self, u, i, t):
        '''
        calculate the surplus of item i for user u at time t
        u user id which is a string
        i item id which is a int
        t time stamp
        '''
        int_u = self.user_map[u]
        return self.utility[int_u, i] * (1 + self.__cal_max_pi_uij(u, i, t)) ** self.mu
    
    def __cal_max_pi_uij(self, u, i, t):
        events = self.table[u]['events']
        piuij = []
        for e in events:
            tj = e[0]
            ej = self.event_map[e[1]]
            # v = 1. / np.log2(np.abs(cal_interval_in_days(tj, t) - self.dp.get((u, ej, i), 0)) + 2)
            v = 1. / np.log2(np.abs(cal_interval_in_days(tj, t) - self.dp[self.user_map[u], ej, i] + 2))
            piuij.append(v)
        
        rlt = np.max(piuij)
        self.logger.debug('computed the max pi_uij for u:{}, i:{}, t:{}'.format(u, i, t))
        self.logger.debug('pi_uij:{}'.format(piuij))
        self.logger.debug('max:{}'.format(rlt))
        return rlt
        
    def __cal_purchae_prob(self, u, i, t):
        self.logger.debug('cal purchase prob u{} i{} t{}'.format(u, i, t))
        rlt = 1. / (1 + np.exp(-1.*self.__cal_us(u, i, t)))
        self.logger.debug('computed prob for user:{},i:{},t:{} is {}'.format(u, i, t, rlt))
        return rlt
        
