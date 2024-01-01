from __future__ import print_function
import numpy as np
import sys
import time
import contextlib
from Util import *
from CLT_class import CLT
import random
import os


class RandomForest():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks

    '''
        Learn Mixtures of Trees using the RF algorithm.
    '''
    def learn(self, dataset, n_components, hyperparameterR):
        
        self.n_components = n_components
        # For RandomForest, weigts can be uniform - keeping them 1
        weights=np.ones((n_components,dataset.shape[0]), dtype=float)
        self.mixture_probs = 1/n_components

        self.clt_list = [CLT() for i in range(n_components)]
        
        for k in range(n_components):
            self.clt_list[k].Bootstrap_learn(dataset, weights[k], hyperparameterR)

    '''
        Compute the log-likelihood score of the dataset
    '''
    
    def computeLL(self, dataset):
        
        ll = 0.0
        for k in range(self.n_components):
            ll += self.clt_list[k].computeLL(dataset)*self.mixture_probs
        return ll/dataset.shape[0]

    def find_optimal_k_r(self, train_dataset,val_dataset, k_list,r_list):

        best_k = 0
        best_r = 0

        best_val_ll = float('-inf')

        for k in k_list:
            for r in r_list:
                rf_clt = RandomForest()
                rf_clt.learn(train_dataset, k, r)
                val_ll = rf_clt.computeLL(val_dataset)

                print(f"Validation Log Likelihood for k={k} & r={r} : {val_ll}") 

                if val_ll > best_val_ll:
                    best_val_ll = val_ll
                    best_k = k
                    best_r = r

        return [best_k,best_r]

    def run_test_set_k_r(self, train_dataset, test_dataset, k, r, num_runs=5):
        results = []

        for _ in range(num_runs):
            rf_clt = RandomForest()
            rf_clt.learn(train_dataset, k, r)
            test_ll = rf_clt.computeLL(test_dataset)
            results.append(test_ll)
            print("Result k&r : ",test_ll)

        return results