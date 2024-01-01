import numpy as np
import sys
import time
from Util import *
from CLT_class import CLT

class MIXTURE_CLT():
    
    def __init__(self):
        self.n_components = 0 # number of components
        self.mixture_probs = None # mixture probabilities
        self.clt_list =[]   # List of Tree Bayesian networks

    def initialize_randomly(self, dataset, n_components):
        # Randomly initialize the Chow-Liu Trees and mixture probabilities
        self.n_components = n_components

        # Randomly initialize mixture probabilities using a Dirichlet distribution
        self.mixture_probs = np.random.dirichlet(np.ones(n_components))
        
        # Initialize CLT list
        self.clt_list = [CLT() for _ in range(n_components)]

        subset_size = min(len(dataset), 1000)  # Adjust the subset size as needed
        for i in range(n_components):
            subset = dataset[np.random.choice(len(dataset), subset_size, replace=False)]
            self.clt_list[i].learn(subset)

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''
    def learn(self, train_dataset, val_dataset, n_components, max_iter=50, epsilon=1e-5):
        weights=np.zeros((n_components,train_dataset.shape[0]))

        # Randomly initialize the chow-liu trees and the mixture probabilities
        # Your code for random initialization goes here

        print("Learning for k = ",n_components)
        self.initialize_randomly(train_dataset, n_components)

        for itr in range(max_iter):
            #E-step: Complete the dataset to yield a weighted dataset
            
            #Your code for E-step here
            train_weights = self.e_step(train_dataset,weights)
            print("done E")
            
            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            
            #Your code for M-Step here
            self.m_step(train_dataset, train_weights)
            print("done M")

            weights = train_weights

            ll = self.computeLL(train_dataset)
            print(f"Iteration {itr + 1}, Log Likelihood: {ll}")

            # Check for convergence
            if itr > 0 and np.abs(ll - prev_ll) < epsilon:
                print("Converged!")
                break

            prev_ll = ll
       
    def e_step(self, dataset,weights):        

        for c in range(self.n_components):
            clt = self.clt_list[c]
            weights[c, :] = self.mixture_probs[c] * clt.getProb(dataset.T)

        # Normalize weights
        weights /= np.sum(weights, axis=0)

        return weights

    def m_step(self, dataset, weights):
        for c in range(self.n_components):
            clt = self.clt_list[c]
            clt.update(dataset, weights[c, :])

        self.mixture_probs = np.sum(weights, axis=1) / dataset.shape[0]


    """
        Compute the log-likelihood score of the dataset
    """
    def computeLL(self, dataset):
        ll = 0.0
        # Write your code below to compute likelihood of data
        #   Hint:   Likelihood of a data point "x" is sum_{c} P(c) T(x|c)
        #           where P(c) is mixture_prob of cth component and T(x|c) is the probability w.r.t. chow-liu tree at c
        #           To compute T(x|c) you can use the function given in class CLT
        
        for i in range(dataset.shape[0]): 
            prob_x = 0.0           
            for c in range(self.n_components):
                clt = self.clt_list[c]
                prob_x += self.mixture_probs[c] * clt.getProb(dataset[i, :])
            ll += np.log(prob_x)

        ll /= dataset.shape[0]
        return ll
    

    def find_optimal_k(self, train_dataset,val_dataset, k_values):

        best_k = 0
        best_val_ll = float('-inf')

        for k in k_values:
            mix_clt = MIXTURE_CLT()
            mix_clt.learn(train_dataset, val_dataset, k)
            val_ll = mix_clt.computeLL(val_dataset)

            print(f"Validation Log Likelihood for k={k}: {val_ll}")

            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_k = k

        return best_k

    def run_test_set(self, train_dataset, test_dataset, k, num_runs=5):
        results = []

        for _ in range(num_runs):
            mix_clt = MIXTURE_CLT()
            mix_clt.learn(train_dataset, test_dataset, n_components=k)
            test_ll = mix_clt.computeLL(test_dataset)
            results.append(test_ll)
            print("Result : ",test_ll)

        return results

    
'''
    After you implement the functions learn and computeLL, you can learn a mixture of trees using
    To learn Chow-Liu trees, you can use
    mix_clt=MIXTURE_CLT()
    ncomponents=10 #number of components
    max_iter=50 #max number of iterations for EM
    epsilon=1e-1 #converge if the difference in the log-likelihods between two iterations is smaller 1e-1
    dataset=Util.load_dataset(path-of-the-file)
    mix_clt.learn(dataset,ncomponents,max_iter,epsilon)
    
    To compute average log likelihood of a dataset w.r.t. the mixture, you can use
    mix_clt.computeLL(dataset)/dataset.shape[0]
'''

    