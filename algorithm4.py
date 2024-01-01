# Name   : Ashwin Sai C
# Course : ML - CS6375-003
# Title  : Mini Project 4 Part - 2
# Term   : Fall 2023

from CLT_class import CLT
from Util import *
from MIXTURE_CLT_org import MIXTURE_CLT
from RandomForest import RandomForest
import numpy as np
import sys
import time

def learn_chow_liu(train_file, valid_file, test_file):
	
	train_dataset=Util.load_dataset(train_file)
	valid_dataset=Util.load_dataset(valid_file)
	test_dataset=Util.load_dataset(test_file)

	concat_train_valid = np.concatenate((train_dataset,valid_dataset))
	clt=CLT()
	clt.learn(concat_train_valid)
	LL = clt.computeLL(test_dataset)/test_dataset.shape[0]

	return LL

def tree_bayesian_newtorks():
	file_name_list = ['accidents','baudio','bnetflix','jester','kdd','msnbc','nltcs','plants','pumsb_star','tretail']
	file_path      = '../datasets/dataset/'

	for name in file_name_list:
		train_file = name+'.ts.data'
		val_file   = name+'.valid.data'
		test_file  = name+'.test.data'

		LL = learn_chow_liu(file_path+train_file, file_path+val_file, file_path+test_file)
		print("Log-Liklihood for ",test_file," : ",LL)

def mixture_tree_bayesian_networks_EM():
	file_name_list = ['accidents','baudio','bnetflix','jester','kdd','msnbc','nltcs','plants','pumsb_star','tretail']
	# file_name_list = ['accidents','baudio','bnetflix','jester']
	file_path      = '../datasets/dataset/'
	k_list         = [2,5,10,15,20]

	for index,name in enumerate(file_name_list):
		train_file = name+'.ts.data'
		val_file   = name+'.valid.data'
		test_file  = name+'.test.data'

		train_file_path = file_path+train_file
		val_file_path   = file_path+val_file
		test_file_path  = file_path+test_file

		train_dataset   = Util.load_dataset(train_file_path)
		val_dataset     = Util.load_dataset(val_file_path)
		test_dataset    = Util.load_dataset(test_file_path)	    
		
		print("\nFile name : ",test_file)

		mix_clt = MIXTURE_CLT()
		# optimal_k = mix_clt.find_optimal_k(train_dataset,val_dataset, k_list)
		# print(f"Optimal k based on validation set: {optimal_k}")

		num_runs     = 5
		# optimal_k    = 20

		optimal_k_list = [20, 20, 20, 20, 2, 2, 20, 20, 20, 20]
		# optimal_k_list = [20, 20, 20, 20]

		print(train_dataset.shape)
		print(val_dataset.shape)
		train_dataset = np.concatenate((train_dataset,val_dataset)) # creating train dataset along with val dataset
		print(train_dataset.shape)

		test_results = mix_clt.run_test_set(train_dataset, test_dataset, optimal_k_list[index], num_runs)

		average_test_ll = np.mean(test_results)
		std_dev_test_ll = np.std(test_results)

		print(f"Average Test Log Likelihood: {average_test_ll}")
		print(f"Standard Deviation of Test Log Likelihood: {std_dev_test_ll}")

def mixture_tree_bayesian_networks_RF():
	file_name_list = ['accidents','baudio','bnetflix','jester','kdd','msnbc','nltcs','plants','pumsb_star','tretail']
	file_path      = '../datasets/dataset/'
	k_list         = [2,5,10,20]
	r_list         = [5, 75, 130, 200]

	for name in file_name_list:
		train_file = name+'.ts.data'
		val_file   = name+'.valid.data'
		test_file  = name+'.test.data'

		train_file_path = file_path+train_file
		val_file_path   = file_path+val_file
		test_file_path  = file_path+test_file

		train_dataset   = Util.load_dataset(train_file_path)
		val_dataset     = Util.load_dataset(val_file_path)
		test_dataset    = Util.load_dataset(test_file_path)	    
		
		print("\nFile name k&r : ",test_file)

		rf_clt = RandomForest()
		optimal_k,optimal_r = rf_clt.find_optimal_k_r(train_dataset,val_dataset, k_list,r_list)
		print(f"Optimal k based on validation set: {optimal_k}")
		print(f"Optimal r based on validation set: {optimal_r}")

		num_runs     = 5

		print(train_dataset.shape)
		print(val_dataset.shape)
		train_dataset = np.concatenate((train_dataset,val_dataset)) # creating train dataset along with val dataset
		print(train_dataset.shape)

		test_results = rf_clt.run_test_set_k_r(train_dataset, test_dataset, optimal_k, optimal_r, num_runs)

		average_test_ll = np.mean(test_results)
		std_dev_test_ll = np.std(test_results)

		print(f"Average Test Log Likelihood: {average_test_ll}")
		print(f"Standard Deviation of Test Log Likelihood: {std_dev_test_ll}")



if __name__ == "__main__":

	print("hi")

	tree_bayesian_newtorks()
	# mixture_tree_bayesian_networks_EM()
	# mixture_tree_bayesian_networks_RF()


