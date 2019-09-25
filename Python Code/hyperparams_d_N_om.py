# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:59:20 2019

@author: anand
"""
import numpy as np
import collections
import csv
import pickle
import os

import fpf_module as fpf
import parameters

from IPython.display import clear_output

## Main code
### Hyperparameter selection
#### Hyperparameter selection for various values of $d$ and $N$ for RKHS OM method
# Iterating over each $d$ and $N$ and obtaining the best choice of $\epsilon$ and $\lambda$ for each
# combination

if __name__ == '__main__':
    
    mse_om_d_N = collections.defaultdict(dict)
    Lambda_best_d_N = collections.defaultdict(dict)
    eps_best_d_N =collections.defaultdict(dict)
    
    d_values = input('Input the range of d -')
    if d_values == '':
        d_values = parameters.d_values
    else:
        d_values = [int(d) for d in d_values.split(',')]
    
    N_values = input('Input the range of N -')
    if N_values == '':
        N_values = parameters.N_values
    else:
        N_values = [int(N) for N in N_values.split(',')]
        
    No_runs = input('Input the number of trials -')
    if No_runs == '':
        No_runs = parameters.No_runs
    else:
        No_runs = int(No_runs)
        
    for d in d_values:
        for N in N_values:
            mse_om_d_N[d][N],Lambda,eps,Lambda_best_d_N[d][N],eps_best_d_N[d][N] = fpf.select_hyperparameters('om', No_runs=No_runs, N=N, dim=d)
    
    # Constructing a dictionary with keys as $d$ and $N$ and values as the best $\epsilon$ and $\lambda$ 
    hyperparams_d_N_om = [eps_best_d_N, Lambda_best_d_N]
    hyperparams_om = collections.defaultdict(dict)
    for key1 in eps_best_d_N.keys():
        for key2 in eps_best_d_N[key1].keys():
            hyperparams_om[key1][key2] = tuple(hyp[key1][key2] for hyp in hyperparams_d_N_om)
    
    # Writing the dictionary to a csv file
    if not os.path.isfile('input/Hyperparams_d_N_om.csv'):
        with open('input/Hyperparams_d_N.csv','w',newline='') as hyperparams_file:
            writer = csv.writer(hyperparams_file)
            writer.writerow(['d','N','epsilon','lambda'])
            for key_d,value_d in hyperparams_om.items():
                for key_N,value_N in hyperparams_om[key_d].items():
                    writer.writerow([key_d, key_N, value_N[0], value_N[1]])
    
    ##### Saving the average MSE values obtained from 100 trials as a pickle file
    if not os.path.isfile('temp/mse_om_d_N_{}.pkl'.format(No_runs)):
        output = open('temp/mse_om_d_N_{}.pkl'.format(No_runs),'wb')
        pickle.dump(mse_om_d_N,output)
        output.close()