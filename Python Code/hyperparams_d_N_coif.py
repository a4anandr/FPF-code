# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 00:05:49 2019

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

if __name__ == '__main__':
    mse_coif_d_N = collections.defaultdict(dict)
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
            mse_coif_d_N[d][N],eps,eps_best_d_N[d][N] = fpf.select_hyperparameters('coif', No_runs=No_runs, N=N, dim=d)
    
    if not os.path.isfile('input/Hyperparams_d_N_coif.csv'):
        with open('input/Hyperparams_d_N_coif.csv','w',newline='') as hyperparams_file:
            writer = csv.writer(hyperparams_file)
            writer.writerow(['d','N','epsilon'])
            for key_d,value_d in eps_best_d_N.items():
                for key_N,value_N in eps_best_d_N[key_d].items():
                    writer.writerow([key_d, key_N, value_N])
    
    ##### Saving the average MSE values obtained from 100 trials as a pickle file
    if not os.path.isfile('temp/mse_coif_d_N.pkl'):
        output = open('temp/mse_coif_d_N.pkl','wb')
        pickle.dump(mse_coif_d_N,output)
        output.close()