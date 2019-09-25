# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 00:37:37 2019

@author: anand
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import parameters
rc('text',usetex = True)

##### Reading the saved MSE values from the pickle file
if os.path.isfile('/temp/mse_om_d_N.pkl'):
    output = open('temp/mse_om_d_N.pkl', 'rb')
    mse_om_d_N = pickle.load(output)
    output.close()

if os.path.isfile('/temp/mse_coiif_d_N.pkl'):
    output = open('temp/mse_coif_d_N.pkl','rb')
    mse_coif_d_N = pickle.load(output)
    output.close()

##### Obtaining the minimum average MSE obtained from the various trials
min_mse_om_N = {}
min_mse_coif_N = {}
for d in parameters.d_values:
    min_mse_om_N[d] = []
    min_mse_coif_N[d] =[]
    for N in parameters.N_values:
        if 'mse_om_d_N' in locals():
            min_mse_om_N[d].append(np.min(np.mean(mse_om_d_N[d][N],axis =0)))
        if 'mse_coif_d_N' in locals():
            min_mse_coif_N[d].append(np.min(np.mean(mse_coif_d_N[d][N],axis=0)))

#### Plotting the minimum average MSEs obtained for various values of $d$ vs $N$
fig = plt.figure(figsize = (10,5))
for d in parameters.d_values:
    if 'mse_om_d_N' in locals():
        plt.plot(parameters.N_values, min_mse_om_N[d], label = 'RKHS OM $d =$'+str(d))
    if 'mse_coif_d_N' in locals():
        plt.plot(parameters.N_values, min_mse_coif_N[d],label = 'Coif $d = $'+str(d))
plt.ylabel('Average MSEs')
plt.xlabel('$N$',size =24)
plt.legend(framealpha = 0)
plt.title('Average MSEs obtained from {} trials'.format(parameters.No_runs))
plt.show()

fig.savefig('Figure/MSEvN_best_hyperparameter.pdf')