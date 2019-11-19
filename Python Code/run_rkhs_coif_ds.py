# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 01:00:38 2019

@author: anand
"""

#### MSE vs $d$ and $N$ with the best hyper parameters
##### Reading the best hyperparameter values for both RKHS OM and Coifman methods
import numpy as np
from sympy import *
import math
from scipy import spatial

import pandas as pd

import collections

import fpf_module as fpf
import parameters


import matplotlib
matplotlib.rc('text',usetex = True)
matplotlib.rc('font', **parameters.font)
matplotlib.rc('legend',fontsize = 24)
# matplotlib.rcParams.update(parameters.font_params)

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'auto')
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import os
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    plt.close('all')
    
    No_runs = parameters.No_runs
    # N_values = parameters.N_values
    d_values = parameters.d_values
    
    if os.path.isfile('input/Hyperparams_d_N_om.csv'):
            hyperparams_om = pd.read_csv('input/Hyperparams_d_N_om.csv')
            hyperparams_om_dict = collections.defaultdict(dict)
            for row in hyperparams_om.iterrows():
                hyperparams_om_dict[row[1][0]][row[1][1]] = [row[1][2],row[1][3]]
        
    if os.path.isfile('input/Hyperparams_d_N_coif.csv'):
        hyperparams_coif = pd.read_csv('input/Hyperparams_d_N_coif.csv')
        hyperparams_coif_dict = collections.defaultdict(dict)
        for row in hyperparams_coif.iterrows():
            hyperparams_coif_dict[row[1][0]][row[1][1]] = row[1][2]
 
    
    mse_finite = np.zeros((No_runs, len(d_values)))
    mse_coif = np.zeros((No_runs, len(d_values)))
    mse_rkhs_N = np.zeros((No_runs, len(d_values)))
    mse_rkhs_dN = np.zeros((No_runs, len(d_values)))
    mse_om   = np.zeros((No_runs, len(d_values)))
    mse_const = np.zeros((No_runs, len(d_values)))

    
    # System parameters
    N = 1000    # dimension of the system
    
    fig, ax = plt.subplots(nrows = 1, ncols = len(d_values), sharey = True, figsize = (21,8))
    for j,d in enumerate(d_values):  
        i = d_values.tolist().index(d)
        x = symbols('x0:%d'%d)
        c_coef = np.ones(d)# np.ones((1,d))   # [1,0]
        c =  c_coef.dot(x)      # Observation function
        c_x = lambdify(x, c, 'numpy')
        p_b = 0
        for m in range(len(parameters.w_b)):
            p_b = p_b + parameters.w_b[m] * (1/ np.sqrt(2 * math.pi * parameters.sigma_b[m]**2))* exp(-(x[0] - parameters.mu_b[m])**2/ (2* parameters.sigma_b[m]**2))
        p_b_x = lambdify(x[0],p_b, 'numpy')
        gm = d     # No. of dimensions with Gaussian mixture densities in the dim-dimensional density, should be <= d
        
        for run in range(No_runs):
            print('Dimensions ', d)
            print('No. particles ',N)
            print('Run ',run)
            if No_runs > 1:
                seed = np.random.randint(1,500)
            else:
                seed = parameters.seed # np.random.randint(1,500)
            print('Seed ', seed)
            Xi  = fpf.get_samples(N, parameters.mu_b, parameters.sigma_b, parameters.w_b, d, gm, parameters.sigma, seed = seed)
#            if d == 1:
#                Xi = np.sort(Xi,kind = 'mergesort')

            C = np.reshape(c_x(*Xi.T),(len(Xi),1))
                                  
            if parameters.exact == 1:
                K_exact = np.zeros((N, d))
#                K_exact  = fpf.gain_num_integrate(Xi, (c_coef[0] * x[0]), p_b, x, 0)
#                K_exact = K_exact.reshape((-1,1))
                for d_i in np.arange(gm):
                    K_exact[:,d_i]  = fpf.gain_num_integrate(Xi, (c_coef[d_i] * x[0]), p_b, x, d_i)
                ax[j].plot(Xi[:,0], K_exact[:,0], '^', label = 'Exact')
                ax[j].set_xlabel('$x$')
             
            if parameters.finite == 1:
                if d == 1:
                    if parameters.method == 'integration':
                        X, K_finite_X = fpf.gain_finite_integrate(c_x, parameters.w_b, parameters.mu_b, parameters.sigma_b, parameters.basis_dim, parameters.basis, parameters.affine,  diag = 0)
                        knn = spatial.KDTree(X.reshape((len(X),1)))
                        K_finite = K_finite_X[knn.query(Xi)[1]].reshape(N,1)
                    else:
                        K_finite  = fpf.gain_finite(Xi, C, parameters.mu_b, parameters.sigma_b, parameters.basis_dim, parameters.basis, parameters.affine, diag =0)
                    ax[j].plot(Xi, K_finite, '*', label = r'Polynomial$\times \rho_i$')
                    if parameters.exact == 1:
                        mse_finite[run,i] = fpf.mean_squared_error(K_exact, K_finite)
                        
            if parameters.coif ==1:
                eps_coif = hyperparams_coif_dict[d][N] if d in hyperparams_coif_dict and N in hyperparams_coif_dict[d] else 1
                Phi = np.zeros(N)
                K_coif,Phi = fpf.gain_coif(Xi, C, eps_coif, Phi, diag = 0)
                ax[j].plot(Xi[:,0], K_coif[:,0],'>', label = 'Markov kernel')
                if parameters.exact == 1:
                    mse_coif[run,i] = fpf.mean_squared_error(K_exact, K_coif)

            if parameters.rkhs_N == 1:
                eps_rkhs_N =hyperparams_om_dict[d][N][0] if d in hyperparams_om_dict and N in hyperparams_om_dict[d] else 1
                Lambda_rkhs_N = hyperparams_om_dict[d][N][1] if d in hyperparams_om_dict and N in hyperparams_om_dict[d] else 1e-2
                K_rkhs_N, beta_N = fpf.gain_rkhs_N(Xi, C, eps_rkhs_N, Lambda_rkhs_N, diag = 0)
                ax[j].plot(Xi[:,0], K_rkhs_N[:,0],'o', label = r'$\nabla$-LSTD-RKHS-N')
                if parameters.exact == 1:
                    mse_rkhs_N[run,i] = fpf.mean_squared_error(K_exact, K_rkhs_N)

            if parameters.rkhs_dN == 1:
                eps_rkhs_dN = hyperparams_om_dict[d][N][0] if d in hyperparams_om_dict and N in hyperparams_om_dict[d] else 1
                Lambda_rkhs_dN = hyperparams_om_dict[d][N][1] if d in hyperparams_om_dict and N in hyperparams_om_dict[d] else 1e-2
                K_rkhs_dN, beta_dN = fpf.gain_rkhs_dN(Xi, C, eps_rkhs_dN, Lambda_rkhs_dN, diag = 0)
                ax[j].plot(Xi[:,0], K_rkhs_dN[:,0],'+', label = r'$\nabla$-LSTD-RKHS-Opt'.format(d+1))
                if parameters.exact == 1:
                    mse_rkhs_dN[run,i] = fpf.mean_squared_error(K_exact, K_rkhs_dN)

            if parameters.om == 1:
                eps_om = hyperparams_om_dict[d][N][0] if d in hyperparams_om_dict and N in hyperparams_om_dict[d] else 1
                Lambda_om = hyperparams_om_dict[d][N][1] if d in hyperparams_om_dict and N in hyperparams_om_dict[d] else 1e-2
                K_om, beta_om = fpf.gain_rkhs_om(Xi, C, eps_om, Lambda_om, diag = 0)
                ax[j].plot(Xi[:,0], K_om[:,0],'>', label = r'$\nabla$-LSTD-RKHS-OM')
                if parameters.exact == 1:
                    mse_om[run,i] = fpf.mean_squared_error(K_exact, K_om)
            
            if parameters.const == 1:
                eta = np.mean(C)
                Y = (C -eta)
                K_const = np.mean(Y * Xi, axis = 0)
                ax[j].axhline(y= K_const, color = 'k', linestyle = '--', label ='Const gain')
                if parameters.exact == 1:
                    mse_const[run,i] = fpf.mean_squared_error(K_exact, K_const)  
                    
            ax[j].set_ylim(-0.5,10)
            ax[j].text(0.05, 0.95, '$N= {}$'.format(N), fontsize = 24, transform = ax[j].transAxes, verticalalignment='top')
            ax[j].text(0.05, 0.90, '$d= {}$'.format(d), fontsize = 24, transform = ax[j].transAxes, verticalalignment='top')
            ax[j].tick_params(labelsize = 24)

            if j == 0:
                ax[j].legend(loc = 1, framealpha = 0, fontsize = 24)
                ax[j].set_ylabel('$\sf K_1(x)$', fontsize  = 24)
            
            ax2 = ax[j].twinx()
            sns.distplot(Xi[:,0], label = 'Hist. of $x^i$')
            ax2.tick_params(labelsize = 24)
            
            domain = np.arange(-3,3,0.1)
            ax2.plot(domain,p_b_x(domain), 'k',label = '$\\rho_1(x)$')
            ax2.set_ylim(0,1)
            if j == 2:
                ax2.set_ylabel('$\\rho_1(x)$', fontsize = 24)
            
    ax2.legend(loc = 1, framealpha = 0, fontsize = 24)
        
    fig2 = plt.figure(figsize = (10,6))
    if parameters.rkhs_N == 1:
        plt.plot(Xi, beta_N, '*',label = r'$\beta^\circ_i$')
    if parameters.rkhs_dN == 1:
        plt.plot(Xi, beta_dN.reshape((N,d+1))[:,0], 'o', label = r'$\beta^{0*}_i$')
        plt.plot(Xi, beta_dN.reshape((N,d+1))[:,1], 'o', label = r'$\beta^{1*}_i$')
    if parameters.om == 1:
        plt.plot(Xi, beta_om[0:N], '+', label = r'$\beta^{OM}_i$')
    plt.ylim(-50,50)
    plt.legend(loc = 1 , framealpha = 0, ncol = 4)
    plt.xlabel('$x$')
    plt.ylabel(r'$\log_{10}\beta_i$')
    plt.title(r'Parameters $\{\beta_i\}$ from RKHS simplified, optimal and OM methods')
    
    # fig2.savefig('Figure/Chap4_beta_comparison.pdf',bbox_inches='tight')
    fig.savefig('Figure/Chap4_gain_om_d2510.pdf',bbox_inches='tight')

                    

 
   