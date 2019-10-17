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

import fpf_module as fpf
import parameters


import matplotlib
matplotlib.rc('text',usetex = True)
# matplotlib.rc('font', **parameters.font)
matplotlib.rcParams.update(parameters.font_params)

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'auto')
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    plt.close('all')
    
    No_runs = parameters.No_runs
    N_values = parameters.N_values
    T_values = parameters.T_values
    
    mse_finite = np.zeros((No_runs,len(parameters.N_values)))
    mse_diff_td = np.zeros((No_runs,len(parameters.N_values)))
    mse_diff_nl_td = np.zeros((No_runs,len(parameters.N_values)))
    mse_const = np.zeros((No_runs, len(parameters.N_values)))
    
    # System parameters
    d = 1    # dimension of the system
    i = parameters.d_values.tolist().index(d)
    x = symbols('x0:%d'%d)
    c_coef = np.ones(d)# np.ones((1,d))   # [1,0]
    c =  c_coef.dot(x)      # Observation function
    c_x = lambdify(x, c, 'numpy')
    p_b = 0
    for m in range(len(parameters.w_b)):
        p_b = p_b + parameters.w_b[m] * (1/ np.sqrt(2 * math.pi * parameters.sigma_b[m]**2))* exp(-(x[0] - parameters.mu_b[m])**2/ (2* parameters.sigma_b[m]**2))
    p_b_x = lambdify(x[0],p_b, 'numpy')
    gm = d     # No. of dimensions with Gaussian mixture densities in the dim-dimensional density, should be <= d
    
    
    fig, ax = plt.subplots(nrows = 1, ncols = len(N_values), sharey = True, figsize = (21,8))
    for i,T in enumerate(T_values):
        n = parameters.T_values.index(T)
        for run in range(No_runs):
            print('Dimensions ', d)
            print('No. particles ',N_values[1])
            print('Simulation time of Langevin SDE ', T)
            print('Run ',run)
            if No_runs > 1:
                seed = np.random.randint(1,500)
            else:
                seed = parameters.seed # np.random.randint(1,500)
            print('Seed ', seed)
            Xi  = fpf.get_samples(N_values[1], parameters.mu_b, parameters.sigma_b, parameters.w_b, d, gm, parameters.sigma, seed = seed)
            if d == 1:
                Xi = np.sort(Xi,kind = 'mergesort')

            C = np.reshape(c_x(*Xi.T),(len(Xi),1))
                                  
            if parameters.exact == 1:
                K_exact  = fpf.gain_num_integrate(Xi, (c_coef[0] * x[0]), p_b, x, 0)
                K_exact = K_exact.reshape((-1,1))
                ax[i].plot(Xi, K_exact, '^', label = 'Exact')
                ax[i].set_xlabel('$x$')
             
            if parameters.diff_td == 1:
                K_diff_td, Phi_td = fpf.gain_diff_td(Xi, c, parameters.w_b, parameters.mu_b, parameters.sigma_b, p_b, x, parameters.basis_dim, parameters.basis, parameters.affine, T, diag = 0)
                ax[i].plot(Xi, K_diff_td, '*', label = 'Polynomial')
                if parameters.exact == 1:
                    mse_diff_td[run,n] = fpf.mean_squared_error(K_exact, K_diff_td)

                            
            if parameters.diff_nl_td == 1:
                K_diff_nl_td,Phi_nl_td = fpf.gain_diff_nl_td(Xi, c, p_b, x, 9, parameters.nlbasis, T, diag = 0)
                ax[i].plot(Xi, K_diff_nl_td, '>', label = 'Nonlinear')
                if parameters.exact == 1:
                    mse_diff_nl_td[run,n] = fpf.mean_squared_error(K_exact, K_diff_nl_td)

            if parameters.const == 1:
                eta = np.mean(C)
                Y = (C -eta)
                K_const = np.mean(Y * Xi, axis = 0)
                ax[i].axhline(y= K_const, color = 'k', linestyle = '--', label ='Const gain')
                if parameters.exact == 1:
                    mse_const[run,n] = fpf.mean_squared_error(K_exact, K_const)  
                    
            ax[i].set_ylim(-0.5,10)
            ax[i].text(0.05, 0.95, '$\sf T = {}$'.format(T), fontsize = 20, transform = ax[i].transAxes, verticalalignment='top')
            if i == 0:
                ax[i].legend(loc = 1, framealpha = 0)
                ax[i].set_ylabel('$\sf K(x)$')
            
            ax2 = ax[i].twinx()
            if parameters.diff_td == 1:
                sns.distplot(Phi_td, label = 'Histogram of $x^i$')
            if parameters.diff_nl_td == 1:
                sns.distplot(Phi_nl_td)
            
            domain = np.arange(-3,3,0.1)
            ax2.plot(domain,p_b_x(domain), 'k',label = '$\\rho(x)$')
            ax2.set_ylim(0,1)
            if i == 2:
                ax2.set_ylabel('$\\rho(x)$')
           
            
    ax2.legend(loc = 1, framealpha = 0)
    
    # fig.savefig('Figure/Chap4_diff_td_linear_wt_polynomials.pdf',bbox_inches='tight')

                    

 
   