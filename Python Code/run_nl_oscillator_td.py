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
matplotlib.rc('font', **parameters.font)
# matplotlib.rcParams.update(parameters.font_params)

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
    basis_dim_values = parameters.basis_dim
    
    mse_finite = np.zeros((No_runs,len(basis_dim_values)))
    mse_diff_td = np.zeros((No_runs,len(basis_dim_values)))
    mse_diff_nl_td = np.zeros((No_runs,len(basis_dim_values)))
    mse_const = np.zeros((No_runs, len(basis_dim_values)))
    
    # System parameters
    d = 1    # dimension of the system
    x = symbols('x0:%d'%d)

    c = 0.5 * (1 + cos(x[0]))
    c_x = lambdify(x, c, 'numpy')

    w_v = [0.5,0.5]
    K_v = [3, 3]
    mu_v = [-60*math.pi/180, 60*math.pi/180]
    p_v = fpf.get_von_mises(w_v, K_v, mu_v)
    p_v_x = lambdify(x,p_v,'numpy')
    
    fig, ax = plt.subplots(nrows = 1, ncols = len(basis_dim_values), sharey = True, figsize = (21,8))
    for i,dim in enumerate(basis_dim_values):
        n = basis_dim_values.index(dim)
        for run in range(No_runs):
            print('Dimensions ', d)
            print('No. particles ',N_values)
            print('Simulation time of Langevin SDE ', T_values[0])
            print('Run ',run)
            if No_runs > 1:
                seed = np.random.randint(1,500)
            else:
                seed = parameters.seed # np.random.randint(1,500)
            print('Seed ', seed)
            Xi = np.arange(-math.pi,math.pi,0.1).reshape((-1,1))

            C = np.reshape(c_x(*Xi.T),(len(Xi),1))
                                  
            if parameters.exact == 1:
                K_exact  = fpf.gain_num_integrate(Xi, c, p_v, x, 0, [-math.pi, math.pi])
                K_exact = K_exact.reshape((-1,1))
                ax[i].plot(Xi, K_exact, '^', label = 'Exact')
                ax[i].set_xlabel('$x$')
             
            if parameters.diff_td == 1:
                K_diff_td, K_be_diff_td, Phi_td = fpf.gain_diff_td_oscillator(Xi, c, p_v, x, dim, parameters.basis, parameters.affine, T_values[0], seed, diag = 0)
                ax[i].plot(Xi, K_diff_td, '*', label = r'$\nabla$-LSTD')
                ax[i].plot(Xi, K_be_diff_td, '>', label = 'BE-minimization')
                if parameters.exact == 1:
                    mse_diff_td[run,n] = fpf.mean_squared_error(K_exact, K_diff_td)

                            
            if parameters.diff_nl_td == 1:
                K_diff_nl_td,Phi_nl_td = fpf.gain_diff_nl_td(Xi, c, p_v, x, 9, parameters.nlbasis, T_values[0], diag = 0)
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
                    
            # ax[i].set_ylim(-0.5,10)
            ax[i].text(0.05, 0.95, '$\sf T = {}$'.format(T_values[0]), fontsize = 20, transform = ax[i].transAxes, verticalalignment='top')
            ax[i].text(0.05, 0.9, r'$\ell  = {}$'.format(dim), fontsize = 20, transform = ax[i].transAxes, verticalalignment='top')
            if i == 0:
                ax[i].legend(loc = 1, framealpha = 0)
                ax[i].set_ylabel('$\sf K(x)$')
            
            ax2 = ax[i].twinx()
            if parameters.diff_td == 1:
                sns.distplot(Phi_td, label = 'Histogram of $x^i$')
            if parameters.diff_nl_td == 1:
                sns.distplot(Phi_nl_td, label = 'Histogram of $x^i$')
            
            domain = np.arange(-math.pi,math.pi,0.1)
            ax2.plot(domain,p_v_x(domain), 'k',label = '$\\rho(x)$')
            ax2.set_ylim(0,1)
            if i == 2:
                ax2.set_ylabel('$\\rho(x)$')
           
            
    ax2.legend(loc = 1, framealpha = 0)
    
    # fig.savefig('Figure/Chap4_diff_td_linear_wt_polynomials.pdf',bbox_inches='tight')

                    

 
   