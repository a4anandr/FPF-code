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

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'auto')
get_ipython().run_line_magic('matplotlib', 'inline')

import pickle
import pandas as pd
import collections
import os

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    plt.close('all')
    
    runyn = input('Do you want to just plot the results?')
    if runyn.lower() == 'n':
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
                
        # Run parameters
        No_runs = input('Input the number of independent runs -')
        if No_runs == '':
            No_runs = parameters.No_runs
        else:
            No_runs = int(No_runs)
        
        # FPF parameters - No. of particles
        N_values = input('Input the number of particles -')
        if N_values == '':
            N_values = parameters.N_values
        else:
            N_values = [int(N) for N in N_values.split(',')]
            
        dim = input('Input the system dimensions -')
        if dim =='':
            dim = parameters.d_values
        else:
            dim = [int(d) for d in dim.split(',')]
            
        saveyn = input('Do you want to save the MSEs in a file?')
        
        mse_finite = np.zeros((No_runs, len(parameters.d_values), len(parameters.N_values)))
        mse_diff_td = np.zeros((No_runs, len(parameters.d_values), len(parameters.N_values)))
        mse_diff_nl_td = np.zeros((No_runs, len(parameters.d_values), len(parameters.N_values)))
        mse_coif = np.zeros((No_runs, len(parameters.d_values), len(parameters.N_values)))
        mse_rkhs_N = np.zeros((No_runs, len(parameters.d_values), len(parameters.N_values)))
        mse_rkhs_dN = np.zeros((No_runs, len(parameters.d_values), len(parameters.N_values)))
        mse_om   = np.zeros((No_runs, len(parameters.d_values), len(parameters.N_values)))
        mse_coif_old = np.zeros((No_runs, len(parameters.d_values), len(parameters.N_values)))
        mse_const = np.zeros((No_runs, len(parameters.d_values), len(parameters.N_values)))
    
        # System parameters
        for d in dim:     # dimension of the system
            i = parameters.d_values.tolist().index(d)
            x = symbols('x0:%d'%d)
            c_coef = np.ones(d)# np.ones((1,d))   # [1,0]
            c =  c_coef.dot(x)      # Observation function
            c_x = lambdify(x, c, 'numpy')
            p_b = 0
            for m in range(len(parameters.w_b)):
                p_b = p_b + parameters.w_b[m] * (1/ np.sqrt(2 * math.pi * parameters.sigma_b[m]**2))* exp(-(x[0] - parameters.mu_b[m])**2/ (2* parameters.sigma_b[m]**2))
            gm = d     # No. of dimensions with Gaussian mixture densities in the dim-dimensional density, should be <= d
            
            for N in N_values:
                n = parameters.N_values.index(N)
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
                    if d == 1:
                        Xi = np.sort(Xi,kind = 'mergesort')
    
                    C = np.reshape(c_x(*Xi.T),(len(Xi),1))
    
                    if parameters.exact == 1:
                        K_exact = np.zeros((N, d))
                        # domain = np.arange(-2,2,0.01)
                        # K_exact_plot = np.zeros((len(domain), d))
                        for d_i in np.arange(gm):
                            # K_exact[:,d_i]  = fpf.gain_num_integrate(Xi, (c_coef[d_i] * x[0])[0], p_b, x, d_i)
                            K_exact[:,d_i]  = fpf.gain_num_integrate(Xi, (c_coef[d_i] * x[0]), p_b, x, d_i)
                            # K_exact_plot[:,d_i]  = fpf.gain_num_integrate(domain.reshape((len(domain),1)),x[0] , p_b, x, d_i)
                    
                    if parameters.diff_td == 1:
                        if d == 1:
                            K_diff_td,_ = fpf.gain_diff_td(Xi, c, parameters.w_b, parameters.mu_b, parameters.sigma_b, p_b, x, parameters.basis_dim, parameters.basis, parameters.affine, parameters.T_values[2], diag = 1)
                        if parameters.exact == 1:
                            mse_diff_td[run,i,n] = fpf.mean_squared_error(K_exact, K_diff_td)
                            
                    if parameters.diff_nl_td == 1:
                        if d ==1:
                            K_diff_nl_td,_ = fpf.gain_diff_nl_td(Xi, c, p_b, x, 9, parameters.nlbasis, parameters.T_values[2], diag = 1)
                        if parameters.exact == 1:
                            mse_diff_nl_td[run,i,n] = fpf.mean_squared_error(K_exact, K_diff_nl_td)

                    if parameters.finite == 1:
                        if d == 1:
                            if parameters.method == 'integration':
                                X, K_finite_X = fpf.gain_finite_integrate(c_x, parameters.w_b, parameters.mu_b, parameters.sigma_b, parameters.basis_dim, parameters.basis, parameters.affine,  diag = 0)
                                knn = spatial.KDTree(X.reshape((len(X),1)))
                                K_finite = K_finite_X[knn.query(Xi)[1]].reshape(N,1)
                            else:
                                K_finite  = fpf.gain_finite(Xi, C, parameters.mu_b, parameters.sigma_b, parameters.basis_dim, parameters.basis, parameters.affine, diag =1)
                            if parameters.exact == 1:
                                mse_finite[run,i,n] = fpf.mean_squared_error(K_exact, K_finite)
                                
                    if parameters.coif ==1:
                        eps_coif = hyperparams_coif_dict[d][N] if d in hyperparams_coif_dict and N in hyperparams_coif_dict[d] else 1
                        Phi = np.zeros(N)
                        K_coif = fpf.gain_coif(Xi, C, eps_coif, Phi, diag = 0)
                        if parameters.exact == 1:
                            mse_coif[run,i,n] = fpf.mean_squared_error(K_exact, K_coif)
    
                    if parameters.rkhs_N == 1:
                        eps_rkhs_N =0.5
                        Lambda_rkhs_N = 10**(-3)
                        K_rkhs_N,_ = fpf.gain_rkhs_N(Xi, C, eps_rkhs_N, Lambda_rkhs_N, diag = 0)
                        if parameters.exact == 1:
                            mse_rkhs_N[run,i,n] = fpf.mean_squared_error(K_exact, K_rkhs_N)
    
                    if parameters.rkhs_dN == 1:
                        eps_rkhs_dN = hyperparams_om_dict[d][N][0] if d in hyperparams_om_dict and N in hyperparams_om_dict[d] else 1
                        Lambda_rkhs_dN = hyperparams_om_dict[d][N][1] if d in hyperparams_om_dict and N in hyperparams_om_dict[d] else 1e-2
                        K_rkhs_dN,_ = fpf.gain_rkhs_dN(Xi, C, eps_rkhs_dN, Lambda_rkhs_dN, diag = 0)
                        if parameters.exact == 1:
                            mse_rkhs_dN[run,i,n] = fpf.mean_squared_error(K_exact, K_rkhs_dN)
    
                    if parameters.om == 1:
                        eps_om = hyperparams_om_dict[d][N][0] if d in hyperparams_om_dict and N in hyperparams_om_dict[d] else 1
                        Lambda_om = hyperparams_om_dict[d][N][1] if d in hyperparams_om_dict and N in hyperparams_om_dict[d] else 1e-2
                        K_om,_ = fpf.gain_rkhs_om(Xi, C, eps_om, Lambda_om, diag = 0)
                        if parameters.exact == 1:
                            mse_om[run,i,n] = fpf.mean_squared_error(K_exact, K_om)
    
                    if parameters.coif_old == 1:
                        eps_coif_old = 0.1
                        Phi = np.zeros(N)
                        K_coif_old = fpf.gain_coif_old(Xi, C, eps_coif, Phi, diag = 0)
                        if parameters.exact == 1:
                            mse_coif_old[run,i,n] = fpf.mean_squared_error(K_exact, K_coif_old)
    
                    if parameters.const == 1:
                        eta = np.mean(C)
                        Y = (C -eta)
                        K_const = np.mean(Y * Xi, axis = 0)
                        if parameters.exact == 1:
                            mse_const[run,i,n] = fpf.mean_squared_error(K_exact, K_const)     
    
                if saveyn.lower() == 'y':
                    if os.path.isfile('temp/mse_coif_d_N_{}.pkl'.format(No_runs)):
                        output = open('temp/mse_coif_d_N_{}.pkl'.format(No_runs), 'rb')
                        mse_coif_file = pickle.load(output)
                        mse_coif_file[:,i,n] = mse_coif[:,i,n]
                        output.close()
                    output = open('temp/mse_coif_d_N_{}.pkl'.format(No_runs),'wb')
                    pickle.dump(mse_coif_file,output)
                    output.close()
                    
                    if os.path.isfile('temp/mse_om_d_N_{}.pkl'.format(No_runs)):
                        output = open('temp/mse_om_d_N_{}.pkl'.format(No_runs), 'rb')
                        mse_om_file = pickle.load(output)
                        mse_om_file[:,i,n] = mse_om[:,i,n]
                        output.close()
                    output = open('temp/mse_om_d_N_{}.pkl'.format(No_runs),'wb')
                    pickle.dump(mse_om_file,output)
                    output.close()
                    
                    if os.path.isfile('temp/mse_const_d_N_{}.pkl'.format(No_runs)):
                        output = open('temp/mse_const_d_N_{}.pkl'.format(No_runs),'rb')
                        mse_const_file = pickle.load(output)
                        mse_const_file[:,i,n]= mse_const[:,i,n]
                        output.close()
                    output = open('temp/mse_const_d_N_{}.pkl'.format(No_runs),'wb')
                    pickle.dump(mse_const_file,output)
                    output.close()
    
    ##### Loading the pickle file of MSEs 
    else:
        No_runs = parameters.No_runs
        dim = parameters.d_values
        N_values = parameters.N_values
        
    if os.path.isfile('output/mse_coif_d_N_{}.pkl'.format(No_runs)):
        output = open('output/mse_coif_d_N_{}.pkl'.format(No_runs),'rb')
        mse_coif = pickle.load(output)
        output.close()
    
    if os.path.isfile('output/mse_om_d_N_{}.pkl'.format(No_runs)):
        output = open('output/mse_om_d_N_{}.pkl'.format(No_runs),'rb')
        mse_om = pickle.load(output)
        output.close()
    
    if os.path.isfile('output/mse_const_d_N_{}.pkl'.format(No_runs)):
        output = open('output/mse_const_d_N_{}.pkl'.format(No_runs),'rb')
        mse_const = pickle.load(output)
        output.close()

    if No_runs == 1:
        if parameters.exact == 1 and parameters.const == 1 and parameters.coif == 0 and parameters.om == 1 and parameters.diff_td ==1 and parameters.diff_nl_td ==1:
            fpf.plot_gains(Xi,p_b,x,K_exact=K_exact, K_const = K_const, K_diff_td = K_diff_td, K_diff_nl_td = K_diff_nl_td, K_om = K_om)
 
    if len(N_values) > 1:
        ##### Plotting MSE v $N$ for all $d$
        for i,d in enumerate(dim):
            fig = plt.figure(figsize = parameters.figure_size)
            plt.plot(N_values, np.mean(mse_coif,axis = 0)[i], label = 'Markov kernel')
            plt.plot(N_values, np.mean(mse_om, axis =0)[i], label = 'RKHS OM')
            plt.plot(N_values, np.mean(mse_const,axis =0)[i], 'k--',label = 'Const')
            plt.ylabel('Average MSEs')
            plt.xlabel('$N$')
            plt.legend(framealpha = 0, prop ={'size' :22})
            plt.title('Average MSEs obtained from ' + str(No_runs) + ' trials for $d=$ '+ str(d))
            plt.show()
            fig.savefig('Figure/MSEvNford'+str(d)+'.jpg')
        
        ##### Log plot of MSE v $\log N$ for all $d$
        for i,d in enumerate(dim):
            fig = plt.figure(figsize = parameters.figure_size)
            plt.plot(N_values, np.mean(mse_coif,axis = 0)[i], label = 'Markov kernel')
            plt.plot(N_values, np.mean(mse_om, axis =0)[i], label = 'RKHS OM')
            plt.plot(N_values, np.mean(mse_const,axis =0)[i], 'k--',label = 'Const')
            plt.plot(N_values, np.divide(1,N_values),'b--', label ='$1/N$')
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('Average MSEs')
            plt.xlabel('$N$')
            plt.legend(framealpha = 0, prop ={'size' :22})
            plt.title('Average MSEs obtained from ' + str(No_runs) + ' trials for $d=$ '+ str(d))
            plt.show()
            fig.savefig('Figure/logMSEvlogNford'+str(d)+'.jpg')
        
        #### Including the intercept to -logN in the plot
        offset = np.mean(np.log10(np.mean(mse_om,axis =0)),axis =1) + np.mean(np.log10(N_values))
        for i,d in enumerate(dim):
            fig = plt.figure(figsize = parameters.figure_size)
            plt.plot(np.log10(N_values), np.log10(np.mean(mse_coif,axis = 0))[i], label = 'Markov kernel')
            plt.plot(np.log10(N_values), np.log10(np.mean(mse_om, axis =0))[i], label = 'RKHS OM')
            plt.plot(np.log10(N_values), np.log10(np.mean(mse_const,axis =0))[i], 'k--',label = 'Const')
            plt.plot(np.log10(N_values), np.log10(np.divide(1,N_values))+offset[i],'b--', label ='$1/N$')
            plt.ylabel('Average MSEs')
            plt.xlabel('$N$')
            plt.legend(framealpha = 0, prop ={'size' :22})
            plt.title('Average MSEs obtained from ' + str(No_runs) + ' trials for $d=$ '+ str(d))
            plt.show()
            fig.savefig('Figure/logMSEvlogNfordWithIntercept'+str(d)+'.jpg')
        
        ##### Plot of MSE v $d$ for all $N$
        for n,N in enumerate(N_values):
            fig = plt.figure(figsize = parameters.figure_size)
            plt.plot(dim, np.mean(mse_coif,axis = 0)[:,n], label = 'Markov kernel')
            plt.plot(dim, np.mean(mse_om, axis =0)[:,n], label = 'RKHS OM')
            plt.plot(dim, np.mean(mse_const,axis =0)[:,n], 'k--',label = 'Const')
            plt.yscale('linear')
            plt.xscale('linear')
            plt.ylabel('Average MSEs')
            plt.xlabel('$d$')
            plt.legend(framealpha = 0, prop ={'size' :22})
            plt.title('Average MSEs obtained from ' + str(No_runs) + ' trials for $N=$ '+ str(N))
            plt.show()
            fig.savefig('Figure/logMSEvdforN'+str(N)+'.jpg')
