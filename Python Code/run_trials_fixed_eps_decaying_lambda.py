# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 01:00:38 2019

@author: anand
"""

#### MSE for each $d$ with fixed \epsilon and $\lambda$ decaying with $N$ 
import numpy as np
from sympy import *
import math

import fpf_module as fpf
import parameters

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'auto')

import pickle
import pandas as pd
import collections
import os

if __name__ == '__main__':
    plt.close('all')
    
    runyn = input('Do you want to just plot the results?')
    if runyn.lower() == 'n':
        
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
            
        lambda_decay = input('Decay \lambda as (1/N)(y) or (1/sqrt(N)(n)')
        ## Computing the constant factor k to multiply, as \lambda(N=25) = 0.1 for all d
        if lambda_decay.lower() == 'y':
            c_lambda = 25*0.1
        else:
            c_lambda = np.sqrt(25)*0.1
            
        saveyn = input('Do you want to save the MSEs in a file?')

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
            c_coef = np.ones((1,d))   # [1,0]
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
                        seed = np.random.randint(1,500)
                    print('Seed ', seed)
                    Xi  = fpf.get_samples(N, parameters.mu_b, parameters.sigma_b, parameters.w_b, d, gm, parameters.sigma, seed = seed)
                    if d == 1:
                        Xi = np.sort(Xi,kind = 'mergesort')
    
                    C = np.reshape(c_x(*Xi.T),(len(Xi),1))
    
                    if parameters.exact == 1:
                        K_exact = np.zeros((N, d))
                        for d_i in np.arange(gm):
                            K_exact[:,d_i]  = fpf.gain_num_integrate(Xi,x[0] , p_b, x, d_i)
    
                    if parameters.rkhs_N == 1:
                        eps_rkhs_N = parameters.eps_fixed_dict[d]
                        Lambda_rkhs_N = c_lambda/N if lambda_decay == 'y' else c_lambda/np.sqrt(N)
                        K_rkhs_N = fpf.gain_rkhs_N(Xi, C, eps_rkhs_N, Lambda_rkhs_N, diag = 0)
                        if parameters.exact == 1:
                            mse_rkhs_N[run,i,n] = fpf.mean_squared_error(K_exact, K_rkhs_N)
    
                    if parameters.rkhs_dN == 1:
                        eps_rkhs_dN = parameters.eps_fixed_dict[d]
                        Lambda_rkhs_dN = c_lambda/N if lambda_decay == 'y' else c_lambda/np.sqrt(N)
                        K_rkhs_dN = fpf.gain_rkhs_dN(Xi, C, eps_rkhs_dN, Lambda_rkhs_dN, diag = 0)
                        if parameters.exact == 1:
                            mse_rkhs_dN[run,i,n] = fpf.mean_squared_error(K_exact, K_rkhs_dN)
    
                    if parameters.om == 1:
                        eps_om = parameters.eps_fixed_dict[d]
                        Lambda_om = c_lambda/N if lambda_decay == 'y' else c_lambda/np.sqrt(N)
                        K_om = fpf.gain_rkhs_om(Xi, C, eps_om, Lambda_om, diag = 0)
                        if parameters.exact == 1:
                            mse_om[run,i,n] = fpf.mean_squared_error(K_exact, K_om)

                    if parameters.const == 1:
                        eta = np.mean(C)
                        Y = (C -eta)
                        K_const = np.mean(Y * Xi, axis = 0)
                        if parameters.exact == 1:
                            mse_const[run,i,n] = fpf.mean_squared_error(K_exact, K_const)     
    
                if saveyn.lower() == 'y':
                    filename_om = 'temp/mse_om_eps_fixed_lambda_N_{}.pkl'.format(No_runs) if lambda_decay =='y' else 'temp/mse_om_eps_fixed_lambda_root_N_{}'.format(No_runs)
                    if os.path.isfile(filename_om):
                        output = open(filename_om, 'rb')
                        mse_om_file = pickle.load(output)
                        mse_om_file[:,i,n] = mse_om[:,i,n]
                        output.close()
                    else:
                        mse_om_file = mse_om
                    output = open(filename_om,'wb')
                    pickle.dump(mse_om_file,output)
                    output.close()
                    
                    
                    filename_const = 'temp/mse_const_eps_fixed_lambda_N_{}.pkl'.format(No_runs) 
                    if os.path.isfile(filename_const):
                        output = open(filename_const,'rb')
                        mse_const_file = pickle.load(output)
                        mse_const_file[:,i,n]= mse_const[:,i,n]
                        output.close()
                    else:
                        mse_const_file = mse_const
                    output = open(filename_const,'wb')
                    pickle.dump(mse_const_file,output)
                    output.close()
    
    ##### Loading the pickle file of MSEs 
    else:
        No_runs = parameters.No_runs
        dim = parameters.d_values
    N_values = parameters.N_values
    
    files = os.listdir('temp/')
    filedict = dict(zip(np.arange(len(files)), files))
    print(filedict)
    chosen_file_keys = input('Choose the output files -')
    chosen_file_keys = [int(i) for i in chosen_file_keys.split(',')]
    if not chosen_file_keys:
        print('No files chosen, Exiting...')
        quit()
    else:
        for file_index in chosen_file_keys:
            filename = filedict[file_index]
            output = open('temp/{}'.format(filename),'rb')
            if 'om' in filename:
                if 'root' in filename:
                    mse_om_root_N = pickle.load(output)
                else:
                    mse_om_N = pickle.load(output)
            elif 'const' in filename:
                mse_const = pickle.load(output)
            output.close()
    
    if No_runs == 1:
        fpf.plot_gains(Xi, K_exact, K_const, K3 = K_om)
    
    if len(N_values) > 1:
        ##### Plotting MSE v $N$ for all $d$
        for d in dim:
            i = parameters.d_values.tolist().index(d)
            print(i)
            fig = plt.figure(figsize=(10,5))
            plt.plot(N_values, np.mean(mse_om_N, axis =0)[i], label = '$\lambda \propto 1/N$')
            plt.plot(N_values, np.mean(mse_om_root_N, axis =0)[i], label = '$\lambda \propto \sqrt{1/N}$')
            plt.plot(N_values, np.mean(mse_const,axis =0)[i], 'k--',label = 'Const')
            plt.ylabel('Average MSEs')
            plt.xlabel('$N$')
            plt.legend(framealpha = 0, prop ={'size' :22})
            plt.title('Average MSEs obtained from ' + str(No_runs) + ' trials for $d=$ '+ str(d))
            plt.show()
            fig.savefig('Figure/MSEvNepsFixedford'+str(d)+'.jpg')
        
        ##### Log plot of MSE v $\log N$ for all $d$
        for d in dim:
            i = parameters.d_values.tolist().index(d)
            print(i)
            fig = plt.figure(figsize=(10,5))
            plt.plot(N_values, np.mean(mse_om_N, axis =0)[i], label = '$\lambda \propto 1/N$')
            plt.plot(N_values, np.mean(mse_om_root_N,axis =0)[i],label = '$\lambda \propto \sqrt{1/N}$')
            plt.plot(N_values, np.mean(mse_const,axis =0)[i], 'k--',label = 'Const')
            plt.plot(N_values, np.divide(1,N_values),'b--', label ='$1/N$')
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('Average MSEs')
            plt.xlabel('$N$')
            plt.legend(framealpha = 0, prop ={'size' :22})
            plt.title('Average MSEs obtained from ' + str(No_runs) + ' trials for $d=$ '+ str(d))
            plt.show()
            fig.savefig('Figure/logMSEvlogNepsFixedford'+str(d)+'.jpg')
        
        #### Including the intercept to -logN in the plot
        # offset = np.mean(np.log10(np.mean(mse_om_N,axis =0)),axis =1) + np.mean(np.log10(N_values))
        offset = np.zeros(len(parameters.d_values))
        for d in dim:
            i = parameters.d_values.tolist().index(d)
            print(i)
            offset[i] = np.mean(np.log10(np.mean(mse_om_N[:,i],axis =0)),axis =0) + (1/d) * np.mean(np.log10(N_values))
            fig = plt.figure(figsize=(10,5))
            plt.plot(np.log10(N_values), np.log10(np.mean(mse_om_N, axis =0))[i], label = '$\lambda \propto 1/N$')
            plt.plot(np.log10(N_values), np.log10(np.mean(mse_om_root_N, axis =0))[i], label = '$\lambda \propto \sqrt{1/N}$')
            plt.plot(np.log10(N_values), np.log10(np.mean(mse_const,axis =0))[i], 'k--',label = 'Const')          
            plt.plot(np.log10(N_values), (1/d) * np.log10(np.divide(1,N_values))+offset[i],'b--', label ='$(1/N)^{1/d}$')
            plt.ylabel('Average MSEs')
            plt.xlabel('$N$')
            plt.legend(framealpha = 0, prop ={'size' :22})
            plt.title('Average MSEs obtained from ' + str(No_runs) + ' trials for $d=$ '+ str(d))
            plt.show()
            fig.savefig('Figure/logMSEvlogNepsFixedfordWithIntercept'+str(d)+'.jpg')
        
        ##### Plot of MSE v $d$ for all $N$
        for N in N_values:
            n = parameters.N_values.index(N)
            fig = plt.figure(figsize=(10,5))
            plt.plot(dim, np.mean(mse_om_N, axis =0)[:,n], label = '$\lambda \propto 1/N$')
            plt.plot(dim, np.mean(mse_om_root_N, axis =0)[:,n], label = '$\lambda \propto \sqrt{1/N}$')
            plt.plot(dim, np.mean(mse_const,axis =0)[:,n], 'k--',label = 'Const')
            plt.yscale('linear')
            plt.xscale('linear')
            plt.ylabel('Average MSEs')
            plt.xlabel('$d$')
            plt.legend(framealpha = 0, prop ={'size' :22})
            plt.title('Average MSEs obtained from ' + str(No_runs) + ' trials for $N=$ '+ str(N))
            plt.show()
            fig.savefig('Figure/MSEvdFixedepsforN'+str(N)+'.jpg')
