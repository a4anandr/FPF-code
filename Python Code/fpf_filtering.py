# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:23:10 2019

@author: a4ana
"""
### Filtering problem with a bimodal prior density, linear observations
### No dynamics on the state model, so just parameter estimation example
import numpy as np
from sympy import *
import math
from scipy import spatial
from scipy import stats

from sklearn.neighbors import KernelDensity

import pandas as pd

import collections

import fpf_module as fpf
import filtering_params as fp


import matplotlib
matplotlib.rc('text',usetex = True)
#matplotlib.rc('font', **fp.font)
#matplotlib.rc('legend',fontsize = 18)
#matplotlib.rc('fontsize' = 18)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)
# matplotlib.rcParams.update(parameters.font_params)

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'auto')
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import os
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    # Using common seed for all filters
    print('Seed ',fp.seed)
    Xi = fpf.get_samples(fp.N,fp.mu,fp.sigma,fp.w,fp.d, seed = fp.seed)
    
    domain = np.arange(-3,3,0.1)

    Ts = int(fp.T/fp.dt)   # Total number of time steps of filtering
    ts = np.arange(Ts)
    # Initializing all the Monte Carlo methods with these particles from the prior 
    if fp.exact  == 1: 
        Xi_exact = np.zeros((fp.N,Ts))
        Xi_exact[:,0] = Xi.reshape(-1)
        K_exact  = np.zeros((fp.N,Ts))
        dI_exact = np.zeros((fp.N,Ts))
        p = fp.p
    
    if fp.coif  == 1: 
        Xi_coif = np.zeros((fp.N,Ts))
        Xi_coif[:,0] = Xi.reshape(-1)
        K_coif  = np.zeros((fp.N,Ts))
        Phi     = np.zeros((fp.N,1))
        dI_coif = np.zeros((fp.N,Ts))

    if fp.rkhs_N  == 1: 
        Xi_rkhs = np.zeros((fp.N,Ts))
        Xi_rkhs[:,0] = Xi.reshape(-1)
        K_rkhs  = np.zeros((fp.N,Ts))
        dI_rkhs = np.zeros((fp.N,Ts))

    if fp.om  == 1: 
        Xi_om = np.zeros((fp.N,Ts))
        Xi_om[:,0] = Xi.reshape(-1)
        K_om = np.zeros((fp.N,Ts))
        dI_om = np.zeros((fp.N,Ts))
        
    if fp.sis  == 1: 
        Xi_sis = np.zeros((fp.N,Ts))
        wi_sis = (1/fp.N) * np.ones((fp.N,Ts))
        Xi_sis[:,0] = Xi.reshape(-1)
        Zi_sis = np.zeros((fp.N,Ts))
        N_eff_sis = np.zeros((fp.N,1))
        p_sis = np.zeros((len(domain),Ts))
        
    if fp.kalman  == 1: 
        Xi_kalman = np.zeros((1,Ts))
        P_kalman  = np.zeros((1,Ts))
        K_kalman  = np.zeros((1,Ts))
        Xi_kalman[:,0] = np.mean(Xi)  # np.dot(fp.w,fp.mu) - Actual mean
        P_kalman[:,0]  = np.var(Xi) # np.dot(fp.w,fp.mu**2 + fp.sigma**2) - np.dot(fp.w,fp.mu)**2
        K_kalman[:,0]  = P_kalman[:,0] * fp.c_dot_x(Xi_kalman) / (fp.sigmaW**2)
        
    if fp.const  == 1:
        Xi_const  = np.zeros((fp.N,Ts))
        Xi_const[:,0] = Xi.reshape(-1)
        K_const   = np.zeros((1,Ts))
        dI_const = np.zeros((fp.N,Ts))

#    diff_td = 0     # Computes the gain using diff TD algorithm using eligibility vectors
#    diff_nl_td = 0  # Computes the gain using diff TD algorithm for nonlinear parameterization using Stochastic Approximation
#    finite = 0      # Computes gain using finite set of basis functions
#    const  = 0      # Computes the constant gain approximation

#%% State and observation process evolution
    X = np.zeros((1,Ts))
    X[:,0] = fp.mu[1]
    
    Z = np.zeros((1,Ts))
    dZ = np.zeros((1,Ts))
    Z[:,0] = fp.c_x(X[:,0]) + fp.sigmaW * fp.dt * np.random.randn()

    for k in np.arange(1,Ts):
        print('Time ',k)
        X[:,k] = X[:,k-1] + fp.a_x(X[:,k-1]) * fp.dt + fp.sigmaB * fp.sdt * np.random.randn()
        # X[:,k] = X[:,k-1]
        Z[:,k] = Z[:,k-1] + fp.c_x(X[:,k]) * fp.dt + fp.sigmaW * fp.sdt * np.random.randn()
        if k == 1:
            dZ[:,k] = Z[:,k] - Z[:,k-1]
        else:
            dZ[:,k] = 0.5 * (Z[:,k] - Z[:,k-2])
        
#%% Filters
        common_rand = np.random.randn()
        #%% Exact computation of gain
        if fp.exact == 1:
            C = fp.c_x(Xi_exact[:,[k-1]])
            dI_exact[:,[k]] = dZ[:,[k]] - 0.5 * (C + np.mean(C)) * fp.dt 
            K_exact[:,[k]] = fpf.gain_num_integrate(Xi_exact[:,[k-1]], fp.c, p, fp.x).reshape(-1,1)
            Xi_exact[:,[k]] = Xi_exact[:,[k-1]] + fp.a_x(Xi_exact[:,[k-1]]) * fp.dt + fp.sigmaB * fp.sdt * common_rand + (1/fp.sigmaW**2) * K_exact[:,[k]] * dI_exact[:,[k]]
            p   = fpf.get_kernel_pdf(Xi_exact[:,[k]], domain, sym=1)
            
        #%% RKHS implementation
        if fp.rkhs_N == 1:
            C = fp.c_x(Xi_rkhs[:,[k-1]])
            dI_rkhs[:,[k]] = dZ[:,[k]] - 0.5 * (C + np.mean(C)) * fp.dt 
            K_rkhs[:,[k]],_   = fpf.gain_rkhs_N(Xi_rkhs[:,[k-1]],C,fp.eps, fp.Lambda)
            Xi_rkhs[:,[k]] = Xi_om[:,[k-1]] + fp.a_x(Xi_rkhs[:,[k-1]]) * fp.dt + fp.sigmaB * fp.sdt * common_rand + (1/fp.sigmaW**2) * K_rkhs[:,[k]] * dI_rkhs[:,[k]]
        #%% Markov semigroup
        if fp.coif == 1:
            C = fp.c_x(Xi_coif[:,[k-1]])
            dI_coif[:,[k]] = dZ[:,[k]] - 0.5 * (C + np.mean(C)) * fp.dt 
            K_coif[:,[k]],Phi   = fpf.gain_coif(Xi_coif[:,[k-1]],C,fp.eps,Phi)
            Xi_coif[:,[k]] = Xi_coif[:,[k-1]] + fp.a_x(Xi_coif[:,[k-1]]) * fp.dt + fp.sigmaB * fp.sdt * common_rand + (1/fp.sigmaW**2) * K_coif[:,[k]] * dI_coif[:,[k]]
        #%% RKHS OM
        if fp.om == 1:
            C = fp.c_x(Xi_om[:,[k-1]])
            dI_om[:,[k]] = dZ[:,[k]] - 0.5 * (C + np.mean(C)) * fp.dt 
            K_om[:,[k]],_   = fpf.gain_rkhs_om(Xi_om[:,[k-1]],C,fp.eps, fp.Lambda)
            Xi_om[:,[k]] = Xi_om[:,[k-1]] + fp.a_x(Xi_om[:,[k-1]]) * fp.dt + fp.sigmaB * fp.sdt * common_rand + (1/fp.sigmaW**2) * K_om[:,[k]] * dI_om[:,[k]]
        #%% Extended Kalman Filter    
        if fp.kalman == 1:
            Xi_kalman[:,k] = Xi_kalman[:,k-1] + fp.a_x(Xi_kalman[:,k-1]) * fp.dt + K_kalman[:,k-1] * (dZ[:,k-1] - fp.c_x(Xi_kalman[:,k-1]) * fp.dt) 
            P_kalman[:,k]  = P_kalman[:,k-1] + 2 * fp.a_dot_x(Xi_kalman[:,k-1]) * P_kalman[:,k-1] * fp.dt + (fp.sigmaB**2) * fp.dt - (K_kalman[:,k-1]**2) * fp.sigmaW**2 * fp.dt
            K_kalman[:,k]  = P_kalman[:,k]* (fp.c_dot_x(Xi_kalman[:,k-1])/fp.sigmaW**2)     
        #%% Constant gain approximation    
        if fp.const == 1:
            C = fp.c_x(Xi_const[:,[k-1]])
            dI_const[:,[k]] = dZ[:,[k]] - 0.5 * (C + np.mean(C)) * fp.dt 
            K_const[:,[k]] = np.mean((C - np.mean(C)) * Xi_const[:,[k-1]])
            Xi_const[:,[k]] = Xi_const[:,[k-1]] + fp.a_x(Xi_const[:,[k-1]]) * fp.dt + fp.sigmaB * fp.sdt * common_rand + (1/fp.sigmaW**2) * K_const[:,[k]] * dI_const[:,[k]]
        #%% SIS PF
        if fp.sis == 1:
            Xi_sis[:,[k]]   = Xi_sis[:,[k-1]] + fp.a_x(Xi_sis[:,[k-1]]) * fp.dt + fp.sigmaB * fp.sdt * common_rand 
            Zi_sis[:,[k]]   = Zi_sis[:,[k-1]] + fp.c_x(Xi_sis[:,[k]])   * fp.dt; 
            wi_sis[:,[k]]   = wi_sis[:,[k-1]] * (1/np.sqrt( 2 * np.pi * fp.sigmaW**2 * fp.dt)) * np.exp(- (Z[:,[k]] - Zi_sis[:,[k]])**2/ (2 * fp.sigmaW**2 * fp.dt))   
            # Normalizing the weights of the SIS - PF
            wi_sis[:,[k]]  = wi_sis[:,[k]]/ np.sum(wi_sis[:,[k]])
            # Deterministic resampling - as given in Budhiraja et al.
            if fp.resampling == 1:
                N_eff = np.zeros((fp.N,1))
                wi_res = np.zeros((fp.N,1))
                Xi_sis_new = np.zeros((fp.N,1))
                if np.mod(k,3)== 0:
                    sum_N_eff = 0
                    wi_cdf    = np.zeros((fp.N,1))
                    for i in np.arange(fp.N):
                        N_eff[i] = np.floor(wi_sis[i,[k]] *  fp.N) 
                        wi_res[i]= wi_sis[i,[k]] - N_eff[i]/ fp.N
                        if i == 0:
                            wi_cdf[i]= wi_res[i]
                        else:
                            wi_cdf[i]= wi_cdf[i-1] + wi_res[i]
                        if N_eff[i] > 0:
                            Xi_sis_new[int(sum_N_eff): int(sum_N_eff + N_eff[i]),:] = np.repeat(Xi_sis[i,k],int(N_eff[i]), axis = 0).reshape((-1,1))
                        sum_N_eff = sum_N_eff + N_eff[i]
                    N_res = fp.N - sum_N_eff
                    wi_cdf = wi_cdf / np.sum(wi_res)
                    wi_res = wi_res / np.sum(wi_res) 
                    
                    for j in np.arange(N_res):
                        r = np.random.rand()
                        for i in np.arange(fp.N):
                            if (r < wi_cdf[i]):
                                Xi_sis_new[int(sum_N_eff+j),:] = Xi_sis[i,[k]]
                    Xi_sis[:,[k]]  = Xi_sis_new
                    wi_sis[:,k]  = (1/fp.N) * np.ones((1,fp.N)) 
                
#                kde = stats.gaussian_kde(Xi_sis[:,k],weights = wi_sis[:,k])
#                p_sis[:,k] = kde.pdf(domain)              
#                kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
#                kde.fit(Xi_sis[:,[k]])
#                ## score_samples returns the log of the probability density
#                logprob = kde.score_samples(domain.reshape(-1,1))
#                p_sis[:,k] = np.exp(logprob)
#                p_sis[:,k] = fpf.get_kernel_pdf(Xi_sis[:,[k]], wi_sis[:,k], domain, sym = 0)
            # N_eff_sis[k] = 1 / (np.sum(wi_sis[:,[k]]**2))
    #%% Various plots
    fig1,ax1 = plt.subplots(figsize = (15,8))
    ax1.plot(ts, np.squeeze(X,axis=0), label = 'True state')
    
    plt.figure(figsize= (15,8))
    plt.plot(ts, np.squeeze(Z,axis=0), label = 'Output')
        
    ts_compare = np.array([0,150,300])
    fig2, ax = plt.subplots(nrows=1, ncols =3, figsize = (15,8), sharex =False, sharey = True)
    
    fig_particles, ax_part = plt.subplots(nrows = 1, ncols = 3, sharex= True, sharey = True, figsize = (15,8))

    if fp.coif == 1:
        sns.distplot(Xi_coif[:,ts_compare[0]],label = 'Markov semigroup', ax = ax[0])
        ax1.plot(ts,np.mean(Xi_coif,axis=0), label ='Markov semigroup')
        sns.distplot(Xi_coif[:,ts_compare[1]], label = 'Markov semigroup', ax = ax[1])
        sns.distplot(Xi_coif[:,ts_compare[2]], label = 'Markov semigroup', ax = ax[2])
        ax_part[0].plot(ts,Xi_coif[::5,:].T, color = 'r', label ='Markov semigroup')
    if fp.rkhs_N == 1:
        sns.distplot(Xi_rkhs[:,ts_compare[0]], label = 'RKHS', ax = ax[0])
        ax1.plot(ts,np.mean(Xi_rkhs,axis=0), label ='RKHS')
        sns.distplot(Xi_rkhs[:,ts_compare[1]], label = 'RKHS', ax = ax[1])
        sns.distplot(Xi_rkhs[:,ts_compare[2]], label = 'RKHS', ax = ax[2])
    if fp.om == 1:
        sns.distplot(Xi_om[:,ts_compare[0]], label = 'RKHS OM', ax = ax[0])
        ax1.plot(ts,np.mean(Xi_om,axis=0), label ='RKHS OM')
        sns.distplot(Xi_om[:,ts_compare[1]], label = 'RKHS OM', ax = ax[1])
        sns.distplot(Xi_om[:,ts_compare[2]],label = 'RKHS OM', ax = ax[2])
        ax_part[1].plot(ts,Xi_om[::5,:].T,color = 'b', label='RKHS OM')
    if fp.const == 1:
        sns.distplot(Xi_const[:,ts_compare[0]], label = 'Const', ax =ax[0])
        ax1.plot(ts,np.mean(Xi_const,axis=0), label='Const')
        sns.distplot(Xi_const[:,ts_compare[1]],label = 'Const', ax = ax[1])
        sns.distplot(Xi_const[:,ts_compare[2]],label = 'Const', ax = ax[2])
        ax_part[2].plot(ts,Xi_const[::5,:].T,color = 'b', label='Const')
    if fp.kalman == 1:
        ax[0].plot(domain,stats.norm.pdf(domain,loc = Xi_kalman[:,ts_compare[0]], scale =P_kalman[:,ts_compare[0]]), label = 'EKF')
        ax1.plot(ts,np.squeeze(Xi_kalman,axis=0), label ='EKF')
        ax[1].plot(domain,stats.norm.pdf(domain,loc = Xi_kalman[:,ts_compare[1]], scale =P_kalman[:,ts_compare[1]]),  label ='EKF')
        ax[2].plot(domain,stats.norm.pdf(domain,loc = Xi_kalman[:,ts_compare[2]], scale = P_kalman[:,ts_compare[2]]), label = 'EKF')
    if fp.sis == 1:
        #ax[0].plot(domain, p_sis[:,0], label = 'SIS PF')
        ax1.plot(ts,np.sum(wi_sis * Xi_sis,axis=0), label ='SIS PF')
        #ax[1].plot(domain, p_sis[:,99], label = 'SIS PF')
        #ax[2].plot(domain, p_sis[:,199],label = 'SIS PF')
    ax1.legend(loc = 0, framealpha = 0)
    
    ax[0].text(0.05,0.95,'$t={}$'.format(ts_compare[0]), fontsize = '18', transform = ax[0].transAxes, verticalalignment='top')
    # ax[0].legend(loc = 1, framealpha = 0)
    ax[0].set_ylabel(r'$\rho_t(x)$')
    
    ax[1].text(0.05,0.95,'$t={}$'.format(ts_compare[1]),fontsize = '18',transform = ax[1].transAxes, verticalalignment='top')
    ax[1].legend(loc = 1, framealpha = 0)
    
    ax[2].text(0.05,0.95,'$t={}$'.format(ts_compare[2]),fontsize = '18',transform = ax[2].transAxes, verticalalignment='top')
    # ax[2].legend(loc = 1, framealpha = 0)
    plt.show()
    
    ax_part[2].legend(loc =0, framealpha =0)
    plt.show()
#    plt.figure(figsize =(15,8))
#    plt.plot(Xi_om[:,0], K_om[:,1],'*')
#    plt.plot(Xi_om[:,89], K_om[:,90],'*')
#%%
# Come to this later for kernel density estimation
#    from sklearn.neighbors import KernelDensity
## instantiate and fit the KDE model
#kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
#kde.fit(x[:, None])
#
## score_samples returns the log of the probability density
#logprob = kde.score_samples(x_d[:, None])
#
#plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
#plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
#plt.ylim(-0.02, 0.22)
    
