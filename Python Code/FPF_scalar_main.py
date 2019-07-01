# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from fpf_module import *
from sympy import *
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from IPython.display import clear_output

# Main code
# #### Defines a 2-component Gaussian mixture density $p$, generates samples from $p$ and passes to the various gain approximation functions, that return the gain vectors. If exact flag is set, the exact gain is computed via numerical integration and is used to compute mean-squared error of each of the approximations.  
# Can change parameters like - no. of independent trials, no. of particles $N$, dimensionality of the system $d$ etc.
if __name__ == '__main__':
    
    ## Flags to be set to choose which methods to compare
    exact  = 1      # Computes the exact gain and plots 
    coif   = 1      # Computes gain using Coifman kernel method
    rkhs_N = 0      # Computes gain using subspace of RKHS
    rkhs_2N= 1      # Computes optimal gain using RKHS 
    rkhs_dN= 1      # Computes optimal gain using RKHS for any arbitrary dimension d
    om     = 0      # Computes gain using RKHS enforcing constant gain constraint
    memory = 0      # Computes gain using RKHS with a memory parameter for previous gain
    om_mem = 0      # Computes gain using const gain approx and a memory parameter for previous gain
    
    coif_old = 0   # Computes old implementation of Coifman kernel approx. 
    const  = 0      # Computes the constant gain approximation
    kalman = 0      # Runs Kalman Filter for comparison
    sis    = 0      # Runs Sequential Importance Sampling Particle Filter 

    # Run parameters
    No_runs = 10
    
    # FPF parameters - No. of particles
    N = 500
    
    # System parameters
    x = Symbol('x')
    dim = 1     # dimension of the system
    c = x       # Observation function
    c_x = lambdify(x, c, 'numpy')
        
    # Parameters of the prior density p(0) - 2 component Gaussian mixture density
    m = 2      # No of components in the Gaussian mixture
    sigma = [0.4472, 0.4472]
    mu  = [-1, 1]
    w   = [0.5, 0.5]
    w[-1] = 1 - sum(w[:-1])
    p = 0
    mu_eq = 0  # Equivalent mean of the density p
    for m in range(len(w)):
        p = p + w[m] * (1/ np.sqrt(2 * math.pi * sigma[m]**2))* exp(-(x - mu[m])**2/ (2* sigma[m]**2))
    p_vec = lambdify(x, p, 'numpy')
    
    mse_coif = np.zeros(No_runs)
    mse_rkhs_N = np.zeros(No_runs)
    mse_rkhs_2N = np.zeros(No_runs)
    mse_rkhs_dN = np.zeros(No_runs)
    mse_om   = np.zeros(No_runs)
    mse_coif_old = np.zeros(No_runs)
    for run in range(No_runs):
        clear_output()
        print('Run ',run)
        Xi  = get_samples(N, mu, sigma, w, dim)
        if dim == 1:
            Xi = np.sort(Xi,kind = 'mergesort')
        C   = c_x(Xi)
# To check consistency with Matlab code output - using the same samples 
#         Xi = np.loadtxt('Xi.txt')
#         Xi = np.sort(Xi,kind = 'mergesort')
#         Xi = np.reshape(Xi,(-1,1))
#         plt.figure()
#         sns.distplot(Xi)
#         plt.show()

        if exact == 1:
            K_exact =  np.zeros((N,dim))
            K_exact[:,0] =  gain_num_integrate(Xi, c, p)  # Uses scipy.integrate function
            # K_exact[:,0]  = gain_exact(Xi, c, p)   # Manual numerical integration

        if coif ==1:
            eps_coif = 0.1
            Phi = np.zeros(N)
            K_coif = gain_coif(Xi, C, eps_coif, Phi, diag = 0)
            if exact == 1:
                mse_coif[run] = mean_squared_error(K_exact, K_coif, p)

        if rkhs_N == 1:
            eps_rkhs_N = 0.1
            Lambda_rkhs_N = 1e-3
            K_rkhs_N = gain_rkhs_N(Xi, C, eps_rkhs_N, Lambda_rkhs_N, diag = 0)
            if exact == 1:
                mse_rkhs_N[run] = mean_squared_error(K_exact, K_rkhs_N, p)

        if rkhs_2N == 1:
            eps_rkhs_2N = 0.1
            Lambda_rkhs_2N = 1e-3
            K_rkhs_2N = gain_rkhs_2N(Xi, C, eps_rkhs_2N, Lambda_rkhs_2N, diag = 0)
            if exact == 1:
                mse_rkhs_2N[run] = mean_squared_error(K_exact, K_rkhs_2N, p)
                
        if rkhs_dN == 1:
            eps_rkhs_dN = 0.1
            Lambda_rkhs_dN = 1e-3
            K_rkhs_dN = gain_rkhs_dN(Xi, C, eps_rkhs_dN, Lambda_rkhs_dN, diag = 0)
            if exact == 1:
                mse_rkhs_dN[run] = mean_squared_error(K_exact, K_rkhs_dN, p)

        if om == 1:
            eps_om = 0.1
            Lambda_om = 1e-3
            K_om = gain_rkhs_om(Xi, C, eps_om, Lambda_om, diag = 0)
            if exact == 1:
                mse_om[run] = mean_squared_error(K_exact, K_om, p)
                
        if coif_old == 1:
            eps_coif_old = 0.1
            Phi = np.zeros(N)
            K_coif_old = gain_coif_old(Xi, C, eps_coif, Phi, diag = 0)
            if exact == 1:
                mse_coif_old[run] = mean_squared_error(K_exact, K_coif_old, p)
         
    print('\n')
     #Plotting the histogram of the mse from the various trials
    plt.figure()
    if exact == 1 & coif == 1:
        print('MSE for Markov kernel approx', np.mean(mse_coif))
        sns.distplot(mse_coif,label='Coifman new')
    if exact == 1 & rkhs_N == 1:
        print('MSE for RKHS N', np.mean(mse_rkhs_N))
        sns.distplot(mse_rkhs_N, label = 'RKHS N')
    if exact == 1 & rkhs_2N == 1:
        print('MSE for RKHS 2N', np.mean(mse_rkhs_2N))
        sns.distplot(mse_rkhs_2N, label = 'RKHS 2N')
    if exact == 1 & rkhs_dN == 1:
        print('MSE for RKHS dN', np.mean(mse_rkhs_dN))
        sns.distplot(mse_rkhs_dN, label = 'RKHS (d+1)*N')
    if exact == 1 & om == 1:
        print('MSE for RKHS OM', np.mean(mse_om))
    if exact == 1 & coif_old == 1:
        print('MSE for old Markov kernel', np.mean(mse_coif_old))
        sns.distplot(mse_coif,label='Coifman old')
    plt.legend()
    plt.show()
    
    ### Displaying the plots
    marker_size  = 3
    plt.rc('text', usetex=True)
    fig,ax1 = plt.subplots()
    if exact == 1:
        ax1.plot(Xi, K_exact, 'b.', markersize = marker_size, label ='Exact gain')
        # ax1.plot(Xi, K_num_int, 'k^', markersize = marker_size, label ='Num int. gain')
    if rkhs_N == 1:
        ax1.plot(Xi, K_rkhs_N, 'r.', markersize = marker_size, label = 'RKHS approx. N')
    if rkhs_2N == 1:
        ax1.plot(Xi, K_rkhs_2N, 'c.', markersize = marker_size, label = 'RKHS approx. 2N')
    if rkhs_dN == 1:
        ax1.plot(Xi, K_rkhs_dN, 'm.', markersize = marker_size, label = 'RKHS approx. dN')
    if coif == 1:
        ax1.plot(Xi, K_coif, 'g.', markersize = marker_size, label ='Markov kernel approx.')
    if om == 1:
        ax1.plot(Xi, K_om, 'm.', markersize = marker_size, label = 'RKHS OM')
    if coif_old == 1:
        ax1.plot(Xi, K_coif_old, 'y.', markersize = marker_size, label ='Old Markov kernel')
    ax2 =ax1.twinx()
    ax2.plot(np.arange(-2,2,0.01), p_vec(np.arange(-2,2,0.01)),'k.-', markersize =1, label = r'$\rho(x)$')
    ax2.set_ylabel(r'$\rho(x)$')
    ax2.legend(loc=1)
    ax1.set_xlabel('Particle Locations')
    ax1.set_ylabel('Gain $K(x)$')
    ax1.legend()
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()     

   