# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 02:58:49 2019

@author: anand
"""
import numpy as np

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 26}
figure_size = (21,8)
# font_params = {'axes.labelsize': 18,'axes.titlesize':20, 'text.fontsize': 20, 'legend.fontsize': 18, 'xtick.labelsize': 20, 'ytick.labelsize': 20}


## Parameters for the run
d_values = np.array([1]) #np.array([2,5,10]) #np.arange(1,11)
N_values = [100,500,1000] #[25,50,75,100,150,200,350,500,750,1000,5000]
No_runs = 1 #100
seed = 199 #350 #np.random.randint(1000) #350

eps = [0.01, 0.05, 0.1, 0.2 , 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# eps = [0.1, 0.2, 0.5, 0.75]
Lambda =[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
# Lambda = [1e-4, 1e-3, 1e-2]

## Flags to be set to choose which methods to compare
exact  = 1      # Computes the exact gain and plots 
diff_td = 0     # Computes the gain using diff TD algorithm using eligibility vectors
diff_nl_td = 0  # Computes the gain using diff TD algorithm for nonlinear parameterization using Stochastic Approximation
finite = 0      # Computes gain using finite set of basis functions
coif   = 0      # Computes gain using Coifman kernel method
rkhs_N = 1      # Computes gain using subspace of RKHS
rkhs_dN= 1      # Computes optimal gain using RKHS 
om     = 0      # Computes gain using RKHS enforcing constant gain constraint
memory = 0      # Computes gain using RKHS with a memory parameter for previous gain
om_mem = 0      # Computes gain using const gain approx and a memory parameter for previous gain
coif_old = 0    # Computes old implementation of Coifman kernel approx. 
const  = 0      # Computes the constant gain approximation
kalman = 0      # Runs Kalman Filter for comparison
sis    = 0      # Runs Sequential Importance Sampling Particle Filter 


# Diff TD 
T_values = [10000, 100000, 1000000] #, 1000000]

# Finite
basis_dim = 10
basis = 'weighted' # Basis functions for the finite parameterization - poly, fourier, weighted etc. 
method = 'montecarlo' # Compute optimal parameters by numerical integration or Monte Carlo - integration or montecarlo
affine = 'y'    # If y, adds a constant vector as one of the basis functions
sa   = 'std'     # If std, it implements standard SA, if snr, it implements stochastic newton raphson with matrix gain, if polyak, it implements Polyak averaging
nlbasis = 1      # 2 is simpler

# Coifman
coif_err_threshold = 1e-3
coif_iterations = 1000

 # Parameters of the prior density \rho_B - 2 component Gaussian mixture density
m = 2      # No of components in the Gaussian mixture
sigma_b =  [0.4472, 0.4472] # [2, 1] # Gives \sigma^2 = 0.2
mu_b  = [-1, 1] #   
w_b   = [0.5, 0.5]
w_b[-1] = 1 - sum(w_b[:-1])
sigma = 0.4472  # Chosen so that \sigma^2 = 0.2 as in the reference

"""
Fixed best epsilon values for each d stored as a dict. This is the most common/frequent value
observed in Hyperparams_om.csv file for each $d$ 
"""
eps_fixed_dict = { 1: 0.2,\
                   2: 0.2,\
                   3: 0.5,\
                   4: 0.5,\
                   5: 0.75,\
                   6: 0.75,\
                   7: 1,\
                   8: 1,\
                   9: 1.5,\
                   10:2,\
        }
