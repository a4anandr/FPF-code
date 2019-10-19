# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 02:58:49 2019

@author: anand
"""
import numpy as np
from sympy import *

#%%
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}
figure_size = (21,8)
# font_params = {'axes.labelsize': 18,'axes.titlesize':20, 'text.fontsize': 20, 'legend.fontsize': 18, 'xtick.labelsize': 20, 'ytick.labelsize': 20}

#%% Parameters for the run
d = 1
x = symbols('x0:%d'%d)

No_runs = 1 #100

seed = 463 # np.random.randint(1000) #304 (Good seed)

#%% Flags to be set to choose which filtering methods to compare
# fpf variants 
exact  = 0      # Computes the exact gain and plots 
diff_td = 0     # Computes the gain using diff TD algorithm using eligibility vectors
diff_nl_td = 0  # Computes the gain using diff TD algorithm for nonlinear parameterization using Stochastic Approximation
finite = 0      # Computes gain using finite set of basis functions
coif   = 0      # Computes gain using Coifman kernel method
rkhs_N = 0      # Computes gain using subspace of RKHS
rkhs_dN= 0      # Computes optimal gain using RKHS 
om     = 0      # Computes gain using RKHS enforcing constant gain constraint
memory = 0      # Computes gain using RKHS with a memory parameter for previous gain
om_mem = 0      # Computes gain using const gain approx and a memory parameter for previous gain
coif_old = 0    # Computes old implementation of Coifman kernel approx. 
const  = 0      # Computes the constant gain approximation

kalman = 0      # Runs Kalman Filter for comparison

sis    = 1      # Runs Sequential Importance Sampling Particle Filter 


#%% Gain approximation parameters
# Diff TD 
T_values = [100000]

# Finite basis
basis_dim = [4, 6, 8]
basis = 'fourier' # Basis functions for the finite parameterization - poly, fourier, weighted etc. 
method = 'montecarlo' # Compute optimal parameters by numerical integration or Monte Carlo - integration or montecarlo
affine = 'y'    # If y, adds a constant vector as one of the basis functions
sa   = 'snr'     # If std, it implements standard SA, if snr, it implements stochastic newton raphson with matrix gain, if polyak, it implements Polyak averaging
nlbasis = 1      # 2 is simpler

#%% FPF methods
K_max = 100
K_min = -100

# Markov semigroup
coif_err_threshold = 1e-3
coif_iterations = 1000

# RKHS methods
eps    = 0.25   # used as eps for Coif also
Lambda = 1e-1

#%% SIS PF
resampling = 0

#%% Filtering parameters
# Filtering problem description
param_est = 1

# Parameters of the prior density \rho_B - 2 component Gaussian mixture density
m = 2      # No of components in the Gaussian mixture
sigma = np.array([0.4472, 0.4472]) # [2, 1] # Gives \sigma^2 = 0.2
mu  = np.array([-1, 1]) # [-3, 3]  
w   = np.array([0.5, 0.5])
w[-1] = 1 - sum(w[:-1])
p = 0
for m in np.arange(len(w)):
    p = p + w[m] * (1/ np.sqrt(2 * np.pi * sigma[m]**2))* exp(-(x[0] - mu[m])**2/ (2*sigma[m]**2))

# Time steps
T = 2
dt = 0.01
sdt = np.sqrt(dt)

# State process
if param_est == 1:
    a = 0       # For parameter estimation example
else:
    a = -2*x[0]    # For a stable decaying system
a_x = lambdify(x[0],a,'numpy')
a_dot = diff(a,x[0])
a_dot_x = lambdify(x[0],a_dot,'numpy')

sigmaB = 0  # No noise for parameter estimation example

# Observation process 
c = x[0]
c_x = lambdify(x, c, 'numpy')
c_dot = diff(c,x[0])
c_dot_x = lambdify(x[0],c_dot,'numpy')

sigmaW = 1

#%% Approximate filter parameters
# Number of particles used in all Monte Carlo based algorithms
N = 1000



