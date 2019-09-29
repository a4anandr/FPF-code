# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:59:04 2019

@author: anand
"""

# =============================================================================
# # FPF Gain approximation module
# =============================================================================
import numpy as np
from sympy import *
# import sympy as sp
from scipy.spatial.distance import pdist,squareform
from scipy.stats import norm
import scipy.integrate as integrate
import math

from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib import rc

import parameters

rc('text',usetex = True)
rc('font', **parameters.font)
import seaborn as sns

from IPython.display import clear_output

#%matplotlib notebook
#%matplotlib inline

## Defining some functions
# =============================================================================
# ### get_samples() - Function to generate samples from a multi-dimensional 1d- Gaussian mixture $\times$  (d-1) independent Gaussian distribution 
# =============================================================================
def get_samples(N, mu, sigma_b, w, dim, gm = 1, sigma = None, seed = None):
    np.random.seed(seed)
    Xi  = np.zeros((N,dim))
    for i in range(N):
        for d in range(gm):
            if np.random.uniform() <= w[0]:
                Xi[i,d] = mu[0]  + sigma_b[0] * np.random.normal()
            else:
                Xi[i,d]  = mu[1]  + sigma_b[1] * np.random.normal()
        for d in range(gm, dim):
            Xi[i,d] = sigma * np.random.normal()
    return Xi

# =============================================================================
# ### mean_squared_error() - Function to compute the mean square error in gain function approximation
# =============================================================================
def mean_squared_error(K_exact, K_approx):
    N = len(K_exact)
    mse = (1/N) * np.linalg.norm(K_exact - K_approx)**2
    # mse2 = np.sum(((K_exact - K_approx)**2) *np.concatenate(p_vec(Xi)))
    return mse

## Defining basis functions
# =============================================================================
# ### get_poly() - Function to compute and return polynomial basis functions upto the passed degree on the passed data points
# =============================================================================
def get_poly(Xi, d):
    N = Xi.shape[0]
    Psi = np.array(np.zeros((N,d)))
    Psi_x = np.array(np.zeros((N,d)))
    
    # print('Using polynomial basis')
    for i,n in enumerate(np.arange(1,d+1)):
        Psi[:,i] = np.reshape(Xi**n,-1)
        Psi_x[:,i]= np.reshape(n * Xi**(n-1),-1)
    
    return Psi,Psi_x

# =============================================================================
# ### get_fourier() - Function to compute and return Fourier basis functions upto the passed degree on the passed data points
# =============================================================================
def get_fourier(Xi, d):
    N = Xi.shape[0]
    Psi = np.array(np.zeros((N,d)))
    Psi_x = np.array(np.zeros((N,d)))
    
    print('Using Fourier basis')
    for n in np.arange(0,d,2):
        Psi[:,n] = np.reshape(np.sin((n/2 +1) * Xi),-1)
        Psi_x[:,n] = np.reshape(np.cos((n/2 +1) * Xi),-1)
        Psi[:,n+1] = np.reshape(np.cos((n/2 +1) * Xi),-1)
        Psi_x[:,n+1] = np.reshape(-np.sin((n/2 +1) * Xi),-1)
        
    return Psi,Psi_x

# =============================================================================
# ### get_weighted_poly() - Function to compute and return weighted polynomial basis functions upto the passed degree on the passed data points
# =============================================================================
def get_weighted_poly(Xi,d,mu,sigma):
    N = Xi.shape[0]
    Psi = np.array(np.zeros((N,d)))
    Psi_x = np.array(np.zeros((N,d)))
    
    # print('Using weighted polynomial basis')
    p_basis = np.zeros((N,len(mu)))
    p_diff_basis = np.zeros((N,len(mu)))
    for i in np.arange(len(mu)):
        p_basis[:,i] = np.reshape(np.exp(-(Xi - mu[i])**2/ (2 * sigma[i]**2)),-1)
        p_diff_basis[:,i] = np.reshape( -((Xi - mu[i])/sigma[i]**2).reshape(-1) * p_basis[:,i], -1)
    
    for n in np.arange(0,d,2):
        Psi[:,n] = (Xi**(n/2 +1)).reshape(-1) * p_basis[:,0]
        Psi_x[:,n] = (((n/2 +1)* Xi**(n/2)).reshape(-1) * p_basis[:,0]) + ((Xi**(n/2 +1)).reshape(-1) * p_diff_basis[:,0])
        Psi[:,n+1] = (Xi**(n/2 +1)).reshape(-1) * p_basis[:,1]
        Psi_x[:,n+1] = (((n/2 +1)* Xi**(n/2)).reshape(-1) * p_basis[:,1]) + ((Xi**(n/2 +1)).reshape(-1) * p_diff_basis[:,1])
    
    return Psi,Psi_x
# =============================================================================
# ### append_const_basis() - Function to append a constant function to make it an affine parameterization
# =============================================================================
def append_const_basis(Xi,Psi,Psi_x):
    Psi = np.append(Psi,Xi, axis =1)
    Psi_x = np.append(Psi_x, np.ones((len(Xi),1)), axis =1)         
    return Psi,Psi_x

## Different gain approximation algorithms
# =============================================================================
# ### gain_diff_td - Function to approximate the gain function with a finite set of basis
# =============================================================================
def gain_diff_td(Xi, c, w, mu, sigma, p, x, d, basis, affine, diag = 0):
    start = timer()
    N,dim = Xi.shape
    K = np.zeros((N,dim))

    # Defining the potential function U and its derivatives 
    U = -log(p)
    Udot = diff(U,x[0])
    Uddot = diff(Udot, x[0])
    
    p_x = lambdify(x[0],p,'numpy')
    U_x = lambdify(x[0],U,'numpy')
    Udot_x = lambdify(x[0],Udot,'numpy')
    Uddot_x = lambdify(x[0], Uddot,'numpy')
    
    # Derivative of c(x)
    cdot = diff(c[0],x[0])
    cdot_x = lambdify(x[0],cdot, 'numpy')
    
    # Running a discretized Langevin diffusion
    T = parameters.T
    dt = 0.01
    sdt = np.sqrt(dt)
    sqrt2 = np.sqrt(2)
    Phi = np.zeros(T)
    
    if affine.lower() == 'y':
        d+= 1
    varphi = np.zeros((T,d))
    b = np.zeros((T,d))
    M = np.zeros((T,d,d))
    M[0,:,:] = 1e-3 * np.eye(d)
    beta_td = np.zeros((T,d))
    for n in np.arange(1,T):
        Phi[n] = Phi[n-1] - Udot_x(Phi[n-1]) * dt + sqrt2 * np.random.randn() * sdt 
        if basis == 'poly':
            Psi,Psi_x = get_poly(np.expand_dims(Phi[n-1],axis=1),d)
        elif basis == 'fourier':
            if affine.lower() == 'y':
                Psi,Psi_x = get_fourier(np.expand_dims(Phi[n-1],axis=1),d-1)
                Psi,Psi_x = append_const_basis(np.expand_dims(np.expand_dims(Phi[n-1],axis=1),axis=1),Psi,Psi_x)
            else:
                Psi,Psi_x = get_fourier(np.expand_dims(Phi[n-1],axis=1),d)
        elif basis == 'weighted':
            if affine.lower() == 'y':
                Psi,Psi_x = get_weighted_poly(np.expand_dims(Phi[n-1],axis=1),d-1,mu,sigma)
                Psi,Psi_x = append_const_basis(np.expand_dims(np.expand_dims(Phi[n-1],axis=1),axis=1),Psi,Psi_x)
            else:
                Psi,Psi_x = get_weighted_poly(np.expand_dims(Phi[n-1],axis=1),d,mu,sigma)
        
        varphi[n,:] = varphi[n-1,:] + (- Uddot_x(Phi[n-1]) * varphi[n-1,:] + Psi_x) * dt
        b[n,:] = b[n-1,:] + varphi[n-1,:] * cdot_x(Phi[n-1]) * dt
        M[n,:,:] = M[n-1,:,:] + np.expand_dims( Psi_x.T * Psi_x * dt, axis =0)
        beta_td[n,:] = np.linalg.solve(M[n,:],b[n,:])
    beta_final = beta_td[n,:]
    
    if basis == 'poly':
        Psi,Psi_x = get_poly(Xi,d)
    elif basis == 'fourier':
        if affine.lower() == 'y':
            Psi,Psi_x = get_fourier(Xi,d-1)
            Psi,Psi_x = append_const_basis(Xi,Psi,Psi_x)
        else:
            Psi,Psi_x = get_fourier(Xi,d)
    elif basis == 'weighted':
        if affine.lower() == 'y':
            Psi,Psi_x = get_weighted_poly(Xi,d-1,mu,sigma)
            Psi,Psi_x = append_const_basis(Xi,Psi,Psi_x)
        else:
            Psi,Psi_x = get_weighted_poly(Xi,d,mu,sigma)

    for i in np.arange(Psi_x.shape[1]):
        K = K + beta_final[i] * np.reshape(Psi_x[:,i],(-1,1))
            
    if diag == 1:
        plt.figure()
        ax1 = sns.distplot(Phi, label = 'Langevin')
        ax1.plot(np.arange(-3,3,0.1),p_x(np.arange(-3,3,0.1)),'--',label ='$\\rho(x)$')
        ax1.legend()
        plt.title('Histogram of Langevin samples vs $\\rho(x)$')
        plt.show()
        
        
        plt.figure()
        plt.plot(Xi, Psi,'*')
        plt.title('Basis funtions')
        plt.show()
        
        plt.figure()
        plt.plot(Xi, Psi_x,'^')
        plt.title('Basis derivatives')
        plt.show()
        
    end = timer()
    print('Time taken for gain_finite()' , end - start)
    
    return K

# =============================================================================
# ### gain_finite_integrate - Function to approximate the gain function with a finite set of basis
# =============================================================================
def gain_finite_integrate(c_x, w, mu, sigma, d, basis, affine, diag = 0):
    start = timer()
    
    step = 0.05
    X = np.expand_dims(np.arange(-3,3,step),axis=1)
    N,dim = X.shape
    
    K = np.zeros((N,dim))
    
    p  = np.zeros((N,1))
    for i in np.arange(len(w)):
        p = p + w[i] * np.exp(-(X - mu[i])**2/ (2 * sigma[i]**2))
    p = p/np.sum(p)
    
    if basis == 'poly':
        Psi,Psi_x = get_poly(X, d)
    elif basis == 'fourier':
        Psi,Psi_x = get_fourier(X, d)
        if affine.lower() == 'y':
            Psi,Psi_x = append_const_basis(X, Psi, Psi_x) 
    elif basis == 'weighted':
        Psi,Psi_x = get_weighted_poly(X,d, mu,sigma)
        if affine.lower() == 'y':
            Psi,Psi_x = append_const_basis(X, Psi, Psi_x)
             
    eta = np.mean(c_x(X)* p)
    Y = c_x(X) -eta
    
    b_psi        = np.dot(np.squeeze(Y * p,axis=2), Psi).T
    M_psi        = np.dot(Psi_x.T * p.reshape(-1), Psi_x)
   
    if(np.linalg.det(M_psi)!=0):
        beta_psi     = np.linalg.solve(M_psi, b_psi) 
    else:
        beta_psi     = np.linalg.lstsq(M_psi,b_psi)[0]
    
    for i in np.arange(Psi_x.shape[1]):
        K = K + beta_psi[i] * np.reshape(Psi_x[:,i],(-1,1))
            
    if diag == 1:
        plt.figure()
        plt.plot(X, Psi,'*')
        plt.title('Basis funtions')
        plt.show()
        
        plt.figure()
        plt.plot(X, Psi_x,'^')
        plt.title('Basis derivatives')
        plt.show()

    end = timer()
    print('Time taken for gain_finite_integrate()' , end - start)
    
    return X,K    

# =============================================================================
# ### gain_finite - Function to approximate the gain function with a finite set of basis
# =============================================================================
def gain_finite(Xi, C, mu, sigma, d, basis, affine, diag = 0):
    start = timer()
    N,dim = Xi.shape
    
    K = np.zeros((N,dim))
    Psi = np.array(np.zeros((N,d)))
    Psi_x = np.array(np.zeros((N,d)))

    if basis == 'poly':
        Psi,Psi_x = get_poly(Xi, d)
    elif basis == 'fourier':
        Psi,Psi_x = get_fourier(Xi, d)
        if affine.lower() == 'y':
            Psi,Psi_x = append_const_basis(Xi, Psi, Psi_x) 
    elif basis == 'weighted':
        Psi,Psi_x = get_weighted_poly(Xi,d, mu,sigma)
        if affine.lower() == 'y':
            Psi,Psi_x = append_const_basis(Xi, Psi, Psi_x)
            
    eta = np.mean(C)
    Y = (C -eta)
    
    b_psi        = (1/N) * np.dot(Psi.T, Y)
    M_psi        = (1/N) * np.dot(Psi_x.T, Psi_x)
   
    if(np.linalg.det(M_psi)!=0):
        beta_psi     = np.linalg.solve(M_psi, b_psi) 
    else:
        beta_psi     = np.linalg.lstsq(M_psi,b_psi)[0]
    
    for i in np.arange(Psi_x.shape[1]):
        K = K + beta_psi[i] * np.reshape(Psi_x[:,i],(-1,1))
            
    if diag == 1:
        plt.figure()
        plt.plot(Xi, Psi,'*')
        plt.title('Basis funtions')
        plt.show()
        
        plt.figure()
        plt.plot(Xi, Psi_x,'^')
        plt.title('Basis derivatives')
        plt.show()
        
    end = timer()
    print('Time taken for gain_finite()' , end - start)
    
    return K

# =============================================================================
# ### gain_rkhs_2N() - Function to approximate FPF gain using optimal RKHS method uses the extended representer theorem in - https://www.sciencedirect.com/science/article/pii/S0377042707004657?via%3Dihub
# =============================================================================
def gain_rkhs_2N(Xi, C, epsilon, Lambda, diag = 0):
    start = timer()
    N,dim = Xi.shape
    K = np.zeros((N,dim))
    Ker_x = np.array(np.zeros((N,N,dim)))
    Ker_xy = np.array(np.zeros((N,N,dim)))
    
    Ker = np.exp(- squareform(pdist(Xi,'euclidean'))**2/ (4 * epsilon))
    for i in range(N):
        for j in range(N):
            Ker_x[i,j,:] = -(Xi[i,:]-Xi[j,:]) * Ker[i,j] / (2 * epsilon)
            Ker_xy[i,j,:] = -(((Xi[i,:] - Xi[j,:])**2) / (2 * epsilon) -1) * Ker[i,j] / (2 * epsilon) # Negative of the second Gaussian derivative, as this is K_xy and not K_x2
    
    eta = np.mean(C)
    Y = (C -eta)
    
    # Constructing block matrices for future use
    # K_big      = [ Ker Ker_x ; Ker_x' Ker_x_y];
    # K_thin_yxy = [ Ker_x ; Ker_x_y]; 
    # K_thin_x   = [ Ker ; Ker_x'];
    K_big      = np.concatenate((np.concatenate((Ker,np.transpose(np.reshape(Ker_x,(N,N)))),axis = 1), np.concatenate((np.reshape(Ker_x,(N,N)), np.reshape(Ker_xy,(N,N))),axis =1)))
    K_thin_yxy = np.concatenate((np.transpose(np.reshape(Ker_x,(N,N))), np.reshape(Ker_xy,(N,N))))
    # K_thin_xxy = np.concatenate((Ker_x,Ker_xy), axis = 1)
    K_thin_x   = np.concatenate((Ker, np.reshape(Ker_x,(N,N))))
    
    # b used in the extended representer theorem algorithm - searching over all of the Hilbert space H
    b_2N        = (1/N) * np.dot(K_thin_x, Y)
    M_2N        = Lambda * K_big + (1/N) * np.dot(K_thin_yxy, np.transpose(K_thin_yxy))
    if(np.linalg.det(M_2N)!=0):
        beta_2N     = np.linalg.solve(M_2N, b_2N) 
    else:
        beta_2N     = np.linalg.lstsq(M_2N,b_2N)[0]
    
    for i in range(N):
        for j in range(N):
            K[i,:] = K[i,:] + beta_2N[j] * Ker_x[i,j,:] + beta_2N[N+j] * Ker_xy[i,j,:]
            
    if diag == 1:
        plt.figure()
        plt.plot(Xi, Ker[:,0],'r*')
        plt.plot(Xi, Ker_x[:,0], 'b*')
        plt.plot(Xi, Ker_xy[:,0],'k*')
        plt.show()
            
    end = timer()
    print('Time taken for gain_rkhs_2N()' , end - start)
    
    return K

# =============================================================================
# ### gain_rkhs_dN() - Extension to d-dimensions
# =============================================================================
def gain_rkhs_dN(Xi, C, epsilon, Lambda, diag = 1):
    start = timer()
    
    N,dim = Xi.shape
    K = np.zeros((N,dim))
    Ker_x  = np.array(np.zeros((N,N,dim)))
    Ker_xy = np.array(np.zeros((N,N, dim+1, dim+1)))
    # Ker_xy = np.array(np.zeros((N,N, dim, dim)))
    
    K_big  = np.array(np.zeros(((dim+1)*N, (dim+1)*N)))
    K_thin_x =np.array(np.zeros(((dim+1)*N, N)))
    K_thin_xy = np.array(np.zeros(((dim+1)*N, dim * N)))
    
    Ker = np.exp(- squareform(pdist(Xi,'euclidean'))**2/ (4 * epsilon))
    for i in range(N):
        for j in range(N):
            Ker_x[i,j,:] = -(Xi[i,:]-Xi[j,:]) * Ker[i,j] / (2 * epsilon)
            # Ker_x2[i,j,:] = -(((Xi[i,:] - Xi[j,:])**2)  / (2 * epsilon) -1) * Ker[i,j] / (2 * epsilon) # Negative of the second Gaussian derivative, as this is K_xy and not K_x2
    
    for d_i in range(dim + 1):
        for d_j in range(dim + 1):
            if d_i == 0:
                if d_j == 0:
                    K_big[d_i * N : (d_i + 1) *N, d_j * N : (d_j +1) * N ] = Ker
                else:
                    K_big[d_i * N : (d_i + 1) *N, d_j * N : (d_j +1) * N ] = np.transpose(Ker_x[:,:,d_j-1])
            elif d_j == 0:
                K_big[d_i * N : (d_i + 1) *N, d_j * N : (d_j +1) * N ] = Ker_x[:,:,d_i-1]
            elif d_i == d_j:
                for i in range(N):
                    for j in range(N):
                        Ker_xy[i,j, d_i, d_j] = -(((Xi[i,d_i-1] - Xi[j,d_i-1])**2)  / (2 * epsilon) -1) * Ker[i,j] / (2 * epsilon) 
                K_big[d_i * N : (d_i + 1) *N, d_j * N : (d_j +1) * N ] = Ker_xy[:,:,d_i, d_j]
            else:
                for i in range(N):
                    for j in range(N):
                        Ker_xy[i,j, d_i, d_j] = -((Xi[i,d_i-1] - Xi[j,d_i-1])* (Xi[i,d_j-1] - Xi[j,d_j-1])) / (2 * epsilon) * Ker[i,j] / (2 * epsilon) # Negative of the second Gaussian derivative, as this is K_xy and not K_x2        
                K_big[d_i * N : (d_i + 1) *N, d_j * N : (d_j +1) * N ] = Ker_xy[:,:,d_i, d_j]
            
    for d_i in range(dim + 1):
        if d_i == 0:
            K_thin_x[d_i *N :(d_i+1)*N,: ] = Ker
        else:
            K_thin_x[d_i *N :(d_i+1)*N,: ] = Ker_x[:,:,d_i-1]
            
    for d_i in range(dim+1):
        for d_j in range(dim):
            if d_i == 0:
                K_thin_xy[d_i * N :(d_i+1)*N, d_j * N : (d_j+1) *N] = np.transpose(Ker_x[:,:,d_j])
            else:
                K_thin_xy[d_i * N :(d_i+1)*N, d_j * N :(d_j+1) *N] = Ker_xy[:,:,d_i,d_j+1]
    
    eta = np.mean(C)
    Y = (C -eta)
    # b used in the extended representer theorem algorithm - searching over all of the Hilbert space H
    b_dN        = (1/N) * np.dot(K_thin_x, Y)
    M_dN        = Lambda * K_big + (1/N) * np.dot(K_thin_xy, np.transpose(K_thin_xy))
    if np.linalg.det(M_dN)!= 0:
        beta_dN     = np.linalg.solve(M_dN, b_dN)   
    else:
        beta_dN     = np.linalg.lstsq(M_dN,b_dN)[0]
    
    for i in range(N):
        for j in range(N):
            K[i,:] = K[i,:] + beta_dN[j] * Ker_x[i,j,:] 
            for d_i in range(dim):
                K[i,:] = K[i,:] + beta_dN[(d_i + 1) *N + j] * Ker_xy[i,j,(d_i+1),1:]
            
    if diag == 1:
        plt.figure()
        plt.plot(Xi[:,0], Ker[:,1],'r*')
        plt.plot(Xi, Ker_x[:,1,0], 'b*')
#         plt.plot(Xi, Ker_xy[:,1,1,1],'k*')
        plt.show()
            
    end = timer()
    print('Time taken for gain_rkhs_dN()' , end - start)
    return K

# =============================================================================
# ### gain_rkhs_N() - Function to approximate FPF gain using subspace RKHS method  
# # uses normal representer theorem, obtains optimal solution on a subspace of RKHS.
# =============================================================================
    def gain_rkhs_N(Xi, C, epsilon, Lambda, diag = 0):
        start = timer()
        
        N,dim = Xi.shape
        K = np.zeros((N,dim))
        Ker_x = np.array(np.zeros((N,N,dim)))
        Ker_x_sum = np.zeros((N,N))
        
        Ker = np.exp(- squareform(pdist(Xi,'euclidean'))**2/ (4 * epsilon))    
        for i in range(N):
            for j in range(N):
                Ker_x[i,j,:] = -(Xi[i,:]-Xi[j,:]) * Ker[i,j] / (2 * epsilon)
        
        eta = np.mean(C)
        Y = (C -eta)
        
        b_N = (1/ N) * np.dot(Ker,Y)
        for d in np.arange(dim):
            Ker_x_sum+= np.dot(Ker_x[:,:,d], Ker_x[:,:,d].transpose())
        M_N = Lambda * Ker + (1/ N) * Ker_x_sum
        if(np.linalg.det(M_N)!=0):
            beta_N = np.linalg.solve(M_N,b_N)
        else:
            beta_N = np.linalg.lstsq(M_N,b_N)[0]
            
        for i in range(N):
            for j in range(N):
                K[i,:] = K[i,:] + beta_N[j] * Ker_x[i,j,:]
                
        if diag == 1:
            plt.figure()
            plt.plot(Xi, Ker[:,100],'r*')
            plt.plot(Xi, Ker_x[:,100,:], 'b*')
            plt.show()
                
        end = timer()
        print('Time taken for gain_rkhs_N()' , end - start)
        
        return K

# =============================================================================
# ### gain_exact() - Function to compute the exact FPF gain by numerical integration
# Algorithm
#\begin{equation} 
#\text{K}(x) =  - \frac{1}{p(x)} \int_{-\infty}^{x} (c(y) - \hat{c}) p(y) dy
#\end{equation}
# =============================================================================
def gain_exact(Xi, c, p):
    start = timer()
    
    N = len(Xi)
    K = np.zeros(N)
    integral = np.zeros(N)
    
    step = 0.01
    xmax = max(mu) + 10
    
    p_vec = lambdify(x, p, 'numpy')
    c_vec = lambdify(x, c, 'numpy')
    cp    = lambdify(x, c*p, 'numpy')
    c_hat = integrate.quad(cp, -np.inf, np.inf)[0]
    
    for i in range(N):
        integral[i] = 0
        for xj in np.arange(Xi[i], xmax + 10,  step):
            integral[i] = integral[i] + p_vec(xj) * ( c_vec(xj) - c_hat) * step
        K[i] = integral[i]/ p_vec(Xi[i])
            
    end = timer()
    print('Time taken' , end - start)
    return K

# =============================================================================
# Using scipy.integrate.quad
# =============================================================================
def gain_num_integrate(Xi, c, p, x, d=0):
    start = timer()
    
    N = len(Xi)
    K = np.zeros(N)
    integral = np.zeros(N)
    p_x = lambdify(x[0], p, 'numpy')
    cp_x  = lambdify(x[0], c*p, 'numpy')
    c_hat = integrate.quad(cp_x, -np.inf, np.inf)[0]
    integrand_x = lambdify(x[0], p * (c - c_hat) , 'numpy')
    integrand = lambda x: integrand_x(x)
   
    for i in range(N):
        if Xi.shape[1] == 1:
            integral[i] = integrate.quad( integrand, -np.inf, Xi[i])[0]
            K[i] = - integral[i]/ p_x(Xi[i])
        else:
            integral[i] = integrate.quad( integrand, -np.inf, Xi[i,d])[0]
            K[i] = - integral[i]/ p_x(Xi[i,d])
    # K = np.reshape(K,(N,1))
    
    end = timer()
    print('Time taken for gain_num_integrate()' , end - start)
    return K

# =============================================================================
# ### gain_coif() - Function to approximate FPF gain using Markov kernel approx. method -
# Based on the Markov semigroup approximation method in https://arxiv.org/pdf/1902.07263.pdf
# 
# Algorithm  
# \begin{enumerate}
# \item Calculate $g_{ij} = \exp(-|X^i - X^j|^2/ 4\epsilon)$ for $i,j = 1$ to $N$  
# \item Calculate $k_{ij} = \frac{g_{ij}}{\sqrt{\sum_l g_{il}}\sqrt{\sum_l g_{jl}}}$  
# \item Calculate $d_i = \sum_j k_{ij}$  
# \item Calculate $\text{T}_{ij} = \frac{k_{ij}}{d_i}$  
# \item Calculate $\pi_i = \frac{d_i}{\sum_j d_j}$  
# \item Calculate $ \hat{h} = \sum_{i = 1}^N \pi_i h(X^i)$  
# \item Until convergence, $\Phi_i = \sum_{j=1}^N \text{T}_{ij} \Phi_j + \epsilon (h - \hat{h})$  
# \item Calculate $r_i = \Phi_i + \epsilon h_i$  
# \item Calculate $s_{ij} = \frac{1}{2\epsilon} \text{T}_{ij} (r_j - \sum_{k=1}^N \text{T}_{ik} r_k)$  
# \item Calulate $\text{K}_i  = \sum_j s_{ij} X^j$
# \end{enumerate}
# 
# =============================================================================
def gain_coif(Xi, C, epsilon, Phi, No_iterations = parameters.coif_iterations, diag = 0):
    start = timer()
    
    N,dim = Xi.shape
    k = np.zeros((N,N))
    # k2 = np.zeros((N,N))
    K = np.zeros((N,dim))
    d = np.zeros(N)
    T = np.zeros((N,N))
    Phi = np.zeros(N)
    sum_term = np.zeros((N,dim))
    max_diff = 1
    
    iterations = 1
    
    g = np.exp(- squareform(pdist(Xi,'euclidean'))**2/ (4 * epsilon))    
#    for i in range(N):
##        for j in range(N):
##            k[i,j] = g[i,j] / (np.sqrt( (1/N) * sum(g[i,:])) * np.sqrt( (1/N)* sum(g[j,:])))
#        k[i,:] = np.divide(g[i,:], np.sqrt((1/N) * np.sum(g,axis = 0)))
    k = np.divide(g, np.sqrt((1/N) * np.sum(g,axis =0)))
   
#    for j in range(N):
#        k[:,j] = np.divide(k[:,j], np.sqrt((1/N) * np.sum(g,axis = 0)))
    k = np.divide(k.T, np.sqrt((1/N) * np.sum(g,axis =0)))
    
#    for i in range(N):
#        d[i] = np.sum(k[i,:])
#        T[i,:] = np.divide(k[i,:], np.sum(k[i,:])) 
    d = np.sum(k, axis=0)
    T = np.divide(k, np.sum(k,axis=0).reshape((-1,1)))   
    
    pi = np.divide(d, np.sum(d))
    C_hat = np.dot(pi, C)
                      
    while((max_diff > parameters.coif_err_threshold) & ( iterations < No_iterations )):
        Phi_new = np.dot(T,Phi) + (epsilon * np.concatenate(C - C_hat)).transpose() 
        max_diff = max(Phi_new - Phi) - min(Phi_new - Phi)
        Phi  = Phi_new
        iterations += 1
#        print(iterations)
#        print(max_diff)
    
    r = Phi + epsilon * np.concatenate(C)
    sum_term = np.dot(T, r)
    for i in range(N):
        #sum_term[i] = np.dot(T[i,:],r)
        K[i,:] = np.zeros(dim)
        for j in range(N):
            K[i,:] = K[i,:] + (1/ (2 * epsilon)) * T[i,j] * (r[j] - sum_term[i]) * Xi[j,:]                                  
    if diag == 1:
        plt.figure()
        plt.plot(Xi, g[1,:], 'r*')
        plt.show()
    
    end = timer()
    print('Time taken for gain_coif()' , end - start)
    return K

# =============================================================================
# Slightly older version of Markov kernel approximation - from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7799105
# =============================================================================
def gain_coif_old(Xi, C, epsilon, Phi, No_iterations =50000, diag = 0):
    start = timer()
    
    N,dim = Xi.shape
    k = np.zeros((N,N))
    K = np.zeros((N,dim))
    d = np.zeros(N)
    T = np.zeros((N,N))
    Phi = np.zeros(N)
    sum_term = np.zeros((N,dim))
    max_diff = 1
        
    iterations = 1
    
    g = np.exp(- squareform(pdist(Xi,'euclidean'))**2/ (4 * epsilon))
    for i in range(N):
        for j in range(N):
            k[i,j] = g[i,j] / (np.sqrt( (1/N) * sum(g[i,:])) * np.sqrt( (1/N)* sum(g[j,:])))
        d[i] = np.sum(k[i,:])
        T[i,:] = np.divide(k[i,:], np.sum(k[i,:]))
                      
    while((max_diff > 0) & ( iterations < No_iterations )):
        Phi_new = np.dot(T,Phi) + (epsilon * np.concatenate(C)).transpose() 
        max_diff = max(Phi_new - Phi) - min(Phi_new - Phi)
        Phi  = Phi_new
        iterations += 1
    
    for i in range(N):
        sum_term[i,:] = np.dot( T[i,:], Xi)
        K[i,:] = np.zeros(dim)
        for j in range(N):
            K[i,:] = K[i,:] + (1/ (2 * epsilon)) * T[i,j] * Phi[j,] * (Xi[j,:] - sum_term[i,:])   
            
    if diag == 1:
        plt.figure()
        plt.plot(Xi, g[1,:], 'r*')
        plt.show()
    
    end = timer()
    print('Time taken for gain_coif_old()' , end - start)
    
    return K

# =============================================================================
# ### gain_rkhs_om() - Function to approximate FPF gain using RKHS OM method - Adds a Lagrangian parameter $\mu$ to make use of the constant gain approximation
# Algorithm
# 
# $\beta^*$ obtained by solving the set of linear equations
# \begin{equation}
# \begin{aligned}
# 0  &=  2 \Bigl(  \frac{1}{N}  \sum_{k=1}^d M_{x_k}^T M_{x_k}   +  \lambda M_0 \Bigr) \beta ^* + \frac{ \kappa \mu ^*}{N}+  \frac{2}{N} \Bigl( \kappa \text{K}^*  -   M_0 \tilde{c} \Bigr)  \\
# 0  & = \kappa^{T} \beta^*
# \end{aligned}
# \end{equation}
# =============================================================================
def gain_rkhs_om(Xi, C, epsilon, Lambda, diag = 0):
    start = timer()
    
    N,dim = Xi.shape
    K = np.zeros((N,dim))
    Ker_x = np.array(np.zeros((N,N,dim)))
    # Ker_xy = np.array(np.zeros((N,N)))
    
    Ker = np.exp(- squareform(pdist(Xi,'euclidean'))**2/ (4 * epsilon))
    
    for i in range(N):
        Ker_x[i,:,:] = np.multiply(-(Xi[i,:]-Xi) ,Ker[i,:].reshape((-1,1))) / (2 * epsilon)
            # Ker_xy[i,j] = (((Xi[i] - Xi[j])**2) / (2 * epsilon) -1) * Ker[i,j] / (2 * epsilon)
    Ker_x = Ker_x.astype(np.float32)
    Ker_x_ones = np.dot(np.transpose(Ker_x), np.ones((N,1)))

    eta = np.mean(C)
    Y = (C -eta)
    
    K_hat = np.mean(Y * Xi, axis = 0)

    b_m = (2/ N) * np.dot(Ker,Y) - (2/ N) * np.dot( np.moveaxis(Ker_x_ones,0,2), K_hat) 
    b_m = np.append(b_m, np.zeros((dim,1)))
    
    # Ker_x_sum = np.zeros((N,N))
    # for d_i in range(dim):
    Ker_x_sum = np.array([np.dot(Ker_x[:,:,d_i], Ker_x[:,:,d_i].transpose()) for d_i in np.arange(dim)]).sum(axis =0)
    M_m = 2 * Lambda * Ker + (2 / N) * Ker_x_sum
    M_m = np.vstack((M_m, (1/N) * np.squeeze(Ker_x_ones)))
    M_m = np.hstack((M_m, np.append(np.squeeze(np.transpose(Ker_x_ones),axis =0),np.zeros((dim,dim)),axis =0))) #.reshape(len(M_m),1))
#    if (np.linalg.det(M_m)!=0):
    beta_m = np.linalg.solve(M_m,b_m)
#    else:
#        beta_m = np.linalg.lstsq(M_m,b_m)[0]

    # K.fill(K_hat)
    K  = np.tile(K_hat, (N,1))
    K  = K + np.dot(beta_m[:-dim].reshape((-1,1)).transpose(), Ker_x)
    K  = np.squeeze(K)  
    K  = K.reshape((K.shape[0],dim))   
    
    if diag == 1:
        plt.figure()
        plt.plot(Xi, Ker[:,0],'r*')
        plt.plot(Xi, Ker_x[:,0], 'b*')
        #plt.plot(Xi, Ker_xy[:,0],'k*')
        plt.show()
            
    end = timer()
    print('Time taken for gain_rkhs_om()' , end - start)
    
    return K

# =============================================================================
# ## Hyperparameter selection using grid search 
# ### select_hyperparameters() -  Hyper parameter selection for RKHS OM for dim $d$
# =============================================================================
def select_hyperparameters(method,Lambda = None,eps = None, No_runs = None, N = None, dim =None): 
    # Choose hyperparameters for which algorithm
    if method is None:
        method = input('Input the algorithm (coif, rkhs_N, rkhs_dN, om,) ')
    
    # Run parameters
    if No_runs is None:
        No_runs = input('Input the number of independent runs - ')  
    if No_runs =='':
        No_runs = 100
    else:
        No_runs = int(No_runs)
        
    # FPF parameters - No. of particles
    if N is None:
        N = input('Input the number of particles - ')
    if N =='':
        N = 500
    else:
        N = int(N)
        
    # System parameters
    if dim is None:
        dim = input('Input the dimension of the system - ') # dimension of the system
    if dim =='':
        dim = 1
    else:
        dim = int(dim)
        
    x = symbols('x0:%d'%dim)
    c_coef = np.ones((1,dim)) 
    c =  c_coef.dot(x)      # Observation function (Eg: for d = 2, c(x) = x1 + x2)
    c_x = lambdify(x, c, 'numpy')

        
    # Parameters of the prior density \rho_B - 2 component Gaussian mixture density
    gm = dim     # No. of dimensions with Gaussian mixture densities in the dim-dimensional density, should be <= dim
    p_b = 0
    for m in range(len(parameters.w_b)):
        p_b = p_b + parameters.w_b[m] * (1/ np.sqrt(2 * math.pi * parameters.sigma_b[m]**2))* exp(-(x[0] - parameters.mu_b[m])**2/ (2* parameters.sigma_b[m]**2))
    # Standard deviation for the Gaussian component (if any)    
    # Hyperparameter grid
    if eps is None:
        eps = parameters.eps
    if Lambda is None:
        Lambda = parameters.Lambda
        
    if method == 'coif':
        mse  = np.zeros((No_runs, len(eps)))
    else:
        mse  = np.zeros((No_runs, len(eps), len(Lambda)))
    mse_const= np.zeros(No_runs)  # Used as baseline to compare the performance of the method
    
    print('Setup')
    print('No. of independent runs ', No_runs)
    print('Dimensions ', dim)
    print('===============')
    for run in range(No_runs):
        clear_output()
        print('No. of particles ', N)
        print('Dimensions ', dim)
        print('Run ',run+1 ,' of ', No_runs)
        # Xi  = get_samples(N, mu_b, sigma_b, w_b, dim, gm, sigma, seed = run)
        Xi  = get_samples(N, parameters.mu_b, parameters.sigma_b, parameters.w_b, dim, gm, parameters.sigma)
        if dim == 1:
            Xi = np.sort(Xi,kind = 'mergesort')
        C = np.reshape(c_x(*Xi.T),(len(Xi),1))
    
        K_exact = np.zeros((N, dim))
        for d in range(gm):
            K_exact[:,d]  = gain_num_integrate(Xi, x[0], p_b, x, d)
        
        for i,eps_i in enumerate(eps):
            if method == 'coif':
                Phi = np.zeros(N)
                K_approx = gain_coif(Xi, C, eps_i, Phi, diag = 0) 
                mse[run, i] = mean_squared_error(K_exact, K_approx)
            elif method == 'rkhs_N':
                for j,Lambda_j in enumerate(Lambda):  
                    K_approx = gain_rkhs_N(Xi, C, eps_i, Lambda_j, diag = 0)
                    mse[run, i,j] = mean_squared_error(K_exact, K_approx)
            elif method == 'rkhs_dN':
                for j,Lambda_j in enumerate(Lambda):  
                    K_approx = gain_rkhs_dN(Xi, C, eps_i, Lambda_j, diag = 0)
                    mse[run, i,j] = mean_squared_error(K_exact, K_approx)
            elif method == 'om':
                for j,Lambda_j in enumerate(Lambda):  
                    K_approx = gain_rkhs_om(Xi, C, eps_i, Lambda_j, diag = 0)
                    mse[run, i,j] = mean_squared_error(K_exact, K_approx)
            else:
                for j,Lambda_j in enumerate(Lambda):  
                    print('Invalid method provided')
        
        # Baseline error calculation        
        eta = np.mean(C)
        Y = (C -eta)
        K_const = np.mean(Y * Xi, axis = 0)
        mse_const[run] = mean_squared_error(K_exact, K_const)     
        
    if method == 'coif':
        i_min = np.argmin(np.mean(mse,axis=0))
        print('Best value of $\epsilon', eps[i_min])
        return mse, eps, eps[i_min]
    else:
        i_min, j_min = np.unravel_index(np.argmin(np.mean(mse,axis =0)),np.mean(mse, axis =0).shape)
        print('Best value of  $\lambda$', Lambda[j_min])
        print('Best value of $\epsilon$', eps[i_min])
        return mse,Lambda, eps, Lambda[j_min], eps[i_min]

# =============================================================================
# ### contour_lambda_eps() - Function to plot contour plots of mses vs a grid of $\lambda$ and $\epsilon$ values
# =============================================================================
def contour_lambda_eps(mse_mean, Lambda, eps, contour_levels = None):
    fig = plt.figure(figsize =(10,8))
    if contour_levels:
        cont = plt.contourf(eps, np.log10(Lambda),mse_mean.transpose(), contour_levels)
    else:
        cont = plt.contourf(eps, np.log10(Lambda),mse_mean.transpose())

    # cont = plt.contourf(eps, np.log10(Lambda),mse_mean.transpose())
    fig.colorbar(cont)
    plt.xlabel('$\epsilon$', size = 24)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.ylabel('$\log_{10}(\lambda)$', size = 24)
    plt.show()

# =============================================================================
# ### plot_hist_mse() - Function to plot a histogram of mses obtained from independent trials
# =============================================================================
def plot_hist_mse(mse,Lambda,eps, No_runs = None):
    plt.figure(figsize = (10,8))    
    for i,eps_i in enumerate(eps):
            for j,Lambda_j in enumerate(Lambda):
                sns.distplot(mse[:,i,j], label = str(Lambda_j) +',' +str(eps_i))
                plt.legend()
    plt.title('Histograms of mse obtained using various algorithms for '+ str(No_runs) + ' trials')
    plt.show()
   
# =============================================================================
# # ### plot_gains() - Function to plot the various gain approximations passed
# =============================================================================
def plot_gains(Xi,p_b,x,K_exact, K_const = None, K_finite = None, K_diff_td = None, K_om = None, K_coif = None):
    dim = Xi.shape[1]
    p_b_x = lambdify(x[0],p_b, 'numpy')
    for d in np.arange(dim):
        fig, ax1 = plt.subplots(figsize = (10,8))
        ax1.plot(Xi[:,d], K_exact[:,d], 'o', label = 'Exact')
        if K_const is not None:
            ax1.axhline(y= K_const[d], color = 'k', linestyle = '--', label ='Const gain')
        if K_finite is not None:
            ax1.plot(Xi[:,d],K_finite[:,d], '^', label = 'Finite basis')
        if K_diff_td is not None:
            ax1.plot(Xi[:,d],K_diff_td[:,d], '>', label = 'Diff TD')
        if K_om is not None:
            ax1.plot(Xi[:,d],K_om[:,d], '+', label = 'RKHS OM')
        if K_coif is not None:
            ax1.plot(Xi[:,d],K_coif[:,d], 'x', label = 'Markov kernel')
        ax1.legend(loc = 2, framealpha = 0)
        ax1.set_ylabel('Gain')
        ax2 = ax1.twinx()
        ax2.plot(Xi[:,d],p_b_x(Xi[:,d]), 'k.',label = '$\\rho(x)$')
        ax2.set_ylim(0,1)
        ax2 = sns.distplot(Xi[:,d], label = 'Particles $X_i$')
        ax2.legend(loc =1, framealpha = 0)
        ax2.set_xlabel('$X$')
        ax2.set_ylabel('Density $\\rho(x)$')
        plt.title('Gain function approximation')
        plt.show()
        
# =============================================================================
# # ### compare_hist() - Function to compare the histograms of the 2 distributions
# =============================================================================        
def compare_hist(Xi,p_b,x):
    dim = Xi.shape[1]
    p_b_x = lambdify(x[0],p_b, 'numpy')
    for d in np.arange(dim):
        plt.figure(figsize=(10,8))
        ax1 = sns.distplot(Xi[:,d], label ='Particles $X_i$')
        ax1.plot(Xi[:,d],p_b_x(Xi[:,d]),'.',label ='$\\rho(x)$')
        plt.legend(framealpha = 0)
        plt.title('Histogram of particles $X_i$ vs $ \\rho(x)$')
        plt.xlabel('$X$')
        plt.ylabel('Density')
        plt.show()
        
    