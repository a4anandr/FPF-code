#!/usr/bin/env python
# coding: utf-8

# ### FPF Gain approximation code

# In[1]:


import numpy as np
from sympy import *
from scipy.spatial.distance import pdist,squareform
from scipy.stats import norm
import scipy.integrate as integrate

import math

from timeit import default_timer as timer

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ridge
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


x = Symbol('x')


# In[90]:


if __name__ == '__main__':
    
    N = 200
    d = 1
    
    # Parameters of the prior density p(0) - 2 component Gaussian mixture density
    m = 2
    sigma = [0.4, 0.4]
    mu  = [-1, 1]
    w   = [0.5, 0.5]
    w[-1] = 1 - sum(w[:-1])
    p = 0
    c_hat = 0
    for m in range(len(w)):
        p = p + w[m] * (1/ np.sqrt(2 * math.pi * sigma[m]**2))* exp(-(x - mu[m])**2/ (2* sigma[m]**2))
        c_hat = c_hat + w[m] * mu[m] 
    p_vec = lambdify(x, p, 'numpy')
    
    Xi  = np.zeros((N,1))
    for i in range(N):
        if np.random.uniform() <= w[0]:
            Xi[i] = mu[0]  + sigma[0] * np.random.normal()
        else:
            Xi[i]  = mu[1]  + sigma[0] * np.random.normal()
    plt.figure()
    sns.distplot(Xi)
    plt.show()
    
    c = x
    
    K_exact  = gain_exact(Xi, c, p, c_hat, diag = 0)
    
    eps_rkhs = 0.1
    Lambda = 1e-3
    alpha = 0
    K_prev = []
    K_rkhs = gain_rkhs(Xi, c, eps_rkhs, Lambda, alpha, K_prev, diag = 0)
    
    eps_coif = 0.1
    Phi = np.zeros(N)
    K_coif = gain_coif(Xi, c, eps_coif, Phi, diag = 0)
    
    ### Displaying the plots
    plt.figure()
    fig,ax1 = plt.subplots()
    ax1.plot(Xi, K_exact, 'b^', markersize = 2, label ='Exact gain')
    ax1.plot(Xi, K_rkhs, 'r*', markersize =2, label = 'RKHS approx.')
    ax1.plot(Xi, K_coif, 'g.', markersize = 2, label ='Markov kernel approx.')
    ax2 =ax1.twinx()
    ax2.plot(np.arange(-2,2,0.01), p_vec(np.arange(-2,2,0.01)),'k--', markersize =2, label = 'p(x)')
    ax2.set_ylabel('p(x)')
    ax2.legend(loc=1)
    ax1.set_xlabel('Particle Locations')
    ax1.set_ylabel('Gain K(x)')
    ax1.legend()
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()
    
    mse_rkhs = mean_squared_error(K_exact, K_rkhs, p)
    mse_coif = mean_squared_error(K_exact, K_coif, p)
    print(mse_rkhs)
    print(mse_coif)
    


# #### Function to approximate FPF gain using RKHS method

# In[92]:


def gain_rkhs(Xi, c, epsilon, Lambda, alpha, K_prev, diag = 0):
    start = timer()
    
    N = len(Xi)
    K = np.zeros(N)
    Ker_x = np.array(np.zeros((N,N)))
    Ker_xy = np.array(np.zeros((N,N)))
    
    Ker = np.exp(- squareform(pdist(Xi,'euclidean'))**2/ (4 * epsilon))
    
    for i in range(N):
        for j in range(N):
            Ker_x[i,j] = -(Xi[i]-Xi[j]) * Ker[i,j] / (2 * epsilon)
            Ker_xy[i,j] = (((Xi[i] - Xi[j])**2) / (2 * epsilon) -1) * Ker[i,j] / (2 * epsilon)
    
    c_vec = lambdify(x, c, 'numpy')
    H = c_vec(Xi)
    eta = np.mean(H)
    Y = (H -eta)
    
    b_m = (1/ N) * np.dot(Ker,Y)
    M_m = Lambda * Ker + ((1 + alpha)/ N) * np.matmul(Ker_x, Ker_x.transpose())
    beta_m = np.linalg.solve(M_m,b_m)
    
    K = np.zeros(N)
    for i in range(N):
        for j in range(N):
            K[i] = K[i] + beta_m[j] * Ker_x[i,j]
            
    if diag == 1:
        plt.figure()
        plt.plot(Xi, Ker[:,0],'r*')
        plt.plot(Xi, Ker_x[:,0], 'b*')
        plt.plot(Xi, Ker_xy[:,0],'k*')
        plt.show()
            
    end = timer()
    print('Time taken' , end - start)
    
    return K


# #### Function to compute the exact FPF gain

# In[70]:


def gain_exact(Xi, c, p, c_hat, diag = 0):
    start = timer()
    
    N = len(Xi)
    K = np.zeros(N)
    integral = np.zeros(N)
    
    step = 0.01
    xmax = max(mu) + 10
    
    p_vec = lambdify(x, p, 'numpy')
    c_vec = lambdify(x, c, 'numpy')
    H = c_vec(Xi)
    
    for i in range(N):
        integral[i] = 0
        for xj in np.arange(Xi[i], xmax + 10,  step):
            integral[i] = integral[i] + p_vec(xj) * ( c_vec(xj) - c_hat) * step
        K[i] = integral[i]/ p_vec(Xi[i])
            
    end = timer()
    print('Time taken' , end - start)
    return K


# #### Function to approximate FPF gain using Markov kernel approx. method

# In[75]:


def gain_coif(Xi, c, epsilon, Phi, diag = 0):
    start = timer()
    
    N = len(Xi)
    k = np.zeros((N,N))
    K = np.zeros(N)
    T = np.zeros((N,N))
    Phi = np.zeros(N)
    sum_term = np.zeros(N)
    max_diff = 1
    
    No_iterations = 50000
    iterations = 1
    
    c_vec = lambdify(x, c, 'numpy')
    H   = c_vec(Xi)
    
    Ker = np.exp(- squareform(pdist(Xi,'euclidean'))**2/ (4 * epsilon))
    for i in range(N):
        for j in range(N):
            k[i,j] = Ker[i,j] / (np.sqrt( (1/N) * sum(Ker[i,:])) * np.sqrt( (1/N)* sum(Ker[j,:])))
        T[i,:] = np.divide(k[i,:], np.sum(k[i,:]))
    
    while((max_diff > 1e-2) & ( iterations < No_iterations )):
        Phi_new = np.matmul(T,Phi) + (epsilon * np.concatenate(H)).transpose() 
        max_diff = max(Phi_new - Phi) - min(Phi_new - Phi)
        Phi  = Phi_new
        iterations += 1
                           
    for i in range(N):
        sum_term[i] = np.dot( T[i,:], Xi)
        K[i] = 0
        for j in range(N):
            K[i] = K[i] + (1/ (2 * epsilon)) * T[i,j] * Phi[j,] * (Xi[j] - sum_term[i]) 
                                               
    if diag == 1:
        plt.figure()
        plt.plot(Xi, Ker[1,:], 'r*')
        plt.show()
    
    end = timer()
    print('Time taken' , end - start)
    
    return K


# #### Function to compute the mean square error in gain function approximation

# In[87]:


def mean_squared_error(K_exact, K_approx, p):
    N = len(K_exact)
    p_vec = lambdify(x, p, 'numpy')
    mse = (1/N) * np.linalg.norm(K_exact - K_approx)**2
    # mse2 = np.sum(((K_exact - K_approx)**2) *np.concatenate(p_vec(Xi)))
    return mse


# ### Rough trials

# In[ ]:


start = timer()
for i in range(len(Xi)):
    for j in range(len(Xi)):
        Ker[i,j] = np.exp(- ((Xi[i] - Xi[j])**2).sum() / ( 4 * epsilon)) 
        # Ker_x[i,j]  = -(Xi[i,0 ] - Xi[j,0]) * Ker[i,j]
        # Ker_xy[i,j] =  
end = timer()
print(end-start)


# In[ ]:


start = timer()
Ker2 = np.exp(- squareform(pdist(Xi,'euclidean'))**2/ (4 * epsilon))
end = timer()
print(end - start)


# In[29]:


a = [[1,2],[4,3]]
b = [2,3]


# In[30]:


np.dot(a,b)


# In[31]:


np.matmul(a,b)


# In[32]:


np.asarray(a).transpose()


# In[35]:


np.transpose(a)


# In[33]:


a


# In[34]:


b


# In[42]:


c_vec = lambdify(x, c, 'numpy')
H   = c_vec(Xi)


# In[ ]:




