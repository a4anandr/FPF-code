# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 02:46:24 2019

@author: anand
"""
import numpy as np
from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as plt
from parameters import *
get_ipython().run_line_magic('matplotlib', 'auto')

## Contour plots of Coifman kernel and Gaussian kernel
step = 0.05
X = np.arange(-3,3.05,step)
X = X.reshape(-1,1)
N = X.shape[0]

rho = np.zeros((len(X),1))
for m in range(len(w_b)):
    rho = rho + \
    w_b[m] * (1/ np.sqrt(2 * np.pi * sigma_b[m]**2))* np.exp(-(X - mu_b[m])**2/ (2* sigma_b[m]**2)) * step

# X = get_samples(500, mu_b, sigma_b, w_b, dim =1, gm=1, sigma=0)
# X.reshape(len(X)).sort()

epsilon = np.arange(0.5,1.25,0.25)

for eps in epsilon:
    d = np.zeros(N)
    k = np.zeros((N,N))
    T = np.zeros((N,N))
    g = (1/np.sqrt(4 * np.pi * eps)) * np.exp(- squareform(pdist(X,'euclidean'))**2/ (4 * eps))    
    for i in range(N):
        g[i,:] = np.divide(g[i,:], np.sum(g[i,:]))
    for i in range(N):
        for j in range(N):
            k[i,j] = g[i,j] / (np.sqrt(np.dot(g[i,:],rho)) * np.sqrt(np.dot(g[j,:],rho)))
        d[i] = np.sum(k[i,:])
        k[i,:] = np.divide(k[i,:], d[i])
    
    fig, axes = plt.subplots(1,2, figsize = (10,5))
    axes[0].contour(X.reshape(len(X)),X.reshape(len(X)),g)
    axes[0].set_title('Standard Gaussian kernel with $\epsilon =$ {}'.format(np.round(eps,2)),size = 22)
    axes[1].contour(X.reshape(len(X)),X.reshape(len(X)),k)
    axes[1].set_title('Coifman kernel with $\epsilon =$ {}'.format(np.round(eps,2)), size = 22)
    #fig.savefig('kernel_contour_compare_{}'.format(eps)+'.pdf')
    #fig.savefig('kernel_contour_compare_{}'.format(eps)+'.jpg')
    plt.show()
    
    fig2 = plt.figure(figsize=(5,5))
    plt.plot(X,g[60,:],'r--', label ='Gaussian')
    plt.plot(X,k[60,:],'b.-', label = 'Coifman')
    plt.legend(framealpha = 0, prop ={'size':15}, loc = 0)
    plt.title('Comparing Standard Gaussian and Coifman kernels with $\epsilon =$ {}'.format(np.round(eps,2)), size = 22)
    plt.show()
    #fig2.savefig('Gaussian_Coifman_for_eps{}'.format(eps)+'.pdf')
    #fig2.savefig('Gaussian_Coifman_for_eps{}'.format(eps)+'.jpg')