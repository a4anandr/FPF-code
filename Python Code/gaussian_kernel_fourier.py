# -*- coding: utf-8 -*-
### Function to plot a standard Gaussian kernel and its Fourier transform

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib','qt')

def gauss_kernel(X, eps):
    K = np.exp(-X**2 / (4 * eps))
    return K

def gauss_fourier(omega, eps):
    F_k = np.abs(np.sqrt(2*eps))*np.exp(- eps * omega**2)
    return F_k
    
if __name__ == '__main__':
    
    d = 1
    step = 0.05
    X = np.arange(-3,3,step)
    omega = np.arange(-20,20,0.05)
    eps = 0.125
    
    kern = gauss_kernel(X,eps)
    kern_fourier = gauss_fourier(omega,eps)
    
    fig,ax = plt.subplots(1,2, figsize =(10,5))
    ax[0].plot(X, kern,'k-')
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$K_\epsilon(x)$')
    ax[0].set_title('Gaussian kernel, $K_\epsilon$')
    
    ax[1].plot(omega, kern_fourier,'k-',)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('$\omega$')
    ax[1].set_ylabel('$F[K_\epsilon](\omega)$')
    ax[1].set_title('Fourier transform of $K_\epsilon$')
    fig.savefig('Figure/GaussKernelFourier.pdf')
    plt.show()