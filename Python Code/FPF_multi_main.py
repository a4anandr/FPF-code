# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:42:06 2019

@author: anand
"""

# #### Multi-dimensional example in section 5.1  - https://arxiv.org/pdf/1902.07263.pdf  
# \begin{equation}
# \rho(x) = \rho_B(x_1) \prod_{n=2}^d \rho_G(x_n), \qquad \text{for } x = (x_1,x_2, \cdots, x_d) \in \mathbb{R}^d 
# \end{equation}
# Here, $\rho_B$ is $\frac{1}{2} \mathcal{N}(-1, \sigma^2) + \frac{1}{2}\mathcal{N}(+1,\sigma^2)$ is bimodal distribution  
# $\rho_G$ is Gaussian distribution, $\mathcal{N}(0,\sigma^2)$  
# Observation function, $h(x) = x_1$  
# Exact gain function, $\text{K}_{\text{exact}}(x) = (\text{K}_{\text{exact}}(x_1), 0, \cdots,0)$  
if __name__ == '__main__':
    
    ## Flags to be set to choose which methods to compare
    exact  = 1      # Computes the exact gain and plots 
    coif   = 0      # Computes gain using Coifman kernel method
    rkhs_N = 0      # Computes gain using subspace of RKHS
    rkhs_dN= 1      # Computes optimal gain using RKHS 
    om     = 0      # Computes gain using RKHS enforcing constant gain constraint
    memory = 0      # Computes gain using RKHS with a memory parameter for previous gain
    om_mem = 0      # Computes gain using const gain approx and a memory parameter for previous gain
    
    coif_old = 0    # Computes old implementation of Coifman kernel approx. 
    const  = 0      # Computes the constant gain approximation
    kalman = 0      # Runs Kalman Filter for comparison
    sis    = 0      # Runs Sequential Importance Sampling Particle Filter 

    # Run parameters
    No_runs = 1
    
    # FPF parameters - No. of particles
    N = 200
    
    # System parameters
    dim = 2     # dimension of the system
    x = symbols('x0:%d'%dim)
    c = x[0]       # Observation function
    # c_x = lambdify(x, c, 'numpy')
    c_x = lambdify(x[0], c, 'numpy')
        
    # Parameters of the prior density \rho_B - 2 component Gaussian mixture density
    m = 2      # No of components in the Gaussian mixture
    sigma_b = [0.4472, 0.4472]   # Gives \sigma^2 = 0.2
    mu_b  = [-1, 1]
    w_b   = [0.5, 0.5]
    w_b[-1] = 1 - sum(w_b[:-1])
    p_b = 0
    for m in range(len(w_b)):
        p_b = p_b + w_b[m] * (1/ np.sqrt(2 * math.pi * sigma_b[m]**2))* exp(-(x[0] - mu_b[m])**2/ (2* sigma_b[m]**2))
    p_b_x = lambdify(x[0], p_b, 'numpy')
    sigma = 0.4472  # Chosen so that \sigma^2 = 0.2 as in the reference
    p = p_b
    for d in np.arange(1,dim):
        p_g = exp(-x[d])**2/ (2 * sigma**2)
        p*= p_g
    
    
    mse_coif = np.zeros(No_runs)
    mse_rkhs_N = np.zeros(No_runs)
    mse_rkhs_dN = np.zeros(No_runs)
    mse_om   = np.zeros(No_runs)
    mse_coif_old = np.zeros(No_runs)
    for run in range(No_runs):
        clear_output()
        print('Run ',run)
        Xi  = get_samples(N, mu_b, sigma_b, w_b, dim, sigma)
        get_samples
        if dim == 1:
            Xi = np.sort(Xi,kind = 'mergesort')
        C = np.reshape(c_x(Xi[:,0]),(len(Xi),1))
        # C   = np.reshape(c_x(Xi[:,0],Xi[:,1]),(len(Xi),1))
        
        if exact == 1:
         
            K_exact = np.zeros((N, dim))
            K_exact[:,0]  = gain_num_integrate(Xi, c, p_b)

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
    if exact == 1 & coif == 1:
        print('MSE for Markov kernel approx', np.mean(mse_coif))
    if exact == 1 & rkhs_N == 1:
        print('MSE for RKHS N', np.mean(mse_rkhs_N))
    if exact == 1 & rkhs_dN == 1:
        print('MSE for RKHS dN', np.mean(mse_rkhs_dN))
    if exact == 1 & om == 1:
        print('MSE for RKHS OM', np.mean(mse_om))
    if exact == 1 & coif_old == 1:
        print('MSE for old Markov kernel', np.mean(mse_coif_old))
    
    ### Displaying the plots
    marker_size  = 3
    plt.rc('text', usetex=True)
    fig,ax1 = plt.subplots()
    if exact == 1:
        ax1.plot(Xi, K_exact, 'bv', markersize = marker_size, label ='Exact gain')
    if rkhs_N == 1:
        ax1.plot(Xi, K_rkhs_N, 'r*', markersize = marker_size, label = 'RKHS approx. N')
    if rkhs_dN == 1:
        ax1.plot(Xi, K_rkhs_dN, 'cs', markersize = marker_size, label = 'RKHS approx. dN')
    if coif == 1:
        ax1.plot(Xi, K_coif, 'g.', markersize = marker_size, label ='Markov kernel approx.')
    if om == 1:
        ax1.plot(Xi, K_om, 'm*', markersize = marker_size, label = 'RKHS OM')
    if coif_old == 1:
        ax1.plot(Xi, K_coif_old, 'y.', markersize = marker_size, label ='Old Markov kernel')
    ax2 =ax1.twinx()
    ax2.plot(np.arange(-2,2,0.01), p_b_x(np.arange(-2,2,0.01)),'k.-', markersize =1, label = r'$\rho(x)$')
    ax2.set_ylabel(r'$\rho(x)$')
    ax2.legend(loc=1)
    ax1.set_xlabel('Particle Locations')
    ax1.set_ylabel('Gain $K(x)$')
    ax1.legend()
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()     

