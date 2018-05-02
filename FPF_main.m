%% Feedback particle filter for state estimation problems 
% - Defines the state and observation processes - X_t and Z_t
% - Defines the prior distribution p(0) 
% - Generates particles Xi from the prior p(0)
% - Passes these particles Xi to four different gain function approximation
% functions - i) diff TD learning (old), ii) finite dim, iii) Coifman kernel, iv) RKHS
% based.

clear;
clc;
close all;
tic

syms x;
diag_main = 1;   % Diagnostics flag for main function, displays figures in main.
diag_fn = 0;     % Diagnostics flag, if 1, then all the functions display plots for diagnostics, Set it to 0 to avoid plots from within the calling functions
% rng(100);     % Set a common seed

%% Flags to be set to choose which methods to compare

exact = 1;           % Computes the exact gain and plots 
fin   = 0;           % Computes gain using finite dimensional basis
coif  = 1;           % Computes gain using Coifman kernel method
rkhs  = 1;           % Computes gain using RKHS

%% FPF parameters

   N = 500;          % No of particles
   
% i) Finite dimensional basis
if fin == 1
   d = 20;           % No of basis functions
   basis = 1;        % 0 for polynomial, 1 for Fourier
   p  =    0;        % 1 for weighting with density, 0 otherwise 
end

% ii) Coifman kernel 
if coif == 1
   eps_coif = 0.1;   % Time step parameter
end

% iii) RKHS
if rkhs == 1
   kernel   = 0;           % 0 for Gaussian kernel, 1 for Coifman kernel, 2 for approximate Coifman kernel using EM
   lambda   = 0.05;        % Regularization parameter - Other tried values ( 0.005,0.001,0.05), For kernel = 0, range 0.005 - 0.01.
   eps_rkhs = 0.1;         % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
end

%% Parameters corresponding to the state and observation processes
% Run time parameters
T   = 0.8;       % Total running time - Using same values as in Amir's CDC paper - 0.8
dt  = 0.01;      % Time increments for the SDE

% State process parameters
a = 0;         % 0 for a steady state process
sigmaB = 0;      % No noise in state process

% Observation process parameters
c = x;
sigmaW = 0.3;

% Parameters of p(0) - 2 component Gaussian mixture density 
m = 2;
sigma = [0.4 0.4]; 
mu    = [-1 1]; 
w     = [0.5 rand]; % Needs to add up to 1.
w(m)  = 1 - sum(w(1:m-1));

%% Additional variables 
sdt = sqrt(dt);

c_x = matlabFunction(c);

%% State and observation process
% Initialization
X(1)   = mu(2);
Z(1)   = c_x(X(1)) * dt + sigmaW * sdt * randn;

% Initializing N particles for FPF from p(0)
gmobj = gmdistribution(mu',reshape(sigma.^2,1,1,m),w);
Xi_0  = random(gmobj,N);
Xi_0  = Xi_0';            % To be consistent with code below
Xi_0  = sort(Xi_0);       % Sort the samples in ascending order for better visualization.
mui_0   = mean(Xi_0);

Xi_exact(1,:)= Xi_0;      % Initializing the particles for all 3 approaches with the same set
Xi_fin(1,:)  = Xi_0;      
Xi_coif(1,:) = Xi_0;
Xi_rkhs(1,:) = Xi_0;   

mu_em = [0 mu];
sigma_em = [0 sigma];
w_em = [0 w];

mu_rkhs = [0 mu];
sigma_rkhs = [0 sigma];
w_rkhs = [0 w];

%% State and observation process evolution

for k = 2: 1: (T/dt)
    k
    X(k) = X(k-1) +   a * X(k-1) * dt + sigmaB * sdt * randn;
    Z(k) = Z(k-1) +   c_x(X(k))  * dt + sigmaW * sdt * randn; 
    
    if k == 2 
        dZ(k) = Z(k) - Z(k-1); 
    else
        dZ(k) = 0.5 * (Z(k) - Z(k-2));
    end
    
    if exact == 1
        [mu_em, sigma_em, w_em ] = em_gmm ( Xi_exact(k-1,:), mu_em, sigma_em, w_em, diag_fn);
        [K_exact(k,:)] = gain_exact(Xi_exact(k-1,:), c_x, mu_em, sigma_em, w_em, diag_fn );
    end
    if fin == 1
        [K_fin(k,:)  ] = gain_fin(Xi_fin(k-1,:), c_x, d , basis, mu, sigma, p, diag_fn);
    end
    if coif == 1
        [K_coif(k,:) ] = gain_coif(Xi_coif(k-1,:) , c_x, eps_coif, diag_fn);
    end 
    if rkhs == 1
        [K_rkhs(k,:) ] = gain_rkhs(Xi_rkhs(k-1,:) , c_x, kernel,lambda, eps_rkhs,diag_fn);
    end
        
    for i = 1:N
       % i) Using exact solution of gain
       if exact == 1
           mu_exact(k-1)    = mean(Xi_exact(k-1,:));
           c_hat_exact(k-1) = mean(c_x(Xi_exact(k-1,:)));
           dI_exact(k)      = dZ(k) - 0.5 * (c_x(Xi_exact(k-1,i)) + c_hat_exact(k-1)) * dt;
           Xi_exact(k,i)    = Xi_exact(k-1,i) + a * Xi_exact(k-1,i) * dt + sigmaB * sdt * randn + (1/ sigmaW^2) * K_exact(k,i) * dI_exact(k);
       end
       
       % ii) Finite dimensional basis 
       if fin == 1
           mu_fin(k-1)      = mean(Xi_fin(k-1,:));
           c_hat_fin(k-1)   = mean(c_x(Xi_fin(k-1,:)));
           dI_fin(k)        = dZ(k) - 0.5 * (c_x(Xi_fin(k-1,i)) + c_hat_fin(k-1)) * dt;
           Xi_fin(k,i)      = Xi_fin(k-1,i) + a * Xi_fin(k-1,i) * dt + sigmaB * sdt * randn + (1/ sigmaW^2) * K_fin(k,i) * dI_fin(k);
       end
       
       % iii) Coifman kernel
       if coif == 1
           mu_coif(k-1)     = mean(Xi_coif(k-1,:));
           c_hat_coif(k-1)  = mean(c_x(Xi_coif(k-1,:)));
           dI_coif(k)       = dZ(k) - 0.5 * (c_x(Xi_coif(k-1,i)) + c_hat_coif(k-1)) * dt;
           Xi_coif(k,i)     = Xi_coif(k-1,i) + a * Xi_coif(k-1,i) * dt + sigmaB * sdt * randn + (1 / sigmaW^2) * K_coif(k,i) * dI_coif(k);
       end
       
       % iv) RKHS
       if rkhs == 1
           mu_rkhs(k-1)     = mean(Xi_rkhs(k-1,:));
           c_hat_rkhs(k-1)  = mean(c_x(Xi_rkhs(k-1,:)));
           dI_rkhs(k)       = dZ(k) - 0.5 * (c_x(Xi_rkhs(k-1,i)) + c_hat_rkhs(k-1)) * dt;
           Xi_rkhs(k,i)     = Xi_rkhs(k-1,i) + a * Xi_rkhs(k-1,i) * dt + sigmaB * sdt * randn + (1 / sigmaW^2) * K_rkhs(k,i) * dI_rkhs(k);
       end
       
    end

%% Displaying figures for diagnostics 

    % Plotting gains at k = 2, 3, 10, 20, 30, 40
    if ( diag_main == 1 && ( k == 2 | k == 3 | k == 11 || k == 21 || k == 31 || k == (T/dt))) 
        figure;
        if exact == 1
            plot(Xi_exact(k-1,:), K_exact(k,:), 'gv','DisplayName','Exact'); 
            hold on;
        end
        if fin == 1
            plot(Xi_fin(k-1,:), K_fin(k,:), 'r*', 'DisplayName','Finite');  
            hold on;
        end
        if coif == 1
            plot(Xi_coif(k-1,:), K_coif(k,:), 'bo','DisplayName','Coifman');
            hold on;
        end
        if rkhs == 1
            plot(Xi_rkhs(k-1,:), K_rkhs(k,:), 'k^','DisplayName','RKHS');
            hold on;
        end
        title(['Gain at particle locations for ' num2str(N) ' particles at t = ' num2str((k-2) * dt)]);
        legend('show');
    end
    
    if ( diag_main == 1 && (k == 2 || k == (T/dt)) && exact == 1)
       p_t = 0;
       step = 0.05;
       range = min(mu_em)- 3 * max(sigma_em): step : max(mu_em) + 3 * max(sigma_em);
       for i = 1: length(mu_em)
           p_t = p_t + w_em(i) * exp(-( range - mu_em(i)).^2 / (2 * sigma_em(i)^2)) * (1 / sqrt(2 * pi * sigma_em(i)^2));
       end
       
       figure(100);
       plot(range, p_t,'DisplayName',['Smoothed density using exact at t = ' num2str( (k-1)*dt )]);
       hold on;
       v = version('-release');
       if (v == '2014a')
            if fin == 1
               hist(Xi_fin(k-1,:),N);
            end
            if coif == 1
               hist(Xi_coif(k-1,:),N);
            end
            if rkhs == 1
               hist(Xi_rkhs(k-1,:),N);
            end                      
       else
            if fin == 1
             % CAUTION : histogram command works only in recent Matlab
             % versions, if it does not work, comment this section out
               histogram(Xi_fin(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName',['Histogram using finite at t =' num2str( (k-1)*dt )]);
            end
            if coif == 1
               histogram(Xi_coif(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName',['Histogram using Coifman at t =' num2str( (k-1)*dt )]);
            end
            if rkhs == 1
               histogram(Xi_rkhs(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName',['Histogram using RKHS at t =' num2str( (k-1)*dt )]);
            end
       end  
       legend('show');
       title('Posterior density p_t - Smoothed and Histogram');
    end
end

if exact ==1
    mu_exact(k)     = mean(Xi_exact(k,:));
end
if fin == 1
    mu_fin(k)       = mean(Xi_fin(k,:));
end
if coif == 1
    mu_coif(k)      = mean(Xi_coif(k,:));
end
if rkhs == 1
    mu_rkhs(k)      = mean(Xi_rkhs(k,:));
end

%% Plots
figure;
plot(0:dt:(k-1)*dt, X(1:k),'k','DisplayName','Actual state');
hold on;
if exact == 1
    plot(0:dt:(k-1)*dt, mu_exact(1:k),'g--','DisplayName','Exact');
    hold on;
end
if fin == 1
    plot(0:dt:(k-1)*dt, mu_fin(1:k),'r--','DisplayName','Finite');
    hold on;
end
if coif == 1
    plot(0:dt:(k-1)*dt, mu_coif(1:k),'b--','DisplayName','Coifman');
    hold on;
end
if rkhs == 1
    plot(0:dt:(k-1)*dt, mu_rkhs(1:k),'k--','DisplayName','RKHS');
    hold on;
end
legend('show');


figure;
plot(0:dt:(k-1)*dt, Z(1:k),'r');
title('Z_t');

toc





