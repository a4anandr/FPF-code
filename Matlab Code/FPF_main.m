%% Feedback particle filter for state estimation problems 
% - Defines the state and observation processes - X_t and Z_t
% - Defines the prior distribution p(0) 
% - Generates particles Xi from the prior p(0)
% - Passes these particles Xi to four different gain function approximation
% functions - i) diff TD learning (old), ii) finite dim, iii) Coifman
% kernel, iv) RKHS v) Kalman filter
% based.

clear;
clc;
close all;
tic;
warning off;

syms x;
diag_main = 1;   % Diagnostics flag for main function, displays figures in main.
diag_output = 1;
diag_fn = 0;     % Diagnostics flag, if 1, then all the functions display plots for diagnostics, Set it to 0 to avoid plots from within the calling functions
% rng(400);      % Set a common seed
No_runs = 1;     % Total number of runs to compute the rmse metric for each of the filters for comparison

%% Flags to be set to choose which methods to compare

exact  = 0;      % Computes the exact gain and plots 
fin    = 1;      % Computes gain using finite dimensional basis
coif   = 0;      % Computes gain using Coifman kernel method
rkhs   = 1;      % Computes gain using RKHS
zero_mean = 1;   % Computes gain using RKHS enforcing constant gain constraint
memory = 0;      % Computes gain using RKHS with a memory parameter for previous gain
zm_mem = 1;      % Computes gain using const gain approx and a memory parameter for previous gain
const  = 1;      % Computes the constant gain approximation
kalman = 1;      % Runs Kalman Filter for comparison
sis    = 0;      % Runs Sequential Importance Sampling Particle Filter 

%% FPF parameters

   N = 500;      % No of particles - Common for all Monte Carlo methods used
   
% i) Finite dimensional basis
if fin == 1
   d = 20;       % No of basis functions
   basis = 1;    % 0 for polynomial, 1 for Fourier
   p  =    0;    % 1 for weighting with density, 0 otherwise 
end

% ii) Coifman kernel 
if coif == 1
   eps_coif = 0.1;   % Time step parameter
   Phi = zeros(N,1); % Initializing the solution to Poisson's equation with zeros
   exact_coif = 0;   % Setting this flag to 1 computes a smooth density via EM and then computes the exact gain using the integral formula. 
end

% iii) RKHS
if rkhs == 1
   kernel   = 0;           % 0 for Gaussian kernel, 1 for Coifman kernel, 2 for approximate Coifman kernel
   lambda   = 1e-2;        %1e-2;        % 0.05, 0.02, Regularization parameter - Other tried values ( 0.005,0.001,0.05), For kernel = 0, range 0.005 - 0.01.
   eps_rkhs = 0.1;         % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
   K_rkhs   = zeros(1,N);  % Initializing the gain to a 1 vector, this value is used only at k = 1. 
   exact_rkhs = 0;         % Setting this flag to 1 computes a smooth density via EM and then computes the exact gain using the integral formula. 
end

% iv) RKHS zero mean
if zero_mean == 1
   kernel    = 0;                % 0 for Gaussian kernel, 1 for Coifman kernel, 2 for approximate Coifman kernel
   lambda_zm = 1e-2;             % 0.05, 0.02, Regularization parameter - Other tried values ( 0.005,0.001,0.05), For kernel = 0, range 0.005 - 0.01.
   eps_zm    = 0.1;              % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
   K_zm      = zeros(1,N);       % Initializing the gain to a 1 vector, this value is used only at k = 1. 
   exact_zm  = 0;                % Setting this flag to 1 computes a smooth density via EM and then computes the exact gain using the integral formula. 
end

% v) RKHS with memory
if memory == 1
   kernel          = 0;          % 0 for Gaussian kernel, 1 for Coifman kernel, 2 for approximate Coifman kernel
   lambda_mem      = 1e-2;       % 0.05, 0.02, Regularization parameter - Other tried values ( 0.005,0.001,0.05), For kernel = 0, range 0.005 - 0.01.
   eps_mem         = 0.1;        % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
   lambda_gain_mem = 0;          % 1e-4; % This parameter decides how much the gain can change in successive time instants, higher value implying less variation. 
   K_mem           = zeros(1,N); % Initializing the gain to a 1 vector, this value is used only at k = 1. 
   exact_mem       = 0;          % Setting this flag to 1 computes a smooth density via EM and then computes the exact gain using the integral formula. 
end

% vi) RKHS ZM with memory
if zm_mem == 1
   kernel          = 0;          % 0 for Gaussian kernel, 1 for Coifman kernel, 2 for approximate Coifman kernel
   lambda_zm_mem   = 1e-2;       % 0.05, 0.02, Regularization parameter - Other tried values ( 0.005,0.001,0.05), For kernel = 0, range 0.005 - 0.01.
   eps_zm_mem      = 0.1;        % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
   lambda_gain_zm_mem = 1;       % 1e-1; % This parameter decides how much the gain can change in successive time instants, higher value implying less variation. 
   K_zm_mem        = zeros(1,N); % Initializing the gain to a 1 vector, this value is used only at k = 1. 
   beta_zm_mem     = zeros(N,1); % Initializing the parameter values beta to zero. 
   exact_zm_mem    = 0;          % Setting this flag to 1 computes a smooth density via EM and then computes the exact gain using the integral formula. 
end

% vi) SIS PF
if sis ==1
   resampling = 1;         % For periodic deterministic resampling of particles
end

% Setting a max and min threshold for gain
K_max = 100;
K_min = -100;

%% Parameters corresponding to the state and observation processes
% Run time parameters
T   = 1;         % Total running time - Using same values as in Amir's CDC paper - 0.8
dt  = 0.01;      % Time increments for the SDE

% State process parameters
% a = - 2 * x;           % 0 for a steady state process
a = 0; 
if a == 0
    a_x      = @(x) 0;
    a_der_x  = @(x) 0;
    a_legend = num2str(a);
else
    a_x = @(x) eval(a);
    a_der_x = eval(['@(x)' char(diff(a_x(x)))]);   %  or matlabFunction(diff(a_x(x)));   
    a_legend = char(a);
end
sigmaB = 0;             % 0 if no noise in state process  -  Comments in Arulampalam et al. 
% If the process noise is zero, then using a particle filter is not entirely appropriate. Particle filtering is a method well suited to the estimation of dynamic states. If static states, which can be regarded as parameters, need to be estimated then alternative approaches are necessary 

% Observation process parameters
c =  x;
c_x = matlabFunction(c);
c_for_der_x = @(x) eval(c);
c_der_x = eval (['@(x)' char(diff(c_for_der_x(x)))]);
sigmaW = 0.3;

%% Parameters of the prior p(0) - 2 component Gaussian mixture density 
m = 2;
sigma = [0.4472 0.4472]; 
mu    = [-1 1]; 
w     = [0.5 rand];  % Needs to add up to 1.
w(m)  = 1 - sum(w(1:m-1));
% Constructing a 3 component Gaussian mixture for EM to make sure gain does not blow up.
mu_em = [0 mu];
sigma_em = [0 sigma];
w_em = [0 w];
% Initializting the parameters of Gaussian mixture for testing
mu_coif_em = mu_em; mu_rkhs_em = mu_em; mu_zm_em = mu_em;   mu_mem_em = mu_em; mu_zm_mem_em = mu_em;
sigma_coif = sigma_em; sigma_rkhs = sigma_em; sigma_zm = sigma_em; sigma_mem = sigma_em; sigma_zm_mem = sigma_em;
w_coif = w_em; w_rkhs = w_em; w_zm = w_em; w_mem = w_em; w_zm_mem = w_em; 

%% Additional variables 
sdt = sqrt(dt);
Q    = sigmaB^2;     % State process noise variance
R    = sigmaW^2;     % Observation process noise variance 

for run = 1: 1 : No_runs
    run
%% Initializing N particles from the prior
gmobj = gmdistribution(mu',reshape(sigma.^2,1,1,m),w);
Xi_0  = random(gmobj,N);
Xi_0  = Xi_0';            % To be consistent with code below
Xi_0  = sort(Xi_0);       % Sort the samples in ascending order for better visualization.
mui_0   = mean(Xi_0);

Xi_exact(1,:)= Xi_0;      % Initializing the particles for all 4 approaches with the same set
Xi_fin(1,:)  = Xi_0;      
Xi_coif(1,:) = Xi_0;
Xi_rkhs(1,:) = Xi_0;
Xi_zm(1,:)   = Xi_0;
Xi_mem(1,:)  = Xi_0;
Xi_zm_mem(1,:) = Xi_0;
Xi_const(1,:)= Xi_0;


%  Sequential Importance Sampling Particle Filter Initialization
Xi_sis(1,:)  = Xi_0;
Wi_sis(1,:)  = (1/N) * ones(1,N);          % Initializing all weights to equal value.
Zi_sis(1,:)  = c_x(Xi_sis(1,:)) * dt;

% Kalman filter - Initialization
if kalman == 1
    mu_0    = 0;
    P_0     = 0; 
    for i = 1:length(mu)
        mu_0 = mu_0 + w(i) * mu(i);
        P_0  = P_0  + w(i) * ( mu(i)^2 + sigma(i)^2);
    end
    X_kal        = mu_0;         % Initializing the Kalman filter state estimate to the mean at t = 0.
    P(1)         = P_0 - mu_0^2; % Initializing the state covariance for the Kalman filter 
    K_kal(1)     = (P(1) * c_der_x(X_kal))/R; 
end


%% State and observation process evolution
X(1)   = mu(2);
Z(1)   = c_x(X(1)) * dt + sigmaW * sdt * randn;

for k = 2: 1: (T/dt)
    k    
    % X(k) = X(k-1) +   a_x(X(k-1)) * dt + sigmaB * sdt * randn;
    X(k) = X(k-1);
    Z(k) = Z(k-1) +   c_x(X(k))  * dt + sigmaW * sdt * randn; 
    
    if k == 2 
        dZ(k) = Z(k) - Z(k-1); 
    else
        dZ(k) = 0.5 * (Z(k) - Z(k-2));
    end
    
    if exact == 1
        % [mu_em, sigma_em, w_em ] = em_gmm ( Xi_exact(k-1,:), mu_em, sigma_em, w_em, diag_fn);  % To obtain the exact solution, a smoothing density that would have generated these particles needs to be computed via Expectation Maximization
        % [K_exact(k,:)]   = gain_exact(Xi_exact(k-1,:), c_x, mu_em, sigma_em, w_em, diag_fn );    % Once the smooth density parameters are computed, they can be passed to the gain computation function
        [K_exact(k,:)]   = gain_exact(Xi_exact(k-1,:), c_x, mu, sigma, w, diag_fn ); 
        mu_exact(k-1)    = mean(Xi_exact(k-1,:));         % Suspect these two lines need to be outside the for loop with N
        c_hat_exact(k-1) = mean(c_x(Xi_exact(k-1,:)));
    end
    
    if fin == 1
        [K_fin(k,:)  ]   = gain_fin(Xi_fin(k-1,:), c_x, d , basis, mu, sigma, p, diag_fn);
        mu_fin(k-1)      = mean(Xi_fin(k-1,:));
        c_hat_fin(k-1)   = mean(c_x(Xi_fin(k-1,:)));
    end
    
    if coif == 1
        [Phi K_coif(k,:)] = gain_coif(Xi_coif(k-1,:) , c_x, eps_coif, Phi, N, diag_fn);
        mu_coif(k-1)      = mean(Xi_coif(k-1,:));
        c_hat_coif(k-1)   = mean(c_x(Xi_coif(k-1,:)));
        
        if exact_coif == 1
            [mu_coif_em, sigma_coif, w_coif ] = em_gmm ( Xi_coif(k-1,:), mu_coif_em, sigma_coif, w_coif, diag_fn);   % To obtain the exact solution, a smoothing density that would have generated these particles needs to be computed via Expectation Maximization
            [K_exact_coif(k,:)]    = gain_exact(Xi_coif(k-1,:), c_x, mu_coif_em, sigma_coif, w_coif, diag_fn );  % Once the smooth density parameters are computed, they can be passed to the gain computation function
        end
    end 
    
    if rkhs == 1
        [beta K_rkhs(k,:)] = gain_rkhs(Xi_rkhs(k-1,:) , c_x, kernel,lambda, eps_rkhs, 0, K_rkhs(k-1,:) , diag_fn);
        % [~,K_rkhs(k,:)] = gain_rkhs_multi(Xi_rkhs(k-1,:)' , c_x, 1, kernel,lambda, eps_rkhs, diag_fn);
        mu_rkhs(k-1)     = mean(Xi_rkhs(k-1,:));
        c_hat_rkhs(k-1)  = mean(c_x(Xi_rkhs(k-1,:)));
        
        % To compare the gain approximation at this instant
        if exact_rkhs == 1
            [mu_rkhs_em, sigma_rkhs, w_rkhs ] = em_gmm ( Xi_rkhs(k-1,:), mu_rkhs_em, sigma_rkhs, w_rkhs, diag_fn);   % To obtain the exact solution, a smoothing density that would have generated these particles needs to be computed via Expectation Maximization
            [K_exact_rkhs(k,:)]    = gain_exact(Xi_rkhs(k-1,:), c_x, mu_rkhs_em, sigma_rkhs, w_rkhs, diag_fn );  % Once the smooth density parameters are computed, they can be passed to the gain computation function
        end
    end
    
    if zero_mean == 1
        [~, K_zm(k,:) ] = gain_rkhs_zero_mean(Xi_zm(k-1,:)', c_x, 1, kernel,lambda_zm, eps_zm, diag_fn);
        mu_zm(k-1)      = mean(Xi_zm(k-1,:));
        c_hat_zm(k-1)   = mean(c_x(Xi_zm(k-1,:)));
        
        % To compare the gain approximation at this instant
        if exact_zm == 1
            [mu_zm_em, sigma_zm, w_zm ] = em_gmm ( Xi_zm(k-1,:), mu_zm_em, sigma_zm, w_zm, diag_fn);   % To obtain the exact solution, a smoothing density that would have generated these particles needs to be computed via Expectation Maximization
            [K_exact_zm(k,:)]    = gain_exact(Xi_zm(k-1,:), c_x, mu_zm_em, sigma_zm, w_zm, diag_fn );  % Once the smooth density parameters are computed, they can be passed to the gain computation function
            % mean(K_zm(k,:))
            % mean(K_exact_zm(k,:))
        end
    end
    
    if memory == 1
        if k == 2
            alpha = 0;
        else
            alpha = lambda_gain_mem;  % Decides how much memory is required in updating the gain, higher value => slow variation.
        end
        [~,K_mem(k,:)]   = gain_rkhs_memory( Xi_mem(k-1,:)', c_x, 1, kernel,lambda_mem, eps_mem, alpha, K_mem((k-1),:), diag_fn);
        mu_mem(k-1)      = mean(Xi_mem(k-1,:));
        c_hat_mem(k-1)   = mean(c_x(Xi_mem(k-1,:)));
        
        % To compare the gain approximation at this instant
        if exact_mem == 1
            [mu_mem_em, sigma_mem, w_mem ] = em_gmm ( Xi_mem(k-1,:), mu_mem_em, sigma_mem, w_mem, diag_fn);   % To obtain the exact solution, a smoothing density that would have generated these particles needs to be computed via Expectation Maximization
            [K_exact_mem(k,:)]    = gain_exact(Xi_mem(k-1,:), c_x, mu_mem_em, sigma_mem, w_mem, diag_fn );  % Once the smooth density parameters are computed, they can be passed to the gain computation function
        end
    end
    
    if zm_mem == 1
        if k == 2
            [~, beta_zm_mem,  K_zm_mem(k,:)]   = gain_rkhs_zm_mem( Xi_zm_mem(k-1,:)', c_x, 1, kernel,lambda_zm_mem, eps_zm_mem, 0, K_zm_mem((k-1),:), beta_zm_mem, zeros(N,1),  diag_fn);
        else
            [~, beta_zm_mem,  K_zm_mem(k,:)]   = gain_rkhs_zm_mem( Xi_zm_mem(k-1,:)', c_x, 1, kernel,lambda_zm_mem, eps_zm_mem, lambda_gain_zm_mem, K_zm_mem((k-1),:), beta_zm_mem, Xi_zm_mem(k-2,:)', diag_fn);
        end        
        mu_zm_mem(k-1)      = mean(Xi_zm_mem(k-1,:));
        c_hat_zm_mem(k-1)   = mean(c_x(Xi_zm_mem(k-1,:)));
        
        % To compare the gain approximation at this instant
        if exact_zm_mem == 1
            [mu_zm_mem_em, sigma_zm_mem, w_zm_mem ] = em_gmm (Xi_zm_mem(k-1,:), mu_zm_mem_em, sigma_zm_mem, w_zm_mem, diag_fn);   % To obtain the exact solution, a smoothing density that would have generated these particles needs to be computed via Expectation Maximization
            [K_exact_zm_mem(k,:)]    = gain_exact(Xi_zm_mem(k-1,:), c_x, mu_zm_mem_em, sigma_zm_mem, w_zm_mem, diag_fn );  % Once the smooth density parameters are computed, they can be passed to the gain computation function
            % mean(K_zm_mem(k,:))
            % mean(K_exact_zm_mem(k,:))
        end
    end
    
    if const == 1
        mu_const(k-1)     = mean(Xi_const(k-1,:));
        c_hat_const(k-1)  = mean(c_x(Xi_const(k-1,:))); 
        K_const(k)        = mean((c_x(Xi_const(k-1,:)) - c_hat_const(k-1)) .* Xi_const(k-1,:));
    end
    
    if sis == 1
        mu_sis(k-1)       = Wi_sis(k-1,:)* Xi_sis(k-1,:)';
        N_eff_sis(k-1)    =  1 / sum(Wi_sis(k-1,:).^2);
    end
        
    for i = 1:N
       % 0) Using exact solution of gain
       common_rand = randn;
       if exact == 1
           dI_exact(k)      = dZ(k) - 0.5 * (c_x(Xi_exact(k-1,i)) + c_hat_exact(k-1)) * dt;
           Xi_exact(k,i)    = Xi_exact(k-1,i) + a_x(Xi_exact(k-1,i)) * dt + sigmaB * sdt * common_rand + (1/ R) * K_exact(k,i) * dI_exact(k);
       end
       
       % i) Finite dimensional basis 
       if fin == 1
           dI_fin(k)        = dZ(k) - 0.5 * (c_x(Xi_fin(k-1,i)) + c_hat_fin(k-1)) * dt;
           Xi_fin(k,i)      = Xi_fin(k-1,i) + a_x(Xi_fin(k-1,i)) * dt + sigmaB * sdt * common_rand + (1/ R) * K_fin(k,i) * dI_fin(k);
       end
       
       % ii) Coifman kernel
       if coif == 1
           dI_coif(k)       = dZ(k) - 0.5 * (c_x(Xi_coif(k-1,i)) + c_hat_coif(k-1)) * dt;
           K_coif(k,i)      = min(max(K_coif(k,i),K_min),K_max);
           Xi_coif(k,i)     = Xi_coif(k-1,i) + a_x(Xi_coif(k-1,i)) * dt + sigmaB * sdt * common_rand + (1 / R) * K_coif(k,i) * dI_coif(k);
       end
       
       % iii) RKHS
       if rkhs == 1
           dI_rkhs(k)       = dZ(k) - 0.5 * (c_x(Xi_rkhs(k-1,i)) + c_hat_rkhs(k-1)) * dt;
           K_rkhs(k,i)      = min(max(K_rkhs(k,i),K_min),K_max);
           Xi_rkhs(k,i)     = Xi_rkhs(k-1,i) + a_x(Xi_rkhs(k-1,i)) * dt + sigmaB * sdt * common_rand + (1 / R) * K_rkhs(k,i) * dI_rkhs(k);
       end
       
       % iv) RKHS zero mean
       if zero_mean == 1
           dI_zm(k)         = dZ(k) - 0.5 * (c_x(Xi_zm(k-1,i)) + c_hat_zm(k-1)) * dt;
           K_zm(k,i)        = min(max(K_zm(k,i),K_min),K_max);
           Xi_zm(k,i)       = Xi_zm(k-1,i) + a_x(Xi_zm(k-1,i)) * dt + sigmaB * sdt * common_rand + (1 / R) * K_zm(k,i) * dI_zm(k);    
       end
       
       % v) Memory
       if memory == 1
           dI_mem(k)        = dZ(k) - 0.5 * (c_x(Xi_mem(k-1,i)) + c_hat_mem(k-1)) * dt;
           K_mem(k,i)       = min(max(K_mem(k,i),K_min),K_max);
           Xi_mem(k,i)      = Xi_mem(k-1,i) + a_x(Xi_mem(k-1,i)) * dt + sigmaB * sdt * common_rand + (1 / R) * K_mem(k,i) * dI_mem(k);
       end
       
       % vi) ZM + memory
       if zm_mem == 1
           dI_zm_mem(k)        = dZ(k) - 0.5 * (c_x(Xi_zm_mem(k-1,i)) + c_hat_zm_mem(k-1)) * dt;
           K_zm_mem(k,i)       = min(max(K_zm_mem(k,i),K_min),K_max);
           Xi_zm_mem(k,i)      = Xi_zm_mem(k-1,i) + a_x(Xi_zm_mem(k-1,i)) * dt + sigmaB * sdt * common_rand + (1 / R) * K_zm_mem(k,i) * dI_zm_mem(k);
       end
       
       % vii) Constant gain approximation 
       if const == 1
           dI_const(k)       = dZ(k) - 0.5 * (c_x(Xi_const(k-1,i)) + c_hat_const(k-1)) * dt;          
           Xi_const(k,i)     = Xi_const(k-1,i) + a_x(Xi_const(k-1,i)) * dt + sigmaB * sdt * common_rand + (1 / R) * K_const(k) * dI_const(k);
       end
       
       % viii) Sequential Importance Sampling Particle Filter (SIS PF)
       if sis == 1
          Xi_sis(k,i)       = Xi_sis(k-1,i) + a_x(Xi_sis(k-1,i)) * dt + sigmaB * sdt * common_rand; 
          Zi_sis(k,i)       = Zi_sis(k-1,i) + c_x(Xi_sis(k,i))   * dt; 
          Wi_sis(k,i)       = Wi_sis(k-1,i) * (1/sqrt( 2 * pi * R * dt)) * exp ( - (Z(k) - Zi_sis(k,i))^2/ (2 * R * dt));   %  Based on eqn(63) of Arulampalam et al. In our example, the importance density is the prior density p(X_t | X_{t-1}). If resampling is done at every step then the recursive form disappears. Wi_sis(k) does not depend on Wi_sis(k-1) as Wi_sis(k-1) = 1/N.
       end
              
    end
    
 if sis == 1
 % Normalizing the weights of the SIS - PF
     Wi_sis(k,:)  = Wi_sis(k,:)/ sum(Wi_sis(k,:));
     if resampling == 1
 % Deterministic resampling - as given in Budhiraja et al.
        if mod(k,3)== 0
            sum_N_eff = 0;
            Wi_cdf    = zeros(N,1);
            for i = 1 : N
                N_eff(i) = floor(Wi_sis(k,i) *  N); 
                Wi_res(i)= Wi_sis(k,i) - N_eff(i)/ N;
                if i == 1
                    Wi_cdf(i)= Wi_res(i);
                else
                    Wi_cdf(i)= Wi_cdf(i-1) + Wi_res(i);
                end
                if N_eff(i) > 0
                    Xi_sis_new (sum_N_eff + 1 : sum_N_eff + N_eff(i),:) = repmat(Xi_sis(k,i),N_eff(i),1);
                end
                sum_N_eff = sum_N_eff + N_eff(i);
            end  
            N_res = N - sum_N_eff;
            Wi_cdf = Wi_cdf / sum(Wi_res);
            Wi_res = Wi_res / sum(Wi_res);  
            for j = 1 : N_res
                r = rand;
                for i = 1 : N
                    if (r < Wi_cdf(i))
                        Xi_sis_new (sum_N_eff + j,:) = Xi_sis(k,i);
                    end
                end
            end
            Xi_sis(k,:)  = Xi_sis_new;
            Wi_sis(k,:)  = (1/N) * ones(1,N);
        end           
     end
     N_eff_sis(k) = 1 / (sum(Wi_sis(k,:).^2)); 
 end
    
 % vii) Extended Kalman Filter for comparison
 if kalman == 1
      X_kal(k)= X_kal(k-1) + a_x(X_kal(k-1)) * dt + K_kal(k-1) * (dZ(k-1) - c_x(X_kal(k-1)) * dt);  % Kalman Filtered state estimate   
      P(k)    = P(k-1)+ 2 * a_der_x(X_kal(k-1)) * P(k-1) * dt+ Q * dt - (K_kal(k-1)^2) * R * dt;     % Evolution of covariance
      K_kal(k)= (P(k)* c_der_x(X_kal(k-1)))/R;                                                % Computation of Kalman Gain     
 end

%% Displaying figures for diagnostics 

    % Plotting gains at k = 2, 3, 10, 20, 30, 40
    if ( No_runs == 1 && diag_main == 1 && ( k == 2 | k == 3 | k == (T/(10 * dt)) || k == 2 *(T/(10 * dt)) || k == 3 * (T/(10 * dt)) || k == (T/(2 * dt)) || k == (T/dt))) 
        figure;
        if exact == 1
            [Xi_exact_s, ind_exact] = sort(Xi_exact(k-1,:));
            plot(Xi_exact_s, K_exact(k,ind_exact), '--rv','MarkerSize',2,'DisplayName','Exact'); 
            hold on;
        end
        if fin == 1
            plot(Xi_fin(k-1,:), K_fin(k,:), '--g*', 'MarkerSize',2,'DisplayName','Finite');  
            hold on;
        end
        if coif == 1
            [Xi_coif_s, ind_coif] = sort(Xi_coif(k-1,:));
            plot(Xi_coif_s, K_coif(k,ind_coif), '--yo','MarkerSize',2,'DisplayName','Coifman');
            hold on;
            if exact_coif == 1
                plot(Xi_coif_s, K_exact_coif(k,ind_coif), '--k+','MarkerSize',2,'DisplayName','Coifman Exact'); 
            end
        end
        if rkhs == 1
            [Xi_rkhs_s, ind_rkhs] = sort(Xi_rkhs(k-1,:));
            plot(Xi_rkhs_s, K_rkhs(k,ind_rkhs), '--b^','MarkerSize',2,'DisplayName','RKHS'); 
            hold on;
            if exact_rkhs == 1
                plot(Xi_rkhs_s, K_exact_rkhs(k,ind_rkhs), '--k+','MarkerSize',2,'DisplayName','RKHS Exact'); 
            end
        end
        if zero_mean == 1
            [Xi_zm_s, ind_zm] = sort(Xi_zm(k-1,:));
            plot(Xi_zm_s, K_zm(k,ind_zm), '--c+','MarkerSize',2,'DisplayName','RKHS (ZM)'); 
            hold on;
            if exact_zm == 1
                plot(Xi_zm_s, K_exact_zm(k,ind_zm), '--k+','MarkerSize',2,'DisplayName','RKHS Exact'); 
            end
        end
        if memory == 1
            [Xi_mem_s, ind_mem] = sort(Xi_mem(k-1,:));
            plot(Xi_mem_s, K_mem(k,ind_mem), '--g+','MarkerSize',2,'DisplayName','RKHS memory'); 
            hold on;
            if exact_mem == 1
                plot(Xi_mem_s, K_exact_mem(k,ind_mem), '--k+','MarkerSize',2,'DisplayName','RKHS Exact'); 
            end
        end
        if zm_mem == 1
            [Xi_zm_mem_s, ind_zm_mem] = sort(Xi_zm_mem(k-1,:));
            plot(Xi_zm_mem_s, K_zm_mem(k,ind_zm_mem), 'Color', [0.2,0.3,0.7],'LineStyle','--','Marker','s','MarkerSize',2,'DisplayName','RKHS ZM + mem'); 
            % plot(Xi_zm_mem(k-1,:), K_zm_mem(k,:), 'Color', [0.2,0.3,0.7],'LineStyle','--','Marker','s','MarkerSize',2,'DisplayName','RKHS ZM + mem'); 
            hold on;
            if exact_zm_mem == 1
                % plot(Xi_zm_mem(k-1,:), K_exact_zm_mem(k,:), '--k+','MarkerSize',2,'DisplayName','RKHS Exact'); 
                plot(Xi_zm_mem_s, K_exact_zm_mem(k,ind_zm_mem), '--k+','MarkerSize',2,'DisplayName','RKHS Exact'); 
            end
        end
        if const ==1
            plot(Xi_const(k-1,:),K_const(k) * ones(1,N),'--mv','MarkerSize',2,'DisplayName','Const');
            hold on;
        end
        title(['Gain at particle locations for ' num2str(N) ' particles at t = ' num2str((k-2) * dt)]);
        legend('show');
    end
    
    if (No_runs==1 && (k == 2 || k == 3 | k == (T/(10 * dt)) || k == 2 *(T/(10 * dt)) || k == 3 * (T/(10 * dt)) || k == 0.5*(T/dt) || k == (T/dt)))
        step = 0.05;
        range = min(mu_em)- 3 * max(sigma_em): step : max(mu_em) + 3 * max(sigma_em);
        figure;
        if exact == 1
          p_t = 0;         
          for i = 1: length(mu_em)
              p_t = p_t + w_em(i) * exp(-( range - mu_em(i)).^2 / (2 * sigma_em(i)^2)) * (1 / sqrt(2 * pi * sigma_em(i)^2));
          end
          plot(range, p_t,'DisplayName','Exact');
          hold on;
       end
       if kalman == 1
           p_kal_t = exp(-( range - X_kal(k)).^2 / (2 * P(k))) * (1 / sqrt(2 * pi * P(k)));
           plot(range, p_kal_t,'DisplayName','EKF');
           hold on;
       end
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
            if zero_mean == 1
               hist(Xi_zm(k-1,:),N);
            end  
            if memory == 1
               hist(Xi_mem(k-1,:),N);
            end
            if zm_mem == 1
               hist(Xi_zm_mem(k-1,:),N);
            end
            if const == 1
               hist(Xi_const(k-1,:),N);
            end             
       else
            if fin == 1
             % CAUTION : histogram command works only in recent Matlab
             % versions, if it does not work, comment this section out
               histogram(Xi_fin(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName','Finite');
               hold on;
            end
            if coif == 1
               histogram(Xi_coif(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName','Coifman');
               hold on;
            end
            if rkhs == 1
               histogram(Xi_rkhs(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName','RKHS');
               hold on;
            end
            if zero_mean == 1
               histogram(Xi_zm(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName','RKHS(ZM)');
               hold on;
            end
            if memory == 1
               histogram(Xi_mem(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName','RKHS memory');
               hold on;
            end
            if zm_mem == 1
               histogram(Xi_zm_mem(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName','RKHS ZM + memory');
               hold on;
            end
            if const == 1
               histogram(Xi_const(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName','Const');
               hold on;
            end
       end  
       if sis == 1
           [~,ind] = sort(Xi_sis(k-1,:));
           % plot(Xi_sis(k-1,ind),Wi_sis(k-1,ind)/step,'linewidth',2.0);
           p_sis_t = 0;
           sigma_sis = 0.05;
           for i = 1 : length(Xi_sis(k-1,:))
               p_sis_t = p_sis_t + Wi_sis(k-1,i) *  exp(- (range - Xi_sis(k-1,i)).^2 / ( 2 * sigma_sis^2)) * (1 / sqrt(2 * pi * sigma_sis^2));
           end
           plot(range, p_sis_t,'DisplayName','SIS PF');
           hold on;
       end
       legend('show');
       title(['Posterior density \rho_t estimates at t =' num2str((k-1)*dt)]);
    end
end

%% Computing the rmse metric

if exact ==1
    mu_exact(k)     = mean(Xi_exact(k,:));
    rmse_exact(run) = (1 / (T/dt)) * sum ( (X - mu_exact).^2);
end
if fin == 1
    mu_fin(k)       = mean(Xi_fin(k,:));
    rmse_fin(run)   = (1 / (T/dt)) * sum ( (X - mu_fin).^2);
end
if coif == 1
    mu_coif(k)      = mean(Xi_coif(k,:));
    rmse_coif(run)  = (1 / (T/dt)) * sum ( (X - mu_coif).^2);
end
if rkhs == 1
    mu_rkhs(k)      = mean(Xi_rkhs(k,:));
    rmse_rkhs(run)  = (1 / (T/dt)) * sum ( (X - mu_rkhs).^2);
end
if zero_mean == 1
    mu_zm(k)      = mean(Xi_zm(k,:));
    rmse_zm(run)  = (1 / (T/dt)) * sum ( (X - mu_zm).^2);
end
if memory == 1
    mu_mem(k)      = mean(Xi_mem(k,:));
    rmse_mem(run)  = (1 / (T/dt)) * sum ( (X - mu_mem).^2);
end
if zm_mem == 1
    mu_zm_mem(k)      = mean(Xi_zm_mem(k,:));
    rmse_zm_mem(run)  = (1 / (T/dt)) * sum ( (X - mu_zm_mem).^2);
end
if const == 1
    mu_const(k)     = mean(Xi_const(k,:));
    rmse_const(run) = (1 / (T/dt)) * sum ( (X - mu_const).^2);
end
if sis == 1
    mu_sis(k)       = Wi_sis(k,:) * Xi_sis(k,:)';
    rmse_sis(run)   = (1 / (T/dt)) * sum ( (X - mu_sis).^2);
end
if kalman == 1
    rmse_kal(run)   = (1 / (T/dt)) * sum ( (X - X_kal).^2);
end

%% Plots
if diag_output == 1
    figure;
    plot(0:dt:(k-1)*dt, X(1:k),'k','DisplayName','Actual state');
    hold on;
    if exact == 1
        plot(0:dt:(k-1)*dt, mu_exact(1:k),'r--','DisplayName','Exact');
        hold on;
    end
    if fin == 1
        plot(0:dt:(k-1)*dt, mu_fin(1:k),'y--','DisplayName','Finite');
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
    if zero_mean == 1
        plot(0:dt:(k-1)*dt, mu_zm(1:k),'b--','DisplayName','RKHS(ZM)');
        hold on;
    end
    if memory == 1
        plot(0:dt:(k-1)*dt, mu_mem(1:k),'g--','DisplayName','RKHS memory');
        hold on;
    end
    if zm_mem == 1
        plot(0:dt:(k-1)*dt, mu_zm_mem(1:k),'Color',[0.2,0.3,0.7],'LineStyle','--','DisplayName','RKHS ZM + memory');
        hold on;
    end
    if const == 1
        plot(0:dt:(k-1)*dt, mu_const(1:k),'g--','DisplayName','Const');
        hold on;
    end
    if kalman == 1
        plot(0:dt:(k-1)*dt, X_kal(1:k),'c--','DisplayName','Kalman');
        hold on;
    end
    if sis == 1
        plot(0:dt:(k-1)*dt, mu_sis(1:k),'m--','DisplayName','SIS');
        hold on;
    end
    legend('show');
    title(['a =' a_legend ', \sigma_B = ' num2str(sigmaB) ', \sigma_W =' num2str(sigmaW) ', c = ' char(c) ]);
end

if No_runs == 1 && diag_main == 1
    figure;
    plot(0:dt:(k-1)*dt, Z(1:k),'r');
    title('Z_t');

    if kalman == 1
        figure;
        plot(0:dt:(k-1)*dt, K_kal(1:k),'r');
        title('Kalman Gain K_t');
    
        figure;
        plot(0:dt:(k-1)*dt, P(1:k),'r');
        hold on;
        title('State Covariance P_t');
    end
    
    if sis == 1
        figure;
        plot(0:dt:(k-1)*dt, N_eff_sis(1:k),'r');
        hold on;
        title('Effective particle size N_{eff} in SIS PF');
    end
    
 end
end

% Overall rmse 

if exact == 1
    rmse_tot_exact = (1 / No_runs) * sum( rmse_exact);
    sprintf('RMSE for exact gain computation - %0.5g', rmse_tot_exact)
end
if fin == 1
    rmse_tot_fin = (1 / No_runs) * sum ( rmse_fin );
    sprintf('RMSE for finite basis - %0.5g', rmse_tot_fin)
end
if coif == 1
    rmse_tot_coif = (1 / No_runs) * sum ( rmse_coif);
    sprintf('RMSE for Coifman method - %0.5g', rmse_tot_coif)
end
if rkhs == 1
    rmse_tot_rkhs = (1 / No_runs) * sum( rmse_rkhs );
    sprintf('RMSE for RKHS method - %0.5g', rmse_tot_rkhs)
end
if zero_mean == 1
    rmse_tot_zm = (1 / No_runs) * sum( rmse_zm );
    sprintf('RMSE for RKHS(ZM) method - %0.5g', rmse_tot_zm)
end
if memory == 1
    rmse_tot_mem = (1 / No_runs) * sum( rmse_mem );
    sprintf('RMSE for RKHS memory method - %0.5g', rmse_tot_mem)
end
if zm_mem == 1
    rmse_tot_zm_mem = (1 / No_runs) * sum( rmse_zm_mem );
    sprintf('RMSE for RKHS ZM + memory method - %0.5g', rmse_tot_zm_mem)
end
if const == 1
    rmse_tot_const = (1 / No_runs) * sum( rmse_const);
    sprintf('RMSE for const gain approximation - %0.5g', rmse_tot_const)
end
if sis == 1
    rmse_tot_sis = (1 / No_runs) * sum ( rmse_sis);
    sprintf('RMSE for SIS PF - %0.5g', rmse_tot_sis)
end
if kalman == 1
    rmse_tot_kal   = (1 / No_runs) * sum( rmse_kal);
    sprintf('RMSE for Kalman Filter - %0.5g', rmse_tot_kal)
end
toc;

filename = 'FPF_results_100.xlsx';
% A = {'Exact','RKHS','RKHS ZM','RKHS memory','RKHS ZM + mem','Const','SIS','EKF'; rmse_tot_exact, rmse_tot_rkhs, rmse_tot_zm, rmse_tot_mem, rmse_tot_zm_mem, rmse_tot_const, rmse_tot_sis, rmse_tot_kal};
% A = [rmse_tot_exact; rmse_tot_rkhs; rmse_tot_zm; rmse_tot_mem; rmse_tot_zm_mem; rmse_tot_const; rmse_tot_sis; rmse_tot_kal];
% xlswrite(filename, A, 2, 'B1');
% dlmwrite(filename, A, '-append');


