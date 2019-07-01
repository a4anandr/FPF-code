%% Feedback particle filter for state estimation problems in the multidimension
% In particular, we look at the dynamics of the ship example in the
% following papers 
% [1] ' A comparative study of nonlinear filtering techniques' - Tilton,
% Ghiotto, Mehta
% [2] ' A survey of numerical methods for nonlinear filtering problems' -
% Budhiraja
% This function performs 
% - Defines the state and observation processes - X_t and Z_t
% - Defines the prior distribution p(0) 
% - Generates particles Xi from the prior p(0)
% - Passes these particles Xi to four different gain function approximation
% functions - i) diff TD learning (old), ii) finite dim, iii) Coifman
% kernel, iv) RKHS v) Kalman filter vi) SIS PF
% based.

clear;
clc;
close all;
tic
warning off;

diag_main = 0;   % Diagnostics flag for main function, displays figures in main.
diag_output = 1; % Diagnostics flag to display the main output in this function
diag_fn = 0;     % Diagnostics flag, if 1, then all the functions display plots for diagnostics, Set it to 0 to avoid plots from within the calling functions
% rng(1000);     % Set a common seed
No_runs = 1;   % Total number of runs to compute the rmse metric for each of the filters for comparison

%% Flags to be set to choose which methods to compare
coif  = 0;           % Computes gain using Coifman kernel method
rkhs  = 1;           % Computes gain using RKHS
zero_mean = 1;       % Computes gain using RKHS that takes into account the constant gain approximation values
zm_mem = 1;          % Computes gain using const gain approx and a memory parameter for previous gain
const = 1;           % Computes the constant gain approximation
sis    = 0;          % Runs Sequential Importance Sampling Particle Filter 
kalman = 0;          % Runs Kalman Filter for comparison

%% Parameters corresponding to the state and observation processes
% Run time parameters
T   = 8.25;         % Total running time, using the same value in references [1],[2]
delta = 0.05;      % Time increments for the SDE and observation model 
sdt   = sqrt(delta); 

% Model parameters
zeta  = 2;
Theta = 50;
rho   = 9;
d     = 2;        % State space dimension
tol   = 10;       % Tolerance limit for 'losing the track' - 22 is used in the paper for square of the distance

% State process initialization
x =sym('x',[1 2]);
mag_x = @(x)sqrt(x(1)^2 + x(2)^2);
f1_x = @(x)zeta * ( x(1) / mag_x(x)^2) - Theta * (x(1) / mag_x(x)) * ( mag_x(x) > rho) ;
f2_x = @(x)zeta * ( x(2) / mag_x(x)^2) - Theta * (x(1) / mag_x(x)) * ( mag_x(x) > rho) ;

% Process noise parameters
e1 = 0.4;
e2 = 0.4; 

% Observation process parameters
h_x = @(x)atan(x(2)/x(1));
theta = 0.32;              % Standard deviation parameter in observation process
R     = 1;                 % theta^2, Observation noise covariance

%% Parameters of the prior p(0) - Multivariate Gaussian density 
X_0  = [ 0.5 -0.5];
% Sig = 10 * eye(2);                 % 10 * eye(2) in the paper
Sig = [5 0; 0 5];
% Sig = [1 0; 0 1];

%% Filter parameters

N = 500;          % No of particles - Common for all Monte Carlo methods used

% Setting a max and min threshold for gain
K_max = 100;
K_min = -100;
 
%% i) Coifman kernel 
if coif == 1
   eps_coif = 0.1;   % Time step parameter
   lose_coif = 0;    % Initializing the "lose the track" count as specified in Budhiraja et al.
end

%% ii) RKHS
if rkhs == 1
   kernel   = 0;            % 0 for Gaussian kernel
   lambda   = 1e-1;         % 0.01 has worked best so far for Sig = [1 0 ; 0 1]
   eps_rkhs = 2;            % Variance parameter of the kernel  - 2 has worked best so far for the same Sig
   K_rkhs   = ones(1,N,d);  % Initializing the gain to a 1 vector, this value is used only at k = 1. 
   
   lose_rkhs = 0;           % Initializing the "lose the track" count as specified in Budhiraja et al.
end

%% iii) RKHS zero mean
if zero_mean == 1
   kernel    = 0;           % 0 for Gaussian kernel
   lambda_zm = 1e-1;        % 1e-1 has worked best so far for Sig = [5 0 ; 0 5]
   eps_zm    = 2;           % Variance parameter of the kernel  - 2 has worked best so far for the same Sig
   K_zm      = zeros(1,N,d);    % Initializing the gain to a 1 vector, this value is used only at k = 1. 
   
   lose_zm = 0;             % Initializing the "lose the track" count as specified in Budhiraja et al.

end

%% iv) RKHS ZM with memory
if zm_mem == 1
   kernel          = 0;          % 0 for Gaussian kernel, 1 for Coifman kernel, 2 for approximate Coifman kernel
   lambda_zm_mem   = 1e-1;       % 0.05, 0.02, Regularization parameter - Other tried values ( 0.005,0.001,0.05), For kernel = 0, range 0.005 - 0.01.
   eps_zm_mem      = 2;          % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
   lambda_gain_zm_mem = 1;       % 1e-4; % This parameter decides how much the gain can change in successive time instants, higher value implying less variation. 
   K_zm_mem        = zeros(1,N,d); % Initializing the gain to a 1 vector, this value is used only at k = 1. 
   beta_zm_mem     = zeros(N,1); % Initializing the parameter values beta to zero.
   
   lose_zm_mem     = 0 ;         % Initializing the "lose the track" count as specified in Budhiraja et al.

end

%% v) Constant gain approximation
if const == 1
    lose_const = 0;        % Initializing the "lose the track" count as specified in Budhiraja et al.
end

%% vi) SIS PF
if sis == 1 
    resampling = 0;        % Whether you need deterministic resampling 
    lag        = 3;        % After how many time samples should periodic resampling be done
    
    lose_sis = 0;          % Initializing the "lose the track" count as specified in Budhiraja et al.
end

%% vii) Extended Kalman Filter
% Declaring symbolic functions and derivatives to be used in EKF
if kalman == 1
   syms y z;
   % State process - Jacobian
   f1_large = - z + zeta * (y/(y^2 + z^2)) - Theta * (y/sqrt(y^2 + z^2));
   f1_small = - z + zeta * (y/(y^2 + z^2));
   f2_large = y + zeta * (z/(y^2 + z^2)) - Theta * (z/sqrt(y^2 + z^2));
   f2_small = y + zeta * (z/(y^2 + z^2));
   
   % Derivative functions to be used if x lies outside the ball of radius
   % rho
   f1_large_y = diff(f1_large,y);
   f1_large_z = diff(f1_large,z);
   f2_large_y = diff(f2_large,y);
   f2_large_z = diff(f2_large,z);
   
   % Derivative functions to be used if x lies within the ball of radius
   % rho
   f1_small_y = diff(f1_small,y);
   f1_small_z = diff(f1_small,z);
   f2_small_y = diff(f2_small,y);
   f2_small_z = diff(f2_small,z);
  
   % Observation process - Jacobian
   h   = atan(z/y);
   h_y = diff(h,y);
   h_z = diff(h,z);
   
   % Process noise covariance - Q
   Q = [ e1 0 ; 0 e2];
   
   lose_kal = 0;    % Initializing the "lose the track" count as specified in Budhiraja et al.
   
end

for run = 1: 1 : No_runs
    run
    ind_count = 0;             % To count the number of times the trajectory goes out of the circle 
    
%% State and observation process initialization
    X(1,:)   = X_0;
    Z(1)     = h_x(X(1,:)) + theta * randn;
    Z_true(1)= h_x(X(1,:));

%% Initializing N particles from the prior
    Xi_0  = mvnrnd(X_0,Sig,N);
    mui_0 = mean(Xi_0);
    
    if coif == 1
        Xi_coif      = Xi_0;
        first_coif   = 0;
    end
    
    if rkhs ==1
        Xi_rkhs      = Xi_0;
        first_rkhs   = 0;
    end
    
    if zero_mean == 1
        Xi_zm        = Xi_0;
        first_zm     = 0;
    end
    
    if zm_mem == 1
        Xi_zm_mem        = Xi_0;
        first_zm_mem     = 0;
    end
    
    if const == 1
        Xi_const     = Xi_0;
        first_const  = 0;
    end
    
%  vi) Sequential Importance Sampling Particle Filter Initialization
    if sis == 1
       Xi_sis       = Xi_0;
       Wi_sis(:,1)  = (1/N) * ones(1,N);          % Initializing all weights to equal value.
       for i = 1:N
           Zi_sis(i,:)  = h_x(Xi_sis(i,:));
       end
       first_sis     = 0;
    end

% vii) Extended Kalman filter - Initialization
    if kalman == 1
       X_kal        = mean(Xi_0);     % Initializing the Kalman filter state estimate to the mean at t = 0, More accurate initialization is X_0.
       P(:,:,1)     = Sig;            % Initializing the state covariance for the Kalman filter 
       H            = [ subs(h_y,{y,z},X_kal) subs(h_z,{y,z},X_kal)];
       K_kal(:,1)   = P(:,:,1) * H';
       first_kal    = 0;
    end

for k = 2: 1: (T/delta)    
    % k  
    %% Actual state - observation process evolution
    X(k,1)   = X(k-1,1) - X(k-1,2) * delta +   f1_x(X(k-1,:)) * delta + e1 * sdt * randn;        
    X(k,2)   = X(k-1,2) + X(k-1,1) * delta +   f2_x(X(k-1,:)) * delta + e2 * sdt * randn;
    if (mag_x(X(k,:)) > rho)                                                                     % Counts the number of times the trajectory goes out of the circle of radius \rho
       ind_count = ind_count + 1;
    end
    Z(k)     = h_x(X(k,:))  + theta * randn; 
    Z_true(k)= h_x(X(k,:));
     
    %% Filters
    if coif == 1
        [h_hat_coif(k-1) K_coif(k,:,:) ] = gain_coif_multi(Xi_coif(:,:,k-1) , h_x, d, eps_coif, diag_fn);
        mu_coif(k-1,:)   = mean(Xi_coif(:,:,k-1));
        if (norm(mu_coif(k-1,:) - X(k-1,:)) > tol && first_coif == 0)
            first_coif = first_coif + 1;
            lose_coif = lose_coif + 1;
        end
        K_const_coif(k,:)= mean(K_coif(k,:,:),2);
    end 
    
    if rkhs == 1
        [h_hat_rkhs(k-1) K_rkhs(k,:,:)] = gain_rkhs_multi(Xi_rkhs(:,:,k-1) , h_x, d , kernel,lambda, eps_rkhs, diag_fn);
        mu_rkhs(k-1,:)   = mean(Xi_rkhs(:,:,k-1));
        if ( norm(mu_rkhs(k-1,:) - X(k-1,:)) > tol && first_rkhs == 0)
            lose_rkhs  = lose_rkhs + 1;
            first_rkhs = first_rkhs + 1;
        end
        K_const_rkhs(k,:)= mean(K_rkhs(k,:,:),2);
    end
    
    if zero_mean == 1
        [h_hat_zm(k-1) K_zm(k,:,:)] = gain_rkhs_zero_mean(Xi_zm(:,:,k-1) , h_x, d , kernel,lambda_zm, eps_zm, diag_fn);
        mu_zm(k-1,:)   = mean(Xi_zm(:,:,k-1));
        if (norm(mu_zm(k-1,:) - X(k-1,:)) > tol && first_zm == 0)
            lose_zm  = lose_zm + 1;
            first_zm = first_zm + 1;
        end
        K_const_zm(k,:)= mean(K_zm(k,:,:),2);
    end
    
    if zm_mem == 1
            if k == 2
                [h_hat_zm_mem(k-1), beta_zm_mem, K_zm_mem(k,:,:)] = gain_rkhs_zm_mem(Xi_zm_mem(:,:,k-1) , h_x, d , kernel,lambda_zm_mem, eps_zm_mem, 0, K_zm_mem(k-1,:,:), beta_zm_mem, zeros(N,d), diag_fn);
            else
                [h_hat_zm_mem(k-1), beta_zm_mem, K_zm_mem(k,:,:)] = gain_rkhs_zm_mem(Xi_zm_mem(:,:,k-1) , h_x, d , kernel,lambda_zm_mem, eps_zm_mem, lambda_gain_zm_mem, K_zm_mem(k-1,:,:), beta_zm_mem, Xi_zm_mem(:,:,k-2),diag_fn);
            end        
            mu_zm_mem(k-1,:)   = mean(Xi_zm_mem(:,:,k-1));
        if (norm(mu_zm_mem(k-1,:) - X(k-1,:)) > tol && first_zm_mem == 0)
            lose_zm_mem  = lose_zm_mem + 1;
            first_zm_mem = first_zm_mem + 1;
        end
        K_const_zm_mem(k,:)= mean(K_zm_mem(k,:,:),2);
    end
    
    % max(abs(K_zm(k,:,:) - K_zm_mem(k,:,:)))
    % K_const_zm_mem
    % K_hat_zm_mem
    
    if const == 1   
        [h_hat_const(k-1) K_const(k,:)] = gain_const_multi(Xi_const(:,:,k-1), h_x, d, diag_fn);
        mu_const(k-1,:)   = mean(Xi_const(:,:,k-1));
        if (norm(mu_const(k-1,:) - X(k-1,:)) > tol && first_const ==0)
            lose_const  = lose_const + 1;
            first_const = first_const + 1;
        end
    end
    
    if sis == 1
        mu_sis(k-1,:)     = Wi_sis(:,k-1)' * Xi_sis(:,:,k-1);
        N_eff_sis(k-1)    = 1 / sum(Wi_sis(:,k-1).^2);
        if (norm(mu_sis(k-1,:) - X(k-1,:)) > tol && first_sis == 0)
            lose_sis  = lose_sis + 1;
            first_sis = first_sis + 1;
        end
    end
        
    for i = 1:N
       common_rand = randn(2,1);
       
       % i) Coifman kernel
       if coif == 1
           dI_coif(k)       = Z(k-1) - 0.5 * (h_x(Xi_coif(i,:,k-1)) + h_hat_coif(k-1));
           K_coif(k,i,1)    = min(max(K_coif(k,i,1),K_min),K_max);
           K_coif(k,i,2)    = min(max(K_coif(k,i,2),K_min),K_max);
           Xi_coif(i,1,k)   = Xi_coif(i,1,k-1) - Xi_coif(i,2,k-1) * delta + f1_x(Xi_coif(i,:,k-1)) * delta + e1 * sdt * common_rand(1) + (K_coif(k,i,1)/R) * dI_coif(k);
           Xi_coif(i,2,k)   = Xi_coif(i,2,k-1) - Xi_coif(i,1,k-1) * delta + f2_x(Xi_coif(i,:,k-1)) * delta + e2 * sdt * common_rand(2) + (K_coif(k,i,2)/R) * dI_coif(k);
       end
       
       % ii) RKHS
       if rkhs == 1
           dI_rkhs(k)       = Z(k-1) - 0.5 * (h_x(Xi_rkhs(i,:,k-1)) + h_hat_rkhs(k-1));
           K_rkhs(k,i,1)    = min(max(K_rkhs(k,i,1),K_min),K_max);
           K_rkhs(k,i,2)    = min(max(K_rkhs(k,i,2),K_min),K_max);
           Xi_rkhs(i,1,k)   = Xi_rkhs(i,1,k-1) - Xi_rkhs(i,2,k-1) * delta + f1_x(Xi_rkhs(i,:,k-1)) * delta + e1 * sdt * common_rand(1) + (K_rkhs(k,i,1)/R) * dI_rkhs(k);       % K_rkhs(k,i,1) * dI_rkhs(k)
           Xi_rkhs(i,2,k)   = Xi_rkhs(i,2,k-1) + Xi_rkhs(i,1,k-1) * delta + f2_x(Xi_rkhs(i,:,k-1)) * delta + e2 * sdt * common_rand(2) + (K_rkhs(k,i,2)/R) * dI_rkhs(k);
       end
       
       % iii) RKHS zero mean
       if zero_mean == 1
           dI_zm(k)       = Z(k-1) - 0.5 * (h_x(Xi_zm(i,:,k-1)) + h_hat_zm(k-1));
           K_zm(k,i,1)    = min(max(K_zm(k,i,1),K_min),K_max);
           K_zm(k,i,2)    = min(max(K_zm(k,i,2),K_min),K_max);
           Xi_zm(i,1,k)   = Xi_zm(i,1,k-1) - Xi_zm(i,2,k-1) * delta + f1_x(Xi_zm(i,:,k-1)) * delta + e1 * sdt * common_rand(1) + (K_zm(k,i,1)/R) * dI_zm(k);       % K_rkhs(k,i,1) * dI_rkhs(k)
           Xi_zm(i,2,k)   = Xi_zm(i,2,k-1) + Xi_zm(i,1,k-1) * delta + f2_x(Xi_zm(i,:,k-1)) * delta + e2 * sdt * common_rand(2) + (K_zm(k,i,2)/R) * dI_zm(k);
       end
       
       % iv) RKHS zero mean with memory
       if zm_mem == 1
           dI_zm_mem(k)     = Z(k-1) - 0.5 * (h_x(Xi_zm_mem(i,:,k-1)) + h_hat_zm_mem(k-1));
           K_zm_mem(k,i,1)  = min(max(K_zm_mem(k,i,1),K_min),K_max);
           K_zm_mem(k,i,2)  = min(max(K_zm_mem(k,i,2),K_min),K_max);
           Xi_zm_mem(i,1,k) = Xi_zm_mem(i,1,k-1) - Xi_zm_mem(i,2,k-1) * delta + f1_x(Xi_zm_mem(i,:,k-1)) * delta + e1 * sdt * common_rand(1) + (K_zm_mem(k,i,1)/R) * dI_zm_mem(k);       % K_rkhs(k,i,1) * dI_rkhs(k)
           Xi_zm_mem(i,2,k) = Xi_zm_mem(i,2,k-1) + Xi_zm_mem(i,1,k-1) * delta + f2_x(Xi_zm_mem(i,:,k-1)) * delta + e2 * sdt * common_rand(2) + (K_zm_mem(k,i,2)/R) * dI_zm_mem(k);
       end
       
       % v) Constant gain approximation 
       if const == 1
           dI_const(k)      = Z(k-1) - 0.5 * (h_x(Xi_const(i,:,k-1)) + h_hat_const(k-1));          
           Xi_const(i,1,k)  = Xi_const(i,1,k-1) - Xi_const(i,2,k-1) * delta + f1_x(Xi_const(i,:,k-1)) * delta + e1 * sdt * common_rand(1) + (K_const(k,1)/R) * dI_const(k);    % (1/theta^2) is actually required?
           Xi_const(i,2,k)  = Xi_const(i,2,k-1) + Xi_const(i,1,k-1) * delta + f2_x(Xi_const(i,:,k-1)) * delta + e2 * sdt * common_rand(2) + (K_const(k,2)/R) * dI_const(k);
       end
       
       % vi) Sequential Importance Sampling Particle Filter (SIS PF)
       if sis == 1
          Xi_sis(i,1,k)     = Xi_sis(i,1,k-1) - Xi_sis(i,2,k-1) * delta  +  f1_x(Xi_sis(i,:,k-1)) * delta + e1 * sdt * common_rand(1); 
          Xi_sis(i,2,k)     = Xi_sis(i,2,k-1) + Xi_sis(i,1,k-1) * delta  +  f2_x(Xi_sis(i,:,k-1)) * delta + e2 * sdt * common_rand(2); 
          Zi_sis(i,k)       = h_x(Xi_sis(i,:,k)); 
          Wi_sis(i,k)       = Wi_sis(i,k-1) * (1/sqrt( 2 * pi * theta^2 * delta)) * exp ( - (Z(k) - Zi_sis(i,k)).^2/ (2 * theta^2 * delta));   %  Based on eqn(63) of Arulampalam et al. In our example, the importance density is the prior density p(X_t | X_{t-1}). If resampling is done at every step then the recursive form disappears. Wi_sis(k) does not depend on Wi_sis(k-1) as Wi_sis(k-1) = 1/N.
       end   
    end
    
 if sis == 1
 % Normalizing the weights of the SIS - PF
   Wi_sis(:,k)  = Wi_sis(:,k)/ sum(Wi_sis(:,k));
   if resampling == 1
 % Deterministic resampling - as given in Budhiraja et al.
        if mod(k,lag)== 0
            sum_N_eff = 0;
            Wi_cdf    = zeros(N,1);
            for i = 1 : N
                N_eff(i) = floor(Wi_sis(i,k) *  N); 
                if isnan(N_eff(i)) == 1
                    N_eff(i) = 0;
                end
                Wi_res(i)= Wi_sis(i,k) - N_eff(i)/ N;
                if i == 1
                    Wi_cdf(i)= Wi_res(i);
                else
                    Wi_cdf(i)= Wi_cdf(i-1) + Wi_res(i);
                end
                if N_eff(i) > 0
                    Xi_sis_new (sum_N_eff + 1 : sum_N_eff + N_eff(i),:) = repmat(Xi_sis(i,:,k),N_eff(i),1);
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
                        Xi_sis_new (sum_N_eff + j,:) = Xi_sis(i,:,k);
                    end
                end
            end
            Xi_sis(:,:,k)= Xi_sis_new;
            Wi_sis(:,k)    = (1/N) * ones(1,N);
        end
   end
   N_eff_sis(k) = 1 / (sum(Wi_sis(k,:).^2)); 
 end
     
 % iv) Extended Kalman Filter for comparison
 if kalman == 1
      X_kal(k,1)        = X_kal(k-1,1) - X_kal(k-1,2) * delta + f1_x(X_kal(k-1,:)) * delta + ( K_kal(1,k-1)/R) * (Z(k-1) - h_x(X_kal(k-1,:)));  % Kalman Filtered state estimate   
      X_kal(k,2)        = X_kal(k-1,2) + X_kal(k-1,1) * delta + f2_x(X_kal(k-1,:)) * delta + ( K_kal(2,k-1)/R) * (Z(k-1) - h_x(X_kal(k-1,:))); 
      if (mag_x(X_kal(k-1,:)) > rho)
          A = [ eval(subs(f1_large_y,{y,z},X_kal(k-1,:))) eval(subs(f1_large_z,{y,z},X_kal(k-1,:))); 
                eval(subs(f2_large_y,{y,z},X_kal(k-1,:))) eval(subs(f2_large_z,{y,z},X_kal(k-1,:)))];
      else
          A = [ eval(subs(f1_small_y,{y,z},X_kal(k-1,:))) eval(subs(f1_small_z,{y,z},X_kal(k-1,:))); 
                eval(subs(f2_small_y,{y,z},X_kal(k-1,:))) eval(subs(f2_small_z,{y,z},X_kal(k-1,:)))];
      end
      if (norm(X_kal(k,:) - X(k,:)) > tol && first_kal == 0)
          lose_kal = lose_kal + 1;
          first_kal = first_kal + 1;
      end
      H                 = [ eval(subs(h_y,{y,z},X_kal(k-1,:)))  eval(subs(h_z,{y,z},X_kal(k-1,:)))];
      P(:,:,k)          = P(:,:,k-1)+ ( A * P(:,:,k-1) + P(:,:,k-1) * A' + Q - K_kal(:,k-1) * H * P(:,:,k-1)) * delta;     % Evolution of covariance    
      K_kal(:,k)        = P(:,:,k) * H';                                   % Computation of Kalman Gain     
 end

%% Displaying figures for diagnostics 
% Plotting histograms of particles at k = 2, 3, 10, 20, 30, 40
    if ( diag_main == 1 && ( k == 2 || k == 3 || k == 11 || k == 21 || k == 31 || k == (T/delta))) 
       clf
       figure(100);
       subplot(2,1,1);
       if const == 1
          hist3(Xi_const(:,:,k),[100 100]);
       end
       subplot(2,1,2);
       if rkhs == 1
          hist3(Xi_rkhs(:,:,k),[100 100]);
       end
    end
end    

%% Computing the rmse metric and the max error metric (to count losing the track)

if coif == 1
    mu_coif(k,:)       = mean(Xi_coif(k,:));
    rmse_coif(run)     = mean(vecnorm(X - mu_coif,2,2));                                            % (1 / (T/delta)) * (sum(sqrt(sum((X - mu_coif).^2,2))));
    max_diff_coif(run) = max(vecnorm(X - mu_coif,2,2)); 
end
if rkhs == 1
    mu_rkhs(k,:)       = mean(Xi_rkhs(:,:,k));
    rmse_rkhs(run)     = mean(vecnorm(X - mu_rkhs,2,2)); 
    max_diff_rkhs(run) = max(vecnorm(X - mu_rkhs,2,2)); 
end
if zero_mean == 1
    mu_zm(k,:)         = mean(Xi_zm(:,:,k));
    rmse_zm(run)       = mean(vecnorm(X - mu_zm,2,2));
    max_diff_zm(run)   = max(vecnorm(X - mu_zm,2,2)); 
end
if zm_mem == 1
    mu_zm_mem(k,:)       = mean(Xi_zm_mem(:,:,k));
    rmse_zm_mem(run)     = mean(vecnorm(X - mu_zm_mem,2,2));
    max_diff_zm_mem(run) = max(vecnorm(X - mu_zm_mem,2,2)); 
end
if const == 1
    mu_const(k,:)      = mean(Xi_const(:,:,k));
    rmse_const(run)    = mean(vecnorm(X - mu_const,2,2));
    max_diff_const(run)= max(vecnorm(X - mu_const,2,2)); 
end
if sis == 1
    mu_sis(k,:)       = Wi_sis(:,k)' * Xi_sis(:,:,k);
    rmse_sis(run)     = mean(vecnorm(X - mu_sis,2,2));
    max_diff_sis(run) = max(vecnorm(X - mu_sis,2,2)); 
end
if kalman == 1
    rmse_kal(run)     = mean(vecnorm(X - X_kal,2,2));
    max_diff_kal(run) = max(vecnorm(X - X_kal,2,2)); 
end

%% Plotting the state trajectory and estimates
if (diag_output == 1 && No_runs == 1)   
    figure;
    plot(X(1:k,1),X(1:k,2),'k','linewidth',2.0,'DisplayName','True state');
    hold on;
    th = 0:pi/50:2*pi;
    xunit = rho * cos(th);
    yunit = rho * sin(th);
    plot(xunit, yunit,'r--','DisplayName','|x| < \rho');
    if coif == 1
        plot(mu_coif(1:k,1), mu_coif(1:k,2),'c--','linewidth',2.0,'DisplayName','FPF - Coifman');
        hold on;
    end
    if rkhs == 1
        plot(mu_rkhs(1:k,1), mu_rkhs(1:k,2),'k--','linewidth',2.0,'DisplayName','FPF - RKHS');
        hold on;
    end
    if zero_mean == 1
        plot(mu_zm(1:k,1), mu_zm(1:k,2),'b--','linewidth',2.0,'DisplayName','FPF - RKHS(ZM)');
        hold on;
    end
    if zm_mem == 1
        plot(mu_zm_mem(1:k,1), mu_zm_mem(1:k,2),'b--','linewidth',2.0,'DisplayName','FPF - RKHS ZM + memory');
        hold on;
    end
    if const == 1
        plot(mu_const(1:k,1), mu_const(1:k,2),'c--','linewidth',2.0,'DisplayName','FPF - Const gain');
        hold on;
    end
    if kalman == 1
        plot(X_kal(1:k,1), X_kal(1:k,2),'g--','linewidth',2.0,'DisplayName','EKF');
        hold on;
    end
    if sis == 1
        plot(mu_sis(1:k,1), mu_sis(1:k,2),'m--','linewidth',2.0,'DisplayName','SIS PF');
        hold on;
    end
    legend('show');
   
    figure;
    plot(0:delta:(k-1)*delta, X(1:k,1),'k','DisplayName','True state');
    hold on;
    if coif == 1
        plot(0:delta:(k-1)*delta, mu_coif(1:k,1),'c--','linewidth',2.0,'DisplayName','FPF - Coifman');
        hold on;
    end
    if rkhs == 1
        plot(0:delta:(k-1)*delta, mu_rkhs(1:k,1),'k--','linewidth',2.0,'DisplayName','FPF - RKHS');
        hold on;
    end
    if zero_mean == 1
        plot(0:delta:(k-1)*delta, mu_zm(1:k,1),'b--','linewidth',2.0,'DisplayName','FPF - RKHS(ZM)');
        hold on;
    end
    if zm_mem == 1
        plot(0:delta:(k-1)*delta, mu_zm_mem(1:k,1),'b--','linewidth',2.0,'DisplayName','FPF - RKHS ZM + memory');
        hold on;
    end
    if const == 1
        plot(0:delta:(k-1)*delta, mu_const(1:k,1),'r--','linewidth',2.0,'DisplayName','FPF - Const gain');
        hold on;
    end
    if kalman == 1
        plot(0:delta:(k-1)*delta, X_kal(1:k,1),'g--','linewidth',2.0,'DisplayName','EKF');
        hold on;
    end
    if sis == 1
        plot(0:delta:(k-1)*delta, mu_sis(1:k,1),'m--','linewidth',2.0,'DisplayName','SIS PF');
        hold on;
    end
    legend('show');
    
    figure;
    plot(0:delta:(k-1)*delta, X(1:k,1),'k','DisplayName','True state');
    hold on;
    if coif == 1
        plot(0:delta:(k-1)*delta, mu_coif(1:k,2),'c--','linewidth',2.0,'DisplayName','FPF - Coifman');
        hold on;
    end
    if rkhs == 1
        plot(0:delta:(k-1)*delta, mu_rkhs(1:k,2),'k--','linewidth',2.0,'DisplayName','FPF - RKHS');
        hold on;
    end
    if zero_mean == 1
        plot(0:delta:(k-1)*delta, mu_zm(1:k,2),'b--','linewidth',2.0,'DisplayName','FPF - RKHS-OM)');
        hold on;
    end
    if zm_mem == 1
        plot(0:delta:(k-1)*delta, mu_zm_mem(1:k,2),'r--','linewidth',2.0,'DisplayName','FPF - RKHS-OM + memory');
        hold on;
    end
    
    if const == 1
        plot(0:delta:(k-1)*delta, mu_const(1:k,2),'c--','linewidth',2.0,'DisplayName','FPF - Const gain');
        hold on;
    end
    if kalman == 1
        plot(0:delta:(k-1)*delta, X_kal(1:k,2),'g--','linewidth',2.0,'DisplayName','EKF');
        hold on;
    end
    if sis == 1
        plot(0:delta:(k-1)*delta, mu_sis(1:k,2),'m--','linewidth',2.0,'DisplayName','SIS PF');
        hold on;
    end
    legend('show');
    
    figure;
    plot(0:delta:(k-1)*delta, Z(1:k),'k','DisplayName','Observations');
    hold on;
    plot(0:delta:(k-1)*delta, Z_true(1:k),'r','DisplayName','True observations');   
    title('Z_t');
    legend('show');
end

 if diag_main == 1
   if sis == 1
        figure;
        plot(0:delta:(k-1)*delta, N_eff_sis(1:k),'r');
        hold on;
        title('Effective particle size N_{eff} in SIS PF');
   end    
 end
 
end


% Overall rmse 
if coif == 1
    rmse_tot_coif = mean(rmse_coif);
    sprintf('---------------- Coifman ---------------')
    sprintf('RMSE for Coifman method - %0.5g', rmse_tot_coif)
    sprintf(['Lost track ' num2str(lose_coif) ' times'])
    sprintf(['Maximum deviation from true state - ' num2str(max(max_diff_coif))])
end
if rkhs == 1
    rmse_tot_rkhs = mean(rmse_rkhs, 'omitnan' );
    sprintf('---------------- RKHS ---------------')
    sprintf('RMSE for RKHS method - %0.5g', rmse_tot_rkhs)
    sprintf(['Lost track ' num2str(lose_rkhs) ' times'])
    sprintf(['Maximum deviation from true state - ' num2str(max(max_diff_rkhs))])
end
if zero_mean == 1
    rmse_tot_zm = mean(rmse_zm, 'omitnan' );
    sprintf('---------------- RKHS (ZM) ---------------')
    sprintf('RMSE for RKHS (ZM) method - %0.5g', rmse_tot_zm)
    sprintf(['Lost track ' num2str(lose_zm) ' times'])
    sprintf(['Maximum deviation from true state - ' num2str(max(max_diff_zm))])
end
if zm_mem == 1
    rmse_tot_zm_mem = mean(rmse_zm_mem, 'omitnan' );
    sprintf('---------------- RKHS (ZM) ---------------')
    sprintf('RMSE for RKHS ZM + memory method - %0.5g', rmse_tot_zm_mem)
    sprintf(['Lost track ' num2str(lose_zm_mem) ' times'])
    sprintf(['Maximum deviation from true state - ' num2str(max(max_diff_zm_mem))])
end
if const == 1
    rmse_tot_const = mean(rmse_const , 'omitnan');
    sprintf('---------------- Const gain ---------------')
    sprintf('RMSE for const gain approximation - %0.5g', rmse_tot_const)
    sprintf(['Lost track ' num2str(lose_const) ' times'])
    sprintf(['Maximum deviation from true state - ' num2str(max(max_diff_const))])
end
if sis == 1
    rmse_tot_sis = mean(rmse_sis, 'omitnan');
    sprintf('---------------- SIS PF ---------------')
    sprintf('RMSE for SIS PF - %0.5g', rmse_tot_sis)
    sprintf(['Lost track ' num2str(lose_sis) ' times'])
    sprintf(['Maximum deviation from true state - ' num2str(max(max_diff_sis))])
end
if kalman == 1
    rmse_tot_kal   = mean(rmse_kal, 'omitnan');
    sprintf('---------------- EKF ---------------')
    sprintf('RMSE for Kalman Filter - %0.5g', rmse_tot_kal)
    sprintf(['Lost track ' num2str(lose_kal) ' times'])
    sprintf(['Maximum deviation from true state - ' num2str(max(max_diff_kal))])
end

toc





