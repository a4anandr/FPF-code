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
% functions - i) const gain approx FPF ii) SIS PF
% based.

clear;
clc;
close all;
format short g;
tic
warning off;

diag_main = 1;   % Diagnostics flag for main function, displays figures in main.
diag_output = 1; % Diagnostics flag to display the main output in this function
% rng(1000);     % Set a common seed
No_runs = 1;     % Total number of runs to compute the rmse metric for each of the filters for comparison

%% SPM parameter initialization
% C_rate_profile = csvread('t_vs_load_C_rate_constant.csv',1,0); % FOR EKF, discharge current is positive [Amps] load current flowing through external circuit. For LIONSIMBA,discharge current is negative
C_rate_profile = csvread('t_vs_load_C_rate_udds.csv',0,0); % FOR EKF, discharge current is positive [Amps] load current flowing through external circuit. For LIONSIMBA,discharge current is negative
I_1C         = 60; % Amps (1C current of the cell, i.e. cell capacity in disguise). Used only for desktop-simulation purposes.
soc_init_pct = 100; % (percentage) Starting SoC
param_spm = Parameters_spm(soc_init_pct);

%% Flags to be set to choose which methods to compare
const = 1;           % Computes the constant gain approximation
sis   = 1;          % Runs Sequential Importance Sampling Particle Filter 

%% Parameters corresponding to the state and observation processes
% Run time parameters
T     = 1300;  % Total running time, using the same value in references [1],[2]
delta = 1;     % Time increments for the SDE and observation model 
sdt   = sqrt(delta); 

d     = 3;        % State space dimension

% Process noise parameters
e1 = 0;
 
% Observation process parameters
theta = 1e-2;                % Standard deviation parameter in observation process

%% Parameters of the prior p(0) - Multivariate Gaussian density 
X_0 = [0 0 param_spm.cs_p_init];    % initial value of the state vector
Sig = 10000 * eye(3);               % initial state covariance ( 1e-2 (mol/m^3)^2)

%% Filter parameters
N = 500;       % No of particles - Common for all Monte Carlo methods used

% i) SIS PF
if sis == 1 
    resampling = 0;        % Whether you need deterministic resampling 
end

for run = 1: 1 : No_runs
    run
    
%% State and observation process initialization
    X(1,:)   = X_0;
    I_load   = interp1(C_rate_profile(:,1),C_rate_profile(:,2),0,'previous','extrap')*I_1C;
    Z(1)     = spm_battery_voltage(X(1,:),I_load,param_spm) + theta * randn;
    Z_true(1)= spm_battery_voltage(X(1,:),I_load,param_spm);

%% Initializing N particles from the prior
    Xi_0  = mvnrnd(X_0,Sig,N);   
    mui_0 = mean(Xi_0);
% i) FPF - const gain approx    
    if const == 1
       Xi_const     = Xi_0;
       for i = 1:N
           Zi_const(i,:)  = spm_battery_voltage(Xi_const(i,:),I_load,param_spm);
       end
       h_hat_const = mean(Zi_const);
    end
    
% ii) Sequential Importance Sampling Particle Filter Initialization
    if sis == 1
       Xi_sis       = Xi_0;
       Wi_sis(:,1)  = (1/N) * ones(1,N);          % Initializing all weights to equal value.
       for i = 1:N
           Zi_sis(i)  = spm_battery_voltage(Xi_sis(i,:,1),I_load,param_spm);
       end
    end

for k = 2: 1: (T/delta)    
    k  
    %% Actual state - observation process evolution
    t_span           = (k-2)*delta : delta : (k-1)*delta;
    I_load           =  interp1(C_rate_profile(:,1),C_rate_profile(:,2),(k-2)*delta,'previous','extrap')*I_1C;
    [~,X_t_matrix]   =  ode45(@(t,x)normalised_spm_state_eqn(X(k-1,:),I_load,param_spm), t_span, X(k-1,:));
    X(k,:)           =  X_t_matrix(end,:) + e1 * sdt * randn;   % need to check dimensions
    Z(k)             =  spm_battery_voltage(X(k,:),I_load,param_spm)  + theta * randn; 
    Z_true(k)        =  spm_battery_voltage(X(k,:),I_load,param_spm);
     
    %% Filters
% i) FPF - const gain approx
    if const == 1
        mu_const(k-1,:)   = mean(Xi_const(:,:,k-1));
        h_hat_const(k-1)  = mean(Zi_const(:,k-1));
        K_const(k,:) = zeros(1,d);
        for i = 1: N
            K_const(k,:) = K_const(k,:) + (1/N) * ((spm_battery_voltage(Xi_const(i,:,k-1),I_load,param_spm) - h_hat_const(k-1)) .* Xi_const(i,:,k-1));
        end
    end
% ii) SIS - PF    
    if sis == 1
        mu_sis(k-1,:)     = Wi_sis(:,k-1)' * Xi_sis(:,:,k-1);
        N_eff_sis(k-1)    = 1 / sum(Wi_sis(:,k-1).^2);
    end
        
    for i = 1:N
       common_rand = randn(1,d);
       
       % i) FPF - const gain approx 
       if const == 1
           dI_const(k)     = Z(k-1) - 0.5 * ( Zi_const(i,k-1) + h_hat_const(k-1));          
           [~,X_t_matrix]  = ode45(@(t,x)normalised_spm_state_eqn(Xi_sis(i,:,k-1),I_load,param_spm), t_span, Xi_sis(i,:,k-1));
           Xi_const(i,:,k) = X_t_matrix(end,:) +  e1 * sdt * common_rand + K_const(k,:) * dI_const(k);   % need to check dimensions
           Zi_const(i,k)   = spm_battery_voltage(Xi_const(i,:,k),I_load,param_spm);
       end
       
       % iv) SIS - PF
       if sis == 1
          [~,X_t_matrix]  = ode45(@(t,x)normalised_spm_state_eqn(Xi_sis(i,:,k-1),I_load,param_spm), t_span, Xi_sis(i,:,k-1));
          Xi_sis(i,:,k)   = X_t_matrix(end,:) + e1 * sdt * common_rand;   % need to check dimensions
          Zi_sis(i,k)     = spm_battery_voltage(Xi_sis(i,:,k),I_load,param_spm); 
          Wi_sis(i,k)     = Wi_sis(i,k-1) * (1/sqrt( 2 * pi * theta^2 * delta)) * exp ( - (Z(k) - Zi_sis(i,k)).^2/ (2 * theta^2 * delta));   %  Based on eqn(63) of Arulampalam et al. In our example, the importance density is the prior density p(X_t | X_{t-1}). If resampling is done at every step then the recursive form disappears. Wi_sis(k) does not depend on Wi_sis(k-1) as Wi_sis(k-1) = 1/N.
       end   
    end
    
 if sis == 1
 % Normalizing the weights of the SIS - PF
   Wi_sis(:,k)  = Wi_sis(:,k)/ sum(Wi_sis(:,k));
   if resampling == 1
 % Deterministic resampling - as given in Budhiraja et al.
        if mod(k,3)== 0
            sum_N_eff = 0;
            Wi_cdf    = zeros(N,1);
            for i = 1 : N
                N_eff(i) = floor(Wi_sis(i,k) *  N); 
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
            N_eff_sis(k) = 1 / (sum(Wi_sis(:,k).^2)); 
        end
   end
   N_eff_sis(k) = 1 / (sum(Wi_sis(:,k).^2)); 
 end
end    

%% Computing the rmse metric

if const == 1
    mu_const(k,:)   = mean(Xi_const(:,:,k));
    rmse_const(run) = (1 / (T/delta)) * sum(sqrt(sum((X - mu_const).^2,2)));
end
if sis == 1
    mu_sis(k,:)     = Wi_sis(:,k)' * Xi_sis(:,:,k);
    rmse_sis(run)   = (1 / (T/delta)) * sum(sqrt(sum ( (X - mu_sis).^2,2)));
end

%% Plotting the state trajectory and estimates
if (diag_output == 1 && No_runs == 1)   
    figure;
    plot(0:delta:(k-1)*delta, X(1:k,3),'DisplayName','True state');
    hold on;
    if const == 1
        plot(0:delta:(k-1)*delta, mu_const(1:k,3),'b--','linewidth',2.0,'DisplayName','FPF - Const gain');
        hold on;
    end
    if sis == 1
        plot(0:delta:(k-1)*delta, mu_sis(1:k,3),'m--','linewidth',2.0,'DisplayName','SIS PF');
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
if const == 1
    rmse_tot_const = mean(rmse_const , 'omitnan');
    sprintf('RMSE for const gain approximation - %0.5g', rmse_tot_const)
end
if sis == 1
    rmse_tot_sis = mean(rmse_sis, 'omitnan');
    sprintf('RMSE for SIS PF - %0.5g', rmse_tot_sis)
end
toc





