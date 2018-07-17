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
format short g;
tic

diag_main = 1;   % Diagnostics flag for main function, displays figures in main.
diag_output = 1;
diag_fn = 0;     % Diagnostics flag, if 1, then all the functions display plots for diagnostics, Set it to 0 to avoid plots from within the calling functions
% rng(3300);        % Set a common seed
No_runs = 1;   % Total number of runs to compute the rmse metric for each of the filters for comparison

C_rate_profile = csvread('t_vs_load_C_rate_constant.csv',1,0); % FOR EKF, discharge current is positive [Amps] load current flowing through external circuit. For LIONSIMBA,discharge current is negative
I_1C         = 60; % Amps (1C current of the cell, i.e. cell capacity in disguise). Used only for desktop-simulation purposes.
soc_init_pct = 100; % (percentage) Starting SoC

param_spm = Parameters_spm(soc_init_pct);

%% Flags to be set to choose which methods to compare

exact = 0;           % Computes the exact gain and plots 
fin   = 0;           % Computes gain using finite dimensional basis
coif  = 0;           % Computes gain using Coifman kernel method
rkhs  = 1;           % Computes gain using RKHS
const = 1;           % Computes the constant gain approximation
kalman = 1;          % Runs Kalman Filter for comparison
sis    = 1;          % Runs Sequential Importance Sampling Particle Filter


%% FPF parameters

   N = 500;          % No of particles - Common for all Monte Carlo methods used
   
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
   lambda   = 0.1;        % 0.05, 0.02, Regularization parameter - Other tried values ( 0.005,0.001,0.05), For kernel = 0, range 0.005 - 0.01.
   eps_rkhs = 0.1;         % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
   lambda_gain = 0; %1e-4;        % This parameter decides how much the gain can change in successive time instants, higher value implying less variation. 
   K_rkhs   = ones(1,N);   % Initializing the gain to a 1 vector, this value is used only at k = 1. 
end

% Setting a max and min threshold for gain
K_max = 100;
K_min = -100;

%% Parameters corresponding to the state and observation processes
% Run time parameters
T   = 1;         % Total running time - Using same values as in Amir's CDC paper - 0.8
dt  = 0.01;        % Time increments for the SDE

sigmaB = 0;             % 0 if no noise in state process
I_load_init = interp1(C_rate_profile(:,1),C_rate_profile(:,2),0,'previous','extrap')*I_1C;

sigmaW = 0.3;


% Parameters of p(0) - 2 component Gaussian mixture density - Prior needs
% to be initialized with the range of concentration values
m = 2; 
sigma = [0.1 0.1]; 
mu    = [-1 1]; 
w     = [0.5 rand]; % Needs to add up to 1.
w(m)  = 1 - sum(w(1:m-1));
% Constructing a 3 component Gaussian mixture for EM to make sure gain does not blow up.
mu_em = [0 mu];
sigma_em = [0 sigma];
w_em = [0 w];

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
Xi_const(1,:)= Xi_0;


%  Sequential Importance Sampling Particle Filter Initialization
Xi_sis(1,:)  = Xi_0;
Wi_sis(1,:)  = (1/N) * ones(1,N);
Zi_sis(1,:)  = spm_battery_voltage(Xi_sis(1,:),I_load_init,param_spm);

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
% Initialization
X(1)   = param_spm.cs_p_init;  % + Gaussian 
Z(1)   = spm_battery_voltage(X(1),I_load_init,param_spm) + sigmaW * sdt * randn;

for k = 2: 1: (T/dt)

    k
    I_load = interp1(C_rate_profile(:,1),C_rate_profile(:,2),k*dt,'previous','extrap')*I_1C;

    X(k) = X(k-1) +   a_spm(param_spm,X(k-1),I_load) * dt + sigmaB * sdt * randn;
    Z(k) = Z(k-1) +   spm_battery_voltage(X(k),I_load,param_spm)  * dt + sigmaW * sdt * randn; 
    
    if k == 2 
        dZ(k) = Z(k) - Z(k-1); 
    else
        dZ(k) = 0.5 * (Z(k) - Z(k-2));
    end
    
    if const == 1
        mu_const(k-1)     = mean(Xi_const(k-1,:));
        c_hat_const(k-1)  = mean(spm_battery_voltage(Xi_const(k-1,:),I_load,param_spm)); 
        K_const(k)        = mean((spm_battery_voltage(Xi_const(k-1,:),I_load,param_spm) - c_hat_const(k-1)) .* Xi_const(k-1,:));
    end
    
    if sis == 1
        mu_sis(k-1)       = Wi_sis(k-1,:)* Xi_sis(k-1,:)';
        N_eff_sis(k-1)    =  1 / sum(Wi_sis(k-1,:).^2);
    end
        
    for i = 1:N
       
       % v) Constant gain approximation 
       if const == 1
           dI_const(k)       = dZ(k) - 0.5 * (spm_battery_voltage(Xi_const(k-1,i),I_load,param_spm) + c_hat_const(k-1)) * dt;          
           Xi_const(k,i)     = Xi_const(k-1,i) + a_x(Xi_const(k-1,i)) * dt + sigmaB * sdt * common_rand + (1 / R) * K_const(k) * dI_const(k);
       end
       
       % vi) Sequential Importance Sampling Particle Filter (SIS PF)
       if sis == 1 
          Xi_sis(k,i)       = Xi_sis(k-1,i) + a_x(Xi_sis(k-1,i)) * dt + sigmaB * sdt * common_rand; 
          Zi_sis(k,i)       = Zi_sis(k-1,i) + spm_battery_voltage(Xi_sis(k,i),I_load,param_spm)   * dt;  
          Wi_sis(k,i)       = Wi_sis(k-1,i) * (1/sqrt( 2 * pi * R * dt)) * exp ( - (Z(k) - Zi_sis(k,i))^2/ (2 * R * dt));   %  Based on eqn(63) of Arulampalam et al. In our example, the importance density is the prior density p(X_t | X_{t-1}). If resampling is done at every step then the recursive form disappears. Wi_sis(k) does not depend on Wi_sis(k-1) as Wi_sis(k-1) = 1/N.
       end
              
    end
    
 if sis == 1
 % Normalizing the weights of the SIS - PF
     Wi_sis(k,:)  = Wi_sis(k,:)/ sum(Wi_sis(k,:));
     N_eff_sis(k) = 1 / (sum(Wi_sis(k,:).^2)); 
 end
    
 % vii) Extended Kalman Filter for comparison
 if kalman == 1
      X_kal(k)= X_kal(k-1) + a_x(X_kal(k-1)) * dt + K_kal(k-1) * (dZ(k-1) - spm_battery_voltage(X_kal(k-1),I_load,param_spm) * dt);  % Kalman Filtered state estimate   
      P(k)    = P(k-1)+ 2 * a_der_x(X_kal(k-1)) * P(k-1) * dt+ Q * dt - (K_kal(k-1)^2) * R * dt;     % Evolution of covariance
      K_kal(k)= (P(k)* c_der_x(X_kal(k-1)))/R;                                                % Computation of Kalman Gain     
 end

%% Displaying figures for diagnostics 

    % Plotting gains at k = 2, 3, 10, 20, 30, 40
    if ( diag_main == 1 && ( k == 2 | k == 3 | k == 11 || k == 21 || k == 31 || k == (T/dt))) 
        figure;
        if exact == 1
            plot(Xi_exact(k-1,:), K_exact(k,:), 'rv','DisplayName','Exact'); 
            hold on;
        end
        if fin == 1
            plot(Xi_fin(k-1,:), K_fin(k,:), 'g*', 'DisplayName','Finite');  
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
        if const ==1
            plot(Xi_const(k-1,:),K_const(k) * ones(1,N),'mv','DisplayName','Const');
            hold on;
        end
        title(['Gain at particle locations for ' num2str(N) ' particles at t = ' num2str((k-2) * dt)]);
        legend('show');
    end
    
    if ( k == 2 || k == 0.5*(T/dt) || k == (T/dt))
        step = 0.05;
        range = min(mu_em)- 3 * max(sigma_em): step : max(mu_em) + 3 * max(sigma_em);
        figure(100);
        if exact == 1
          p_t = 0;         
          for i = 1: length(mu_em)
              p_t = p_t + w_em(i) * exp(-( range - mu_em(i)).^2 / (2 * sigma_em(i)^2)) * (1 / sqrt(2 * pi * sigma_em(i)^2));
          end
          plot(range, p_t,'DisplayName',['Using exact p_t at t = ' num2str( (k-1)*dt )]);
          hold on;
       end
       if kalman == 1
           p_kal_t = exp(-( range - X_kal(k)).^2 / (2 * P(k))) * (1 / sqrt(2 * pi * P(k)));
           plot(range, p_kal_t,'DisplayName',['Using Kalman p_t at t = ' num2str( (k-1)*dt )]);
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
            if const == 1
               hist(Xi_const(k-1,:),N);
            end             
       else
            if fin == 1
             % CAUTION : histogram command works only in recent Matlab
             % versions, if it does not work, comment this section out
               histogram(Xi_fin(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName',['Hist using finite at t =' num2str( (k-1)*dt )]);
               hold on;
            end
            if coif == 1
               histogram(Xi_coif(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName',['Hist using Coifman at t =' num2str( (k-1)*dt )]);
               hold on;
            end
            if rkhs == 1
               histogram(Xi_rkhs(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName',['Hist using RKHS at t =' num2str( (k-1)*dt )]);
               hold on;
            end
            if const == 1
               histogram(Xi_const(k-1,:),'Normalization','pdf','DisplayStyle','stairs','BinWidth',step,'BinLimits',[ min(mu_em) - 3 * max(sigma_em), max(mu_em) + 3 * max(sigma_em)],'DisplayName',['Hist using const at t =' num2str( (k-1)*dt )]);
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
           plot(range, p_sis_t,'DisplayName',['Using SIS PF p_t at t = ' num2str( (k-1)*dt )]);
           hold on;
       end
       legend('show');
       title('Posterior density p_t - Smoothed and Histogram');
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

if diag_main == 1
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

toc





