%% To compute rmse for various lambda_1 values for the multidimensional example
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
No_runs = 100;   % Total number of runs to compute the rmse metric for each of the filters for comparison


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
Sig  = [5 0; 0 5];

%% Filter parameters
N = 500;          % No of particles - Common for all Monte Carlo methods used

% Setting a max and min threshold for gain
K_max = 100;
K_min = -100;
 
%% RKHS ZM with memory
kernel          = 0;          % 0 for Gaussian kernel, 1 for Coifman kernel, 2 for approximate Coifman kernel
lambda          = 1e-1;       % 0.05, 0.02, Regularization parameter - Other tried values ( 0.005,0.001,0.05), For kernel = 0, range 0.005 - 0.01.
eps             = 2;          % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
lambda_gain     = [0 1e-2 1e-1 1 10];          % 1e-4; % This parameter decides how much the gain can change in successive time instants, higher value implying less variation. 
K               = zeros(1,N,d); % Initializing the gain to a 1 vector, this value is used only at k = 1. 
beta            = zeros(N + d,length(lambda_gain)); % Initializing the parameter values beta to zero.

lose_track      = zeros(length(lambda_gain),1);         % Initializing the "lose the track" count as specified in Budhiraja et al.

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
    for lambda_i = 1 : 1 : length(lambda_gain)
        Xi(:,:,1,lambda_i)  = Xi_0;
        first(lambda_i)     = 0;
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
     
    %% Filter
        common_rand = randn(2,N);
        for lambda_i = 1 : 1: length(lambda_gain)
            if k == 2
               [h_hat(k-1), beta(:,lambda_i), K(k,:,:)] = gain_rkhs_zm_mem(Xi(:,:,k-1,lambda_i) , h_x, d , kernel,lambda, eps, 0, K(k-1,:,:), beta(:,lambda_i), zeros(N,d), diag_fn);
            else
               [h_hat(k-1), beta(:,lambda_i), K(k,:,:)] = gain_rkhs_zm_mem(Xi(:,:,k-1,lambda_i) , h_x, d , kernel,lambda, eps, lambda_gain(lambda_i), K(k-1,:,:), beta(:,lambda_i), Xi(:,:,k-2,lambda_i),diag_fn);
            end
        
            mu(k-1,:,lambda_i)   = mean(Xi(:,:,k-1,lambda_i));
        
            if (norm(mu(k-1,:,lambda_i) - X(k-1,:)) > tol && first(lambda_i) == 0)
                lose_track(lambda_i)  = lose_track(lambda_i) + 1;
                first(lambda_i) = first(lambda_i) + 1;
            end
    
            for i = 1:N
                dI(k)     = Z(k-1) - 0.5 * (h_x(Xi(i,:,k-1,lambda_i)) + h_hat(k-1));
                K(k,i,1)  = min(max(K(k,i,1),K_min),K_max);
                K(k,i,2)  = min(max(K(k,i,2),K_min),K_max);
                Xi(i,1,k,lambda_i) = Xi(i,1,k-1,lambda_i) - Xi(i,2,k-1,lambda_i) * delta + f1_x(Xi(i,:,k-1,lambda_i)) * delta + e1 * sdt * common_rand(1,i) + (K(k,i,1)/R) * dI(k);       % K_rkhs(k,i,1) * dI_rkhs(k)
                Xi(i,2,k,lambda_i) = Xi(i,2,k-1,lambda_i) + Xi(i,1,k-1,lambda_i) * delta + f2_x(Xi(i,:,k-1,lambda_i)) * delta + e2 * sdt * common_rand(2,i) + (K(k,i,2)/R) * dI(k);
            end
        end
    end
%% Computing the rmse metric and the max error metric (to count losing the track)
    for lambda_i = 1 : 1 : length(lambda_gain)
        mu(k,:,lambda_i)     = mean(Xi(:,:,k,lambda_i));
        rmse(lambda_i, run)  = mean(vecnorm(X - mu(:,:,lambda_i),2,2));
        max_diff_zm_mem(run) = max(vecnorm(X - mu(:,:,lambda_i),2,2)); 
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
        plot(mu(1:k,1,lambda_i), mu(1:k,2,lambda_i),'b-x','linewidth',2.0,'DisplayName',['\lambda_1 = ' num2str(lambda_gain(lambda_i))]);   
        legend('show');
   
        figure;
        plot(0:delta:(k-1)*delta, X(1:k,1),'k','DisplayName','True state');
        hold on;
        plot(0:delta:(k-1)*delta, mu(1:k,1,lambda_i),'b--','linewidth',2.0,'DisplayName',['\lambda_1 = ' num2str(lambda_gain(lambda_i))]);
        legend('show');
    
        figure;
        plot(0:delta:(k-1)*delta, X(1:k,1),'k','DisplayName','True state');
        hold on;
        plot(0:delta:(k-1)*delta, mu(1:k,2,lambda_i),'b--','linewidth',2.0,'DisplayName',['\lambda_1 = ' num2str(lambda_gain(lambda_i))]);
        hold on;
        legend('show');
  
        end
end
    
% Overall rmse for each \lambda_1
for lambda_i = 1 : 1: length(lambda_gain)
    rmse_tot(lambda_i) = mean(rmse(lambda_i,:), 'omitnan' );
    sprintf('---------------- RKHS (ZM) ---------------')
    sprintf('RMSE for RKHS ZM + memory method - %0.5g', rmse_tot(lambda_i))
    sprintf(['Lost track ' num2str(lose_track(lambda_i)) ' times'])
end
toc








