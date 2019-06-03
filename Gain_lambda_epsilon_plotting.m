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

display_figure = 1;
optimal  =0;
subspace =1;
zm       =1;

syms x;
% rng(400);      % Set a common seed
No_runs = 1;

%% FPF parameters
   N = 200;      % No of particles - Common for all Monte Carlo methods used

%% Hyper parameter choices
   % eps      = [0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];         % Variance parameter of the kernel  - Other tried values (0.25,0.1), For kernel = 0, range 0.1 - 0.25.
   % nlog     = 6:-0.5:0;
   % lambda   = 10.^(-nlog); %[1e-5 1e-4 1e-3 1e-2 1e-1];  
   eps = 0.25;
   lambda = 1e-2
   K_rkhs   = zeros(1,N);  % Initializing the gain to a 1 vector, this value is used only at k = 1. 
   
%% Parameters corresponding to observation process
c =  x;
c_x = matlabFunction(c);
sigmaW = 0.3;

%% Parameters of the prior p(0) - 2 component Gaussian mixture density 
m = 2;
sigma = [0.4 0.4]; 
mu    = [-1 1]; 
w     = [0.5 rand];  % Needs to add up to 1.
w(m)  = 1 - sum(w(1:m-1));

%% Gaussian mixture density
p = 0;
eta = 0;
% eta = mean(c_x(Xi));
step = 0.01;
xmax = max(mu) + 10;
if max(abs(mu)) > 100
   K = zeros(N,1);
   return;
end

for i = 1:1:length(mu)
    p   = p + w(i) * exp (- norm(x - mu(i))^2 / (2 * sigma(i)^2)) * (1 / sqrt( 2 * pi * sigma(i)^2)) * step;
end
p_x   = matlabFunction(p);

%%  100 independent trials
% Sampling particles from the prior density p(0)
for run = 1:No_runs
    run 
    
    gmobj = gmdistribution(mu',reshape(sigma.^2,1,1,m),w);
    Xi  = random(gmobj,N);
    Xi  = Xi';            % To be consistent with code below
    Xi  = sort(Xi);       % Sort the samples in ascending order for better visualization.

%% Exact gain computation  
    for i = 1 : N
        integral(i) = 0;
        for xj = Xi(i) : step : xmax + 10
            integral(i) = integral(i) + p_x(xj) * (c_x(xj) - eta) * step;
        end
        K(i) = integral(i) / p_x(Xi(i));
    end

%% Comparing the gain approximations using hyper parameter values
    hyp_i = 0;
    for eps_i = 1:length(eps)
        Ker         =  exp(-(ones(N,1) * Xi - Xi' * ones(1,N)).^2/(4 * eps(eps_i)));        
        Ker_x       = (ones(N,1) * Xi - Xi' * ones(1,N))/ (2 * eps(eps_i)) .* Ker;
        if optimal == 1
            Ker_x_y     = (ones(N,N) - (ones(N,1) * Xi - Xi' * ones(1,N)).^2 /(2 * eps(eps_i)))/(2 * eps(eps_i)) .* Ker;
        end
        Y      =  c_x(Xi) - eta;
        if zm == 1
            K_hat       =  mean((c_x(Xi) - eta) .* Xi);          % Constant gain approximation
            Ker_x_ones  =  (ones(1,N) * Ker_x');
            Ker_x_sum   =  Ker_x'* Ker_x;  
            M_m(1 : N, N + 1) =  (1/N) * Ker_x_ones;             % There was no (1/N) previously, testing it now
            M_m(N + 1, 1 : N) =  (1/N) * Ker_x_ones';
        end
    
        if optimal == 1
% Constructing block matrices for future use
           K_big      = [ Ker Ker_x' ; Ker_x Ker_x_y];
           K_thin_yxy = [ Ker_x' ; Ker_x_y]; 
           K_thin_x   = [ Ker ; Ker_x];
% b used in the extended representer theorem algorithm - searching over all of the Hilbert space H
           b_2m        =  (1/N) * K_thin_x * Y';
        end
        if subspace == 1
% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
            if zm == 1
                b_m    =  (1/N) * Ker * Y' -  (1/N) * Ker_x_ones' * K_hat; 
            else
                b_m    =  (1/N) * Ker * Y'; 
            end
        end
    
        for lambda_i = 1:length(lambda)
            hyp_i  =  hyp_i + 1; 
            hyp(hyp_i,:) = [eps(eps_i) lambda(lambda_i)];
            if optimal == 1
                M_2m = lambda(lambda_i) * K_big + (1/N) * K_thin_yxy * K_thin_yxy';
                beta_2m = M_2m \ b_2m;   
            end
            if subspace == 1
                M_m  = lambda(lambda_i) * Ker + (1/N) * Ker_x' * Ker_x;       % Ker_x * Ker_x' = Ker_x' * Ker_x - Hence either one works
                beta_m  = M_m \ b_m;
            end 
        
            for i = 1: 1 : N
                K_m(hyp_i,i)     = 0;
                K_2m(hyp_i,i)    = 0;
                for j = 1 : 1 : N
                    if optimal == 1
                        K_2m(hyp_i,i)     = K_2m(hyp_i,i)    + beta_2m(j) * Ker_x(i,j) + beta_2m(N+j) * Ker_x_y(i,j);
                    end
                    if subspace == 1
                        K_m(hyp_i,i)      = K_m(hyp_i,i)     + beta_m(j)  * Ker_x(i,j);      
                    end
                end
            end
            if subspace == 1
                mse_m(hyp_i,run) = (1/N) * sum((K_m(hyp_i,:) - K).^2);
            end
            if optimal == 1
                mse_2m(hyp_i,run) = (1/N) * sum((K_2m(hyp_i,:) - K).^2);
            end
        end
    end
end
toc;     

%% Computing the average mse over 100 runs
mse_m_avg  = mean(mse_m,2);
mse_2m_avg = mean(mse_2m,2);

[min_mse_m ind_m] = min(mse_m_avg);
[min_mse_2m ind_2m] = min(mse_2m_avg);

% display("Best hyperparameter");
display(['\epsilon =' num2str(hyp(ind_m,1))]);
display(['\lambda =' num2str(hyp(ind_m,2))]);



if display_figure == 1
    figure;
    plot(Xi,K,'r*');
    hold on;
    for i = 1: hyp_i
        plot(Xi,K_m(i,:),'b--');
        plot(Xi,K_2m(i,:),'k:');
    end
    
    figure;
    contourf(eps, log10(lambda), reshape(log(mse_m_avg),length(lambda),length(eps)));
    xlabel('\epsilon');
    ylabel('log(\lambda)');
    title('Contour plot of mse using the suboptimal solution (m) parameters');
    savefig('Figures/contour_mse_m.fig');
    
    
    figure;
    contourf(eps, log10(lambda), reshape(log(mse_2m_avg),length(lambda),length(eps)));
    xlabel('\epsilon');
    ylabel('log(\lambda)');
    title('Contour plot of mse using the optimal solution (2m) parameters');
    savefig('Figures/contour_mse_2m.fig');
    
    figure;
    surfc(eps, log10(lambda), reshape(mse_m_avg,length(lambda),length(eps)));
    savefig('Figures/surf_mse_m.fig');
    
    figure;
    surfc(eps, log10(lambda), reshape(mse_2m_avg,length(lambda),length(eps)));
    savefig('Figures/surf_mse_2m.fig');
end
    
  