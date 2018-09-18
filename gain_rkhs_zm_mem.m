function [eta K] = gain_rkhs_zm_mem( Xi , h , d, kernel, lambda, epsilon, alpha, K_prev, diag)
% Returns the gain computed at particle locations Xi using an RKHS 
N = length(Xi);
v = version('-release');

% Evaluation of kernel matrices 
eta = 0;
if kernel == 0  % Gaussian
    if v == '2018a'
        for i = 1:N
            eta  = eta + (1/N) * h(Xi(i,:));
        end
        for k = 1:N
            Ker(:,k) = exp(- vecnorm(Xi - repmat(Xi(k,:),N,1),2,2).^2/( 4 * epsilon));
            for d_i = 1 : d
                Ker_x(:,k,d_i) =  -((Xi(:,d_i) - Xi(k,d_i))./ (2 * epsilon)) .* Ker(:,k);   % Derivative of Ker with respect to the first variable
            end
        end
    else
        for i = 1:N
            eta  = eta + (1/N) * h(Xi(i,:));
            for k = 1:N          
                Ker(i,k) =  exp(-(norm(Xi(i,:) - Xi(k,:)).^2/(4 * epsilon)));  
                for d_i = 1 : d
                    Ker_x(i,k,d_i) =  -((Xi(i,d_i) - Xi(k,d_i)) / (2 * epsilon)) .* Ker(i,k);   % Derivative of Ker with respect to the first variable
               % If we fix the second variable and run through all values
               % of i, then we get the derivative of Ker with respect to
               % the first variable.
                end
            end
        end
    end
end

% Computing Y vector and the constant gain approximation K^
K_hat   = zeros(1,d);
for i = 1:N
    Y(i)  = h(Xi(i,:)) - eta;
    K_hat = K_hat + (1/N) * ((h(Xi(i,:)) - eta) .* Xi(i,:));      % Constant gain approximation
end

% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
Ker_x_sum = zeros(N,N);
for d_i = 1 : d
    Ker_x_ones(:,d_i) =  Ker_x(:,:,d_i) * ones(N,1);
    Ker_x_sum         =  Ker_x_sum + Ker_x(:,:,d_i)'* Ker_x(:,:,d_i);
    M(1 : N, N + d_i) =  (1/N) * Ker_x_ones(:, d_i);              
    M(N + d_i, 1 : N) =  (1/N) * Ker_x_ones(:, d_i)';
end
b(1 : N, :)        =  (2/N) * Ker * Y'- (2/N) * ( Ker_x_ones * K_hat' + Ker_x' * K_prev'); %  Regularization in \clH norm with \tilg gives this  
% b(1 : N, :)        =  (2/N) * Ker * Y'- (2/N) * ( Ker_x_ones * K_hat' - alpha * Ker_x' * K_prev'); % Regularization in \clH norm with just g gives this
b(N+1 : N+d, :)    =  zeros(d,1);    
M(1 : N, 1 : N)    =  2 * lambda * Ker + ( 2 * (1 + alpha) / N) * Ker_x_sum;       % Ker_x * Ker_x' = Ker_x' * Ker_x - Hence either one works
beta               =  M \ b;
       
%  for i = 1: 1 : N
%      K(i,:)     = K_hat;
%      for k = 1 : 1 : N
%          for d_i = 1 : d
%              K(i,d_i)      = K(i,d_i)     + beta(k)  * Ker_x(i,k,d_i);      % Ker_x(pj,pi)
%          end   
%      end
%  end

for d_i = 1 : d
    K(:,d_i)      = beta(1:N)'  * Ker_x(:,:,d_i)';     
end 
K = repmat(K_hat,N,1)  + K + K_prev';


end


