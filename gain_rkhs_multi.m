function [eta K] = gain_rkhs_multi( Xi , h , d, kernel, lambda, epsilon, diag)
% Returns the gain computed at particle locations Xi using an RKHS 
% tic;
N = length(Xi);

% Evaluation of kernel matrices 
eta = 0;
if kernel == 0  % Gaussian
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

for i = 1:N
    Y(i)          = h(Xi(i,:)) - eta;
end

% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
Ker_x_sum = zeros(N,N);
for d_i = 1 : d
    Ker_x_sum         =  Ker_x_sum + Ker_x(:,:,d_i)'* Ker_x(:,:,d_i);
end
b     = (1/N) * ( Ker * Y'); 
M     = lambda * Ker + (1 / N) * Ker_x_sum;       % Ker_x * Ker_x' = Ker_x' * Ker_x - Hence either one works
beta  = M \ b;

% for i = 1: 1 : N
%      K(i,:)     = zeros(1,d);
%      for k = 1 : 1 : N
%          for d_i = 1 : d
%              K(i,d_i)      = K(i,d_i)     + beta(k)  * Ker_x(i,k,d_i);      % Here, we are running through the second index k for a fixed i and so it is the derivative with respect to the second variable.
%          end   
%      end
% end

for d_i = 1 : d
    K(:,d_i)      = beta(1:N)'  * Ker_x(:,:,d_i)';     
end 


if diag == 1
    figure(100);
    clf;
    plot3(Xi(:,1),Xi(:,2),K(:,1),'b*');
    hold on;
    plot3(Xi(:,1),Xi(:,2),K(:,2),'r^');   
end  
end

