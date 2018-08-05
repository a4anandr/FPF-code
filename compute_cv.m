function [ cv ] = compute_cv(X , K, grad_U)
% This function accepts the approximation to the gradient of the solution to Poisson's
% equation and accepts the samples Xi and computes and returns the control variates
% corresponding to each of the samples 
N  = length(X);
[ X_sort ind_sort] = sort(X);       % Possible only in one dimension, Need to be careful to avoid this in 2 dimensions
K_sort = K(ind_sort);
[ind ind_unsort] = sort(ind_sort);

% Approximation of the derivative K' 
K_der(1) = 0;
for n = 2:N
    if (X_sort(n) ~= X_sort(n-1))
        K_der(n) = (K_sort(n) - K_sort(n-1))/(X_sort(n) - X_sort(n-1));
    else
        K_der(n) = K_der(n-1);
    end
end
K_der(1) = K_der(2);

% Computing control variates 

cv  = - grad_U(X) .* K + K_der(ind_unsort);

end

