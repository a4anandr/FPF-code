function [Phi K] = gain_coif(Xi, c, epsilon, Phi,N_ker, diag)
% Returns the gain computed at particle locations Xi using the Coifman
% kernel based method.
tic
N = length(Xi);
T = zeros(N_ker);
max_diff = 1;
No_iterations = 50000;
iterations = 1;
Phi = zeros(N_ker,1);
H = c(Xi);

for i = 1:1:N    
    for j = 1:1:N_ker
        g(i,j) = exp( - (norm(Xi(i)- Xi(j)))^2 /  (4 * epsilon));     % Gaussian kernel for Coifman
    end
end
for i = 1:1:N
    for j = 1:1:N_ker
        k(i,j) = g(i,j)./( sqrt( (1/N_ker) * sum(g(i,:))) * sqrt( (1/N_ker) * sum(g(j,:)))); % Coifman kernel
    end
    T(i,:)  = k(i,:)./sum(k(i,:));                                                   % Markov semigroup approximation
end
while (max_diff > 1e-3 && iterations < No_iterations)                                % Can adjust this exit criteria - (norm_diff > 1e-2 & iterations < 50000) 
    Phi(:,iterations + 1) = T(1:N_ker,:) * Phi(:,iterations) + epsilon * H(1:N_ker)';
    max_diff = max(Phi(:,iterations + 1) - Phi(:,iterations)) - min(Phi(:,iterations + 1) - Phi(:,iterations));
    iterations = iterations + 1;
end
for i = 1:1:N
    sum_term(i) = T(i,:) * Xi(1:N_ker)';
    K(i) = 0;
    for j = 1:1:N_ker
        K(i)    = K(i) + (1/(2 * epsilon)) * T(i,j) * Phi(j,end) * (Xi(j) - sum_term(i));  % Gain computed for particle index pi
    end
end

% For comparison and trouble-shooting - Approximating the gradient
% Phi_delta = [ Phi(1,end); Phi(1:end-1,end)];
% Xi_delta  = [ Xi_f(k-1,1)-0.01 ; Xi_f(k-1,1:end-1)']; 
% grad_Phi_approx = (Phi(:,end) - Phi_delta)./(Xi_f(k-1,:)' - Xi_delta);

toc

%% For displaying figures
if diag == 1
    figure;
    plot(Xi,K,'b*');
    hold on;
end
end

