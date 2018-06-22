function [K] = gain_coif(Xi, c, epsilon, diag )
% Returns the gain computed at particle locations Xi using the Coifman
% kernel based method.
% tic;
N = length(Xi);
T = zeros(N);
max_diff = 1;
No_iterations = 50000;
iterations = 1;
Phi = zeros(N,1);
H = c(Xi);

for pi = 1:1:N    
    for pj = 1:1:N
        G(pi,pj) = exp( - (norm(Xi(pi)- Xi(pj)))^2 /  (4 * epsilon));     % Gaussian kernel for Coifman
    end
end
for pi = 1:1:N
    for pj = 1:1:N
        k(pi,pj) = G(pi,pj)./( sqrt( (1/N) * sum(G(pi,:))) * sqrt( (1/N) * sum(G(pj,:)))); % Coifman kernel
    end
    T(pi,:)  = k(pi,:)./sum(k(pi,:));                                       % Markov semigroup approximation
end
while (max_diff > 1e-3 && iterations < No_iterations)                        % Can adjust this exit criteria - (norm_diff > 1e-2 & iterations < 50000) 
    Phi(:,iterations + 1) = T * Phi(:,iterations) + epsilon * H';
    max_diff = max(Phi(:,iterations + 1) - Phi(:,iterations)) - min(Phi(:,iterations + 1) - Phi(:,iterations));
    iterations = iterations + 1;
end
for pi = 1:1:N
    sum_term(pi) = T(pi,:) * Xi';
    K(pi) = 0;
    for pj = 1:1:N
        K(pi)    = K(pi) + (1/(2 * epsilon)) * T(pi,pj) * Phi(pj,end) * (Xi(pj) - sum_term(pi));  % Gain computed for particle index pi
    end
end

% For comparison and trouble-shooting - Approximating the gradient
% Phi_delta = [ Phi(1,end); Phi(1:end-1,end)];
% Xi_delta  = [ Xi_f(k-1,1)-0.01 ; Xi_f(k-1,1:end-1)']; 
% grad_Phi_approx = (Phi(:,end) - Phi_delta)./(Xi_f(k-1,:)' - Xi_delta);
% toc

if diag == 1
    figure;
    plot(Xi,K,'b*');
    hold on;
end
end

