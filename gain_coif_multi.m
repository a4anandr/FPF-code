function [eta K] = gain_coif_multi(Xi, h, d, epsilon, diag )
% Returns the gain computed at particle locations Xi using the Coifman
% kernel based method.
% tic;
N = length(Xi);
T = zeros(N);
max_diff = 1;
No_iterations = 50000;
iterations = 1;
Phi = zeros(N,1);

for i = 1:N
    H(i) = h(Xi(i,:));
end
eta = mean(H);

for i = 1:1:N    
    for k = 1:1:N
        G(i,k) = exp( - (norm(Xi(i,:)- Xi(k,:))).^2 /  (4 * epsilon));     % Gaussian kernel for Coifman
    end
end
for i = 1:1:N
    for k = 1:1:N
        Ker(i,k) = G(i,k)./( sqrt( (1/N) * sum(G(i,:))) * sqrt( (1/N) * sum(G(k,:)))); % Coifman kernel
    end
    T(i,:)  = Ker(i,:)./sum(Ker(i,:));                                     % Markov semigroup approximation
end
while (max_diff > 1e-3 && iterations < No_iterations)                      % Can adjust this exit criteria - (norm_diff > 1e-2 & iterations < 50000) 
    Phi(:,iterations + 1) = T * Phi(:,iterations) + epsilon * H';
    max_diff = max(Phi(:,iterations + 1) - Phi(:,iterations)) - min(Phi(:,iterations + 1) - Phi(:,iterations));
    iterations = iterations + 1;
end
for i = 1:1:N
    sum_term(i,:) = T(i,:) * Xi;
    K(i,:) = zeros(1,d);
    for k = 1:1:N
        K(i,:)    = K(i,:) + (1/(2 * epsilon)) * T(i,k) * Phi(k,end) * (Xi(k,:) - sum_term(i,:));  % Gain computed for particle index pi
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

