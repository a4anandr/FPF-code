function [ K ] = gain_exact( Xi, c, mu, sigma, w, diag )
% Computes the exact gain by solving the one dimensional Poisson's equation
tic;

syms x
N = length(Xi);
p = 0;
eta = 0;
% eta = mean(c(Xi));
step = 0.01;
xmax = max(mu) + 10;

for i = 1:1:length(mu)
    p   = p + w(i) * exp (- norm(x - mu(i))^2 / (2 * sigma(i)^2)) * (1 / sqrt( 2 * pi * sigma(i)^2)) * step;
    eta = eta + w(i) * mu(i);
end
p_x   = matlabFunction(p);

for i = 1 : N
    integral(i) = 0;
    for xj = Xi(i) : step : xmax + 10
        integral(i) = integral(i) + p_x(xj) * (c(xj) - eta) * step;
    end
    K(i) = integral(i) / p_x(Xi(i));
end

toc 
%% For displaying figures
if diag == 1
    figure;
    plot(Xi,K,'ro');
    hold on;
    
    range = min(mu)- 3 * max(sigma): 0.05 : max(mu) + 3 * max(sigma);
    figure;
    plot(range, p_x(range),'r');
    hold on;
end

end

