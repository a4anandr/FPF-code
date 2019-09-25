function [eta K] = gain_const_multi( Xi, h, d, diag )
% Function that takes in the particle locations Xi and the observation
% function h and computes the constant gain approximation of the FPF gain.

N = length(Xi);

eta = 0;
for i = 1:N
    eta  = eta + (1/N) * h(Xi(i,:));
end

K   = zeros(1,d);
for i = 1: N
    K = K + (1/N) * ((h(Xi(i,:)) - eta) .* Xi(i,:));
end

if diag == 1
    figure(100);
    plot3(Xi(:,1),Xi(:,2),K(1) * ones(N,1),'bo');
    hold on;
    plot3(Xi(:,1),Xi(:,2),K(2) * ones(N,1),'ro');   
    % pause(2);
end  
end

