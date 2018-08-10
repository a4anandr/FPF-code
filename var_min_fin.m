function [K] = var_min_fin(Xi,c,d,basis,mu,sigma,p,grad_U,diag)
%% Strictly for MCMC application
% Computes the optimal parameter values for variance minimization - using
% the principle in ZV-MCMC paper by Mira et al. 
% Used for comparison with the asymptotic variance minimization method. 
  tic
  syms x 
  N = length(Xi);
  M = zeros(d);
  b = zeros(d,1);
  m = length(mu);
  d0 = d/2;    
 
  for i = 1:m
    p_basis(i) = exp(-(x-mu(i))^2/(2   * 1 * sigma(i)^2));    % factor a 
  end
  
  if basis == 0
  % Density weighted polynomial basis
   if p == 0
      for k=1:1:d
          psi(k)    = x^k;
      end
   else
      for k=0:1:d0-1
          psi(k+1)    = x^k * (p_basis(1))^p;
      end
      for k= d0+1:1:d
          psi(k)      = x^(k-(d0+1)) * (p_basis(2))^p;
      end
   end
  elseif basis == 1
%  Density weighted Fourier series basis 
    if p == 0
      for k=1:2:d-1
          psi(k)    = sin(x*k); 
          psi(k+1)  = cos(x*k);
      end
    else
      for k=1:2:d-1
          psi(k)    = sin(x*k) * (p_basis(1))^p;
          psi(k+1)  = cos(x*k) * (p_basis(1))^p;
      end
      for k= d0+1:2:d-1
          psi(k)     = sin((k-d0)*x) * (p_basis(2))^p;
          psi(k+1)   = cos((k-d0)*x) * (p_basis(2))^p;
      end
    end
  end
  grad_psi = diff(psi); 
  psi_ddot = diff(grad_psi);
  D_psi    = - grad_U * grad_psi + psi_ddot;
  
  grad_psi_x = matlabFunction(grad_psi);
  D_psi_x = matlabFunction(D_psi);
  
  H   = c(Xi);
  eta = mean(c(Xi));

for i = 1:1:N
    M = M + (1/N) * (D_psi_x(Xi(i))' * D_psi_x(Xi(i))) ;
    b = b + (1/N) * (H(i) - eta) * D_psi_x(Xi(i))';
end

theta = -(M\b);

for i = 1:N
    K(i) = theta' * grad_psi_x(Xi(i))';
end
toc
%% For displaying figures
if diag == 1
    figure;
    plot(Xi,K,'b*');
    hold on;
end
end