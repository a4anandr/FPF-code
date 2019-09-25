function [K] = gain_fin(Xi,c,d,basis,mu,sigma,p,diag)
% Returns the gain computed at particle locations Xi using a finite
% dimensional basis
  tic;
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
psi_x = matlabFunction(psi);
grad_psi = diff(psi); 
grad_psi_x = matlabFunction(grad_psi);

H   = c(Xi);
eta = mean(c(Xi));

for i = 1:1:N
    grad_psi_inner = (grad_psi_x(Xi(i))'* grad_psi_x(Xi(i)));
    M = M + (1/N) * grad_psi_inner ;
    b = b + (1/N) * (H(i) - eta) * psi_x(Xi(i))';
end

theta = (M\b);

for i = 1:N
    K(i) = theta' * grad_psi_x(Xi(i))';
end

if diag == 1
    figure;
    plot(Xi,K,'b*');
    hold on;
end
end