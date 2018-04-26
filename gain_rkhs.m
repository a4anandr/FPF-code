function [K] = gain_rkhs(Xi,c,kernel,lambda,epsilon,diag)
% Returns the gain computed at particle locations Xi using an RKHS 
tic;
N = length(Xi);

% Evaluation of Gaussian kernel matrices 
if kernel == 0 
   Ker         =   exp(-(ones(N,1) * Xi - Xi' * ones(1,N)).^2/(4 * epsilon));        
   Ker_x       = -(ones(N,1) * Xi - Xi' * ones(1,N))/ (2 * epsilon) .* Ker;
   Ker_x_y     = (ones(N,N) - (ones(N,1) * Xi - Xi' * ones(1,N)).^2 /(2 * epsilon))/(2 * epsilon) .* Ker;
else
    
end
    
% Constructing block matrices for future use
 K_big      = [ Ker Ker_x ; Ker_x' Ker_x_y];
 K_thin_yxy = [ Ker_x ; Ker_x_y]; 
 K_thin_x   = [ Ker ; Ker_x'];
 
 H     = c(Xi);       
 eta   = mean(c(Xi));
 Y           =  (H - eta)';
% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
 b_m         =  (1/N) * Ker * Y; 
% b used in the extended representer theorem algorithm - searching over all of the Hilbert space H
 b_2m        =  (1/N) * K_thin_x * Y;
    
 M_m  = lambda * Ker + (1/N) * Ker_x * Ker_x';
 M_2m = lambda * K_big + (1/N) * K_thin_yxy * K_thin_yxy';
   
 beta_m  = M_m \ b_m;
 beta_2m = M_2m \ b_2m;   
        
 for pi = 1: 1 : N
     K(pi)     = 0;
     K_beta_2mi(pi)    = 0;
     K_beta_dot_mi(pi) = 0;
     for pj = 1 : 1 : N
         K(pi)      = K(pi)     + beta_m(pj)  * Ker_x(pj,pi);
         K_beta_2mi(pi)     = K_beta_2mi(pi)    + beta_2m(pj) * Ker_x(pj,pi) + beta_2m(N+pj) * Ker_x_y(pj,pi);
         K_beta_dot_mi(pi)  = K_beta_dot_mi(pi) - beta_m(pj)  * Ker_x_y(pj,pi);
     end
 end
toc 
 
if diag == 1
    figure;
    plot(Xi,K,'b*');
    hold on;
    plot(Xi,K_beta_2mi,'k^');
end
  
end

