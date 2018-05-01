function [K] = gain_rkhs(Xi,c,kernel,lambda,epsilon,diag)
% Returns the gain computed at particle locations Xi using an RKHS 
% tic;
N = length(Xi);
simplified = 1;   % 0 for the optimal solution, 1 for the reduced complexity solution

% Evaluation of kernel matrices 
if kernel == 0  % Gaussian
   Ker         =  exp(-(ones(N,1) * Xi - Xi' * ones(1,N)).^2/(4 * epsilon));        
   Ker_x       = -(ones(N,1) * Xi - Xi' * ones(1,N))/ (2 * epsilon) .* Ker;
   if simplified == 0
       Ker_x_y     = (ones(N,N) - (ones(N,1) * Xi - Xi' * ones(1,N)).^2 /(2 * epsilon))/(2 * epsilon) .* Ker;
   end
   
elseif kernel == 1  % Approximate Coifman basis, choosing epsilon = 0    
   g         =  exp(-(ones(N,1) * Xi - Xi' * ones(1,N)).^2/(4 * epsilon));  % Basic Gaussian kernel for constructing Coifman kernel
  for pi = 1:1:N
        for pj = 1:1:N
            k(pi,pj) = g(pi,pj)./( sqrt( (1/N) * sum(g(pj,:)))); % Coifman kernel - sqrt( (1/N) * sum(g(pi,:)))
        end
        Ker(pi,:)  = k(pi,:)./sum(k(pi,:));                                       % Markov semigroup approximation
  end
  for pi = 1:1:N
      sum_term(pi) = Ker(pi,:) * Xi';
      for pj = 1:1:N
          Ker_x(pi,pj)    = (1/(2 * epsilon)) * Ker(pi,pj)* (Xi(pj) - sum_term(pi));  % Gain computed for particle index pi
      end
  end
end
    
 H     = c(Xi);       
 eta   = mean(c(Xi));
 Y           =  (H - eta)';
% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
 b_m         =  (1/N) * Ker * Y; 

if simplified == 0
     % Constructing block matrices for future use
    K_big      = [ Ker Ker_x ; Ker_x' Ker_x_y];
    K_thin_yxy = [ Ker_x ; Ker_x_y]; 
    K_thin_x   = [ Ker ; Ker_x'];
% b used in the extended representer theorem algorithm - searching over all of the Hilbert space H
    b_2m        =  (1/N) * K_thin_x * Y;
    M_2m = lambda * K_big + (1/N) * K_thin_yxy * K_thin_yxy';
    beta_2m = M_2m \ b_2m;   
else
    M_m  = lambda * Ker + (1/N) * Ker_x * Ker_x';
    beta_m  = M_m \ b_m;
end  
        
 for pi = 1: 1 : N
     K(pi)     = 0;
     K_beta_2mi(pi)    = 0;
     K_beta_dot_mi(pi) = 0;
     for pj = 1 : 1 : N
         K(pi)      = K(pi)     + beta_m(pj)  * Ker_x(pj,pi);
         if simplified == 0
            K_beta_2mi(pi)     = K_beta_2mi(pi)    + beta_2m(pj) * Ker_x(pj,pi) + beta_2m(N+pj) * Ker_x_y(pj,pi);
            K_beta_dot_mi(pi)  = K_beta_dot_mi(pi) - beta_m(pj)  * Ker_x_y(pj,pi);
         end
     end
 end
% toc 
 
if diag == 1
    figure;
    plot(Xi,K,'b*');
    hold on;
    % plot(Xi,K_beta_2mi,'k^');
    
    for pi = 1:1:N
        figure(1)
        plot(Xi,Ker(pi,:),'r*');
        hold on;
        
        figure(2)
        plot(Xi,Ker_x(pi,:),'b*');
        hold on;
        
        figure(3)
        plot(Xi,k(pi,:),'r*');
        hold on;
        
    end
end
  
end

