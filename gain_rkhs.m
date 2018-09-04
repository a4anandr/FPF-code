function [beta_m K] = gain_rkhs( Xi , c , kernel, lambda, epsilon, alpha, K_prev, diag)
% Returns the gain computed at particle locations Xi using an RKHS 
% tic;
N = length(Xi);
K_prev = zeros(1,N);
simplified = 1;   % 0 for the optimal solution, 1 for the reduced complexity solution

% Evaluation of kernel matrices 
if kernel == 0  % Gaussian
   Ker         =  exp(-(ones(N,1) * Xi - Xi' * ones(1,N)).^2/(4 * epsilon));        
   Ker_x       = (ones(N,1) * Xi - Xi' * ones(1,N))/ (2 * epsilon) .* Ker;
   if simplified == 0
       Ker_x_y     = (ones(N,N) - (ones(N,1) * Xi - Xi' * ones(1,N)).^2 /(2 * epsilon))/(2 * epsilon) .* Ker;
   end
   
elseif kernel == 1  % Coifman basis - NOT valid as Coifman kernel is not symmetric, but still keeping the code 
   g         =  exp(-(ones(N,1) * Xi - Xi' * ones(1,N)).^2/(4 * epsilon));  % Basic Gaussian kernel for constructing Coifman kernel
  for i = 1:1:N
        for j = 1:1:N
            k(i,j) = g(i,j)./( sqrt( (1/N) * sum(g(j,:)))); % Coifman kernel - sqrt( (1/N) * sum(g(pi,:)))
        end
        Ker(i,:)  = k(i,:)./sum(k(i,:));                                       % Markov semigroup approximation
  end
  for i = 1:1:N
      sum_term(i) = Ker(i,:) * Xi';
      for j = 1:1:N
          Ker_x(i,j)    = (1/(2 * epsilon)) * Ker(i,j)* (Xi(j) - sum_term(i));  % Gain computed for particle index pi
      end
  end     
end
    
 H     = c(Xi);       
 eta   = mean(c(Xi));
 Y           =  (H - eta)';
% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
 b_m         =  (1/N) * ( Ker * Y + alpha * Ker_x' * K_prev'); 

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
    M_m  = lambda * Ker + ((1 + alpha)/ N) * Ker_x' * Ker_x;       % Ker_x * Ker_x' = Ker_x' * Ker_x - Hence either one works
    beta_m  = M_m \ b_m;
end  
        
 for i = 1: 1 : N
     K(i)     = 0;
     K_beta_2mi(i)    = 0;
     K_beta_dot_mi(i) = 0;
     for j = 1 : 1 : N
         K(i)      = K(i)     + beta_m(j)  * Ker_x(i,j);      % Ker_x(pj,pi)
         if simplified == 0
            K_beta_2mi(i)     = K_beta_2mi(i)    + beta_2m(j) * Ker_x(j,i) + beta_2m(N+j) * Ker_x_y(j,i);
            K_beta_dot_mi(i)  = K_beta_dot_mi(i) - beta_m(j)  * Ker_x_y(j,i);
         end
     end
 end
% toc 
 
if diag == 1
    figure;
    plot(Xi,K,'b*');
    hold on;
    plot(Xi,K_beta_2mi,'k^');
    
    for i = 1:249:N
        figure(1)
        plot(Xi,Ker(:,i),'r*');
        hold on;
        plot(Xi,Ker_x(:,i),'b*');  
    end
end
  
end

