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
  
% elseif kernel == 2 % Approximate coifman basis, requires to run em and get p at each instant
%     syms x
%     p = 0;
%     for i = 1:1:length(mu)
%         p   = p + w(i) * exp (- norm(x - mu(i))^2 / (2 * sigma(i)^2)) * (1 / sqrt( 2 * 3.1416 * sigma(i)^2));
%     end
%     % Defining potential function U(x)
%     p_x    = matlabFunction(p);
%     Utot   = - log(p);
%     Utot_mf  = matlabFunction(Utot);
%     del_Utot = diff(Utot);
%     del_Utot_mf= matlabFunction(del_Utot);
%     
%     % g         =  exp(-(ones(N,1) * Xi - Xi' * ones(1,N)).^2/(4 * epsilon));
%     for pi = 1:1:N
%         for pj = 1:1:N
%             g(pi,pj) = exp(-(Xi(pi) - Xi(pj))^2/ (4 * epsilon));
%             k(pi,pj) = g(pi,pj) * sqrt(p_x(Xi(pj))) / sqrt(p_x(Xi(pi))); % Coifman kernel - sqrt( (1/N) * sum(g(pi,:)))
%         end
%         Ker(pi,:)  = k(pi,:)./sum(k(pi,:));                                       % Markov semigroup approximation
%     end
%     for pi = 1:1:N
%        for pj = 1:1:N
%            Ker_x(pi,pj) = ((Xi(pj) - Xi(pi))/(2 * epsilon) + del_Utot_mf(Xi(pi))/2) * Ker(pi,pj);
%        end
%     end    
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
        
 for pi = 1: 1 : N
     K(pi)     = 0;
     K_beta_2mi(pi)    = 0;
     K_beta_dot_mi(pi) = 0;
     for pj = 1 : 1 : N
         K(pi)      = K(pi)     + beta_m(pj)  * Ker_x(pi,pj);      % Ker_x(pj,pi)
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
    plot(Xi,K_beta_2mi,'k^');
    
    for pi = 1:249:N
        figure(1)
        plot(Xi,Ker(:,pi),'r*');
        hold on;
        plot(Xi,Ker_x(:,pi),'b*');  
    end
end
  
end

