function [beta_m K] = gain_rkhs( Xi , c , kernel, lambda, epsilon, diag)
% Returns the gain computed at particle locations Xi using an RKHS 
% tic;
N = length(Xi);

% Evaluation of kernel matrices 
if kernel == 0  % Gaussian
    for i = 1:N
       for k = 1:N          
           if (Xi(i) ~= Xi(k)) || (i == k)
               Ker(i,k) =  exp(-(norm(Xi(i) - Xi(k)).^2/(4 * epsilon)));  
               Ker_x(i,k) =  - (Xi(i) - Xi(k)) / (2 * epsilon) .* Ker(i,k);
           end
       end
    end 
end  
   
 H     = c(Xi);       
 eta   = mean(c(Xi));
 Y     =  (H - eta)';
% b used in simplified algorithm - searching on a smaller subspace of the Hilbert space H
 b_m     = (1/N) * ( Ker * Y); 
 M_m     = lambda * Ker + (1 / N) * Ker_x' * Ker_x;       % Ker_x * Ker_x' = Ker_x' * Ker_x - Hence either one works
 beta_m  = M_m \ b_m;
 
        
 for i = 1: 1 : N
     K(i)     = 0;
     for k = 1 : 1 : N
         K(i)      = K(i)     + beta_m(k)  * Ker_x(i,k);      % Ker_x(pj,pi)
     end
 end
% toc 
 
if diag == 1
    figure;
    plot(Xi,K,'b*');
    
    for i = 1:249:N
        figure(1)
        plot(Xi,Ker(:,i),'r*');
        hold on;
        plot(Xi,Ker_x(:,i),'b*');  
    end
end
  
end

