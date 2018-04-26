function [mu, sigma, w] = em_gmm (Xi, mu, sigma, w, diag)
% Given N particles Xi, this runs the EM algorithm and gives the parameters
% corresponding to the best GMM that would have given rise to the particles
% Xi

N = length(Xi);
epsilon = 0.01;
w0_thresh=0.05;    % Other tried values - (0.1)

% Means of the 3 component Gaussians
mu_em(1)    = mean(Xi);           % Mean of the first component is the ensemble mean - Don't change this in this function
sigma_em(1) = std(Xi);            % Std deviation of the first component is the ensemble std deviation - Don't change this in this function
w_em(1)     = max(w(1) , w0_thresh);    % Weight for the first component is hard-coded to a minimum of 0.1 to avoid high gain values

mu_em_new(1) = mu_em(1);
sigma_em_new(1) = sigma_em(1);
w_em_new(1) = w_em(1);

for i = 2:length(mu)
    mu_em(i)    = mu(i);          % Initialize the remaining two components with values passed to the function 
    sigma_em(i) = sigma(i);
    w_em(i)     = w(i);
end

diff_mu = 1;                     % Values to be checked for exit criteria 
diff_w  = 1;
iterations = 1;

while (norm(diff_mu) > epsilon| norm(diff_w) > epsilon)
    
    trash=0;                      % Any particle that does not belong to any of the components being thrown out
    for j=1:1:N
       for i = 1:length(mu)
            p(i) = w_em(i) * (1/(sqrt(2*pi)*sigma_em(i))) * exp (-(Xi(j)-mu_em(i))^2/ (2*sigma_em(i)^2));  % Relative contribution of each Gaussian to particle Xi(j)
       end
       pj(:,j)=p ./ sum(p);        % Probability of Xi(j) having come from i^th Gaussian

       if (sum(isnan(pj(:,j))))
           
            if(isnan(pj(1,j)))
                pj(1,j)=0;
            end
            
            if(isnan(pj(2,j)))
                pj(2,j)=0;
            end
            
            if(isnan(pj(3,j)))
                pj(3,j)=0;
            end
            
            if(sum(pj(:,j))==0)
                trash=trash+1;
                i=i+1;
            end            
            j;           
        else
            continue;
        end
    end
    
     % M - step 
     
     %% Mixing probabilities
     for i = 1: length(mu)
        w_em_t(i) = (1/(N-trash)) * sum(pj(i,:));
     end
     
     w_em_new = w_em_t./sum(w_em_t);
     w_em_new(1)=max(w_em_new(1),w0_thresh);
     for i = 2: length(mu)
        w_em_new(i)= w_em_t(i)/(sum(w_em_t(2:end))) * (1-w_em_new(1));
        if(isnan(w_em_new(i)))
            w_em_new(i)=0;
        end
     end    
                
     %% Means
     for i = 2:length(mu)
        if(sum(pj(i,:))==0)
            mu_em_new(i) = mu_em(i);
        else
            mu_em_new(i) = pj(i,:) * Xi'/ sum(pj(i,:));
        end
     end
         
     %% Standard deviations
     for i = 2:length(mu)
         if(sum(pj(i,:))==0)
            sigma_em_new(i) = max(sigma_em(i),0.1);
         else
            sigma_em_new(i) = sqrt(pj(i,:)*((Xi - mu_em_new(i)).^2)'/ sum(pj(i,:)));        
            sigma_em_new(i) = max(sigma_em_new(i),0.1);
         end
     end
    
     %% Criteria for convergence of EM
     
    diff_mu = mu_em_new - mu_em ;
    diff_w  = w_em_new  - w_em;
    
    mu_em   = mu_em_new;
    w_em    = w_em_new;
    sigma_em = sigma_em_new;
    
    iterations = iterations + 1;
end

mu = mu_em;
w  = w_em;
sigma = sigma_em;

if(isnan(mu)| isnan(sigma)|isnan(w))
    display('Error in EM algorithm');
    pause;
    
end
