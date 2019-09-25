% Function module that takes in the parameters of the current Gaussian
% mixture and outputs the TD learning approximation of the FPF gain
% function - Nonlinear parameterization

function [p,ml,K_td_x,K_td_dot_x] =gain_td_nl_snr(mu0,mu1,mu2,sigma0,sigma1,sigma2,w0,w1,Td)

% Declaration of global variables
global h;
global a;
global sigmaW;
global Ntd; % For Monte Carlo simulations
global X;
testfigure=0;

Ntd=100;

% Declaration of symbolic variables
syms x; 
syms mu_s;
syms sig;

% Declaring the parameters as symbolic variables
syms at1 at2 at3;
syms bt1 bt2 bt3;
syms ct1 ct2 ct3;
syms kt;

persistent a1_final a2_final a3_final b1_final b2_final b3_final c1_final c2_final c3_final k0_final;

% Timescale
dt=0.1;
sdt=sqrt(dt);
interval=(Td/dt)/5;

% This is the posterior density estimate provided by the EM function.
mut0=mu0;
sigt0=sigma0;
mut1=mu1;
sigt1=sigma1;
mut2=mu2;
sigt2=sigma2;
wt0=w0;
wt1=w1;
wt2=1-wt0-wt1;

% Symbolic functions
pt =(1/(sqrt(2*pi)*sig))*exp(-(x-mu_s)^2/(2*sig^2));
% Matlab functions
pt_x = matlabFunction(pt);

pt0_x = pt_x(mut0,sigt0,x);
pt1_x = pt_x(mut1,sigt1,x);
pt2_x = pt_x(mut2,sigt2,x);
ptot  = wt0*pt0_x + wt1*pt1_x + wt2*pt2_x;

Utot   = -log(ptot);
ptot_x = matlabFunction(ptot);
del_Utot = diff(Utot);
Utot_double_prime= (x+diff(del_Utot))-(1+1e-10)*x;
Utot_x=matlabFunction(Utot);
del_Utot_x=matlabFunction(del_Utot);
Utot_double_prime_x=matlabFunction(Utot_double_prime);

p=ptot_x(X);
[m ind]=max(p);
ml=X(ind);
    
d=10; % Dimension of the basis functions
varPhi=zeros(d,Ntd);       % Eligibility vector

%% For use in matrix inversion lemma to obtain the evolution of theta_{T} with respect to various Ts
zeta=(at1/((x-bt1)^2+ct1^2)) + (at2/((x-bt2)^2+ct2^2)) + (at3/((x-bt3)^2+ct3^2)) + kt;

zeta_at1_dot = 1 /( (x-bt1)^2 + ct1^2 );
zeta_bt1_dot = 2 * at1* (x-bt1) /( (x-bt1)^2  +  ct1^2 )^2;
zeta_ct1_dot = - 2 *at1 *ct1 /( (x-bt1)^2 + ct1^2 )^2;

zeta_at2_dot = 1 /( (x-bt2)^2 + ct2^2 );
zeta_bt2_dot = 2 * at1 * (x-bt2) /( (x-bt2)^2  +  ct2^2 )^2;
zeta_ct2_dot = -2 * at2 * ct2 /( (x-bt2)^2 + ct2^2 )^2;

zeta_at3_dot = 1 /( (x-bt3)^2 + ct3^2 );
zeta_bt3_dot = 2 * at3 * (x-bt3) /( (x-bt3)^2  +  ct3^2 )^2;
zeta_ct3_dot = - 2 * at3 * ct3 /( (x-bt3)^2 + ct3^2 )^2;

zeta_kt_dot=1;

%% Initializing theta; weight parameter for each of these basis functions

if(isempty(a1_final)&& isempty(a2_final) && isempty(a3_final) && isempty(b1_final) && isempty(b2_final) && isempty(b3_final) && isempty(c1_final) && isempty(c2_final) && isempty(c3_final) && isempty(k0_final))
    a1_final =rand;
    a2_final =rand;
    a3_final =rand;
    b1_final =rand;
    b2_final =rand;
    b3_final =rand;
    c1_final =rand;
    c2_final =rand;
    c3_final =rand;
    k0_final = rand;
end

a1=a1_final; % rand;
a2=a2_final; % rand;
a3=a3_final; % rand;

b1=b1_final;
b2=b2_final;
b3=b3_final;

c1=c1_final;
c2=c2_final;
c3=c3_final;

k0=k0_final;

K_theta = zeta;
K_theta_x = matlabFunction(K_theta);
K_theta_initial = K_theta_x(a1,a2,a3,b1,b2,b3,c1,c2,c3,k0,X);

i=1;

% Gain term in Newton Raphson method, at = 1/(t+1)^beta
beta=1;

% Initializing Phi(1) ~ pt
r=rand;
for i=1:1:Ntd
    if (r<=wt0)
        Phi(i)=mut0+sigt0*randn;
    elseif (r<= wt0+wt1) 
        Phi(i)=mut1+sigt1*randn;
    else
        Phi(i)=mut2+sigt2*randn;
    end
end

int_Mt_inv=eye(d);

i=1;
% dt1=1;

for k=1:1:Td/dt   
    if (mod(k,1000)==0)
        k
    end
    gain=(1/(k+1))^beta;
   
    Phi = Phi - del_Utot_x(Phi)*dt + sqrt(2)*randn(1,Ntd)*sdt;
    
    psi_theta_snr(1,:)=1./((Phi - b1(k)).^2 + c1(k)^2);
    psi_theta_snr(2,:)=2 * a1(k) * (Phi - b1(k))./ ((Phi - b1(k)).^2 + c1(k)^2).^2;
    psi_theta_snr(3,:)= -2 * a1(k) * c1(k) ./( (Phi-b1(k)).^2 + c1(k)^2).^2;   
    psi_theta_snr(4,:)=1./((Phi - b2(k)).^2 + c2(k)^2);
    psi_theta_snr(5,:)=2 * a2(k) * (Phi - b2(k))./ ((Phi - b2(k)).^2 + c2(k)^2).^2;
    psi_theta_snr(6,:)= -2 * a2(k) * c2(k) ./( (Phi-b2(k)).^2 + c2(k)^2).^2;  
    psi_theta_snr(7,:)=1./((Phi - b3(k)).^2 + c3(k)^2);
    psi_theta_snr(8,:)=2 * a2(k) * (Phi - b3(k))./ ((Phi - b3(k)).^2 + c3(k)^2).^2;
    psi_theta_snr(9,:)= -2 * a3(k) * c3(k) ./( (Phi-b3(k)).^2 + c3(k)^2).^2; 
    psi_theta_snr(10,:)=1;
        
    varPhi  =  varPhi-(ones(d,1)*Utot_double_prime_x(Phi)).*varPhi*dt + psi_theta_snr*dt;    
    int_Mt_inv=int_Mt_inv+(1/Ntd)*(psi_theta_snr * psi_theta_snr');  % Factor of 2 is probably not required for SNR; it may be required for SA.
    Mt_inv=(1/(k))*int_Mt_inv;
    
    Ktheta_psitheta=(1/Ntd)*psi_theta_snr*K_theta_x(a1(k),a2(k),a3(k),b1(k),b2(k),b3(k),c1(k),c2(k),c3(k),k0(k),Phi)';
        
    Mt = inv(Mt_inv);
    varPhi_mean = mean(varPhi,2);
    term= Mt * ((h/sigmaW^2)*varPhi_mean - Ktheta_psitheta);
        
    a1(k+1)=min(max(a1(k)+gain*term(1),-20),100);
    b1(k+1)=min(max(b1(k)+gain*term(2),-20),100);
    c1(k+1)=min(max(c1(k)+gain*term(3),0),100);
    
    a2(k+1)=min(max(a2(k)+gain*term(4),-20),100);
    b2(k+1)=min(max(b2(k)+gain*term(5),-20),100);
    c2(k+1)=min(max(c2(k)+gain*term(6),0),100);
    
    a3(k+1)=min(max(a3(k)+gain*term(7),-20),100);
    b3(k+1)=min(max(b3(k)+gain*term(8),-20),100);
    c3(k+1)=min(max(c3(k)+gain*term(9),0),100);
    
    k0(k+1)=min(max(k0(k)+gain*term(10),0),100);
end
a1_final=a1(k+1);
a2_final=a2(k+1);
a3_final=a3(k+1);
b1_final=b1(k+1);
b2_final=b2(k+1);
b3_final=b3(k+1);
c1_final=c1(k+1);
c2_final=c2(k+1);
c3_final=c3(k+1);
k0_final=k0(k+1);

K_td= K_theta_x(a1_final,a2_final,a3_final,b1_final,b2_final,b3_final,c1_final,c2_final,c3_final,k0_final,x);
K_td_dot=diff(K_td);
K_td_x=matlabFunction(K_td);
K_td_dot_x=matlabFunction(K_td_dot);

% h_double_dot_x=matlabFunction(-(-del_Utot*K_td+(1/sigmaW^2)*(h*x-eta)));
% h_ddot_convex_x=matlabFunction(sqrt(ptot)*(-(-del_Utot*K_td+(1/sigmaW^2)*(h*x-eta)))+(1-sqrt(ptot))*K_td_dot);   

t=0:dt:Td;
if(testfigure)
figure;
plot((1:Td/dt),a1(1:Td/dt),'k','linewidth',2.0);
hold on;
plot((1:Td/dt),b1(1:Td/dt),'r','linewidth',2.0);
plot((1:Td/dt),c1(1:Td/dt),'b','linewidth',2.0);
legend('a_{1}','b_{1}','c_{1}');
title('Convergence of \Theta_{1}');
xlabel('T ->');
ylabel('a_{1},b_{1},c_{1}');

figure;
plot((1:Td/dt),a2(1:Td/dt),'k','linewidth',2.0);
hold on;
plot((1:Td/dt),b2(1:Td/dt),'r','linewidth',2.0);
plot((1:Td/dt),c2(1:Td/dt),'b','linewidth',2.0);
legend('a_{2}','b_{2}','c_{2}');
title('Convergence of \Theta_{2}');
xlabel('T ->');
ylabel('a_{2},b_{2},c_{2}');

figure;
plot((1:Td/dt),a3(1:Td/dt),'k','linewidth',2.0);
hold on;
plot((1:Td/dt),b3(1:Td/dt),'r','linewidth',2.0);
plot((1:Td/dt),c3(1:Td/dt),'b','linewidth',2.0);
legend('a_{3}','b_{3}','c_{3}');
title('Convergence of \Theta_{3}');
xlabel('T ->');
ylabel('a_{3},b_{3},c_{3}');

end






