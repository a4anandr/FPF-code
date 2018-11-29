function dXbar_dt = normalised_spm_state_eqn(X,U,param)
% X(1) = qbar_pos, X(2) = qbar_neg, X(3) = csbar_neg
% U = I_load (input current in amps)

A        = param.A;
D_n      = param.D_n;
D_p      = param.D_p;
De       = param.De;
F        = param.F;
R_n      = param.R_n;
R_p      = param.R_p;
a_n      = param.a_n;
a_p      = param.a_p;
brugg_n  = param.brugg_n;
brugg_p  = param.brugg_p;
brugg_s  = param.brugg_s;
ce_c     = param.ce_c;
cs_avg_c = param.cs_avg_c;
eps_n    = param.eps_n;
eps_p    = param.eps_p;
eps_s    = param.eps_s;
len_n    = param.len_n;
len_p    = param.len_p;
len_s    = param.len_s;
qc       = param.qc;
tplus    = param.tplus;

%% Current density and flux density calcs
I_density =  U/A; % [A/m^2] U = I_load (input current in Amps)
j_p       = -I_density/(F*a_p*len_p);
j_n       =  I_density/(F*a_n*len_n);

%% RHS of state eqn (i.e. the state derivatives w.r.t time)
dqbar_dt_pos = (-30 * (D_p/R_p^2))*X(1) + (45/2)*(-j_p/R_p^2); 
dqbar_dt_neg = (-30 * (D_n/R_n^2))*X(2) + (45/2)*(-j_n/R_n^2); 
dcsbar_dt_neg = -3*j_n/R_n; % csbar_pos
dXbar_dt      = [dqbar_dt_pos; dqbar_dt_neg; dcsbar_dt_neg];

end