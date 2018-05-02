function [cell_voltage_spm] = spm_battery_voltage(X,U,param)
% normalised_q_3states_spm_battery_voltage  - computes the output voltage
% using the normalised state vector, SPM parameters and load current as it's input
% X(1) = qbar_pos; X(2) = qbar_neg; X(3) = csbar_pos X(4) = ce_bar
% U = I_load; (load current in amps)
% U > 0 -> discharge, U < 0 -> charge

%% Extract parameters required for calculating all the components of output voltage
A             = param.A;
D_n           = param.D_n;
D_p           = param.D_p;
De            = param.De;
F             = param.F;
R             = param.R;             % universal gas constant
R_n           = param.R_n;
R_p           = param.R_p;
T             = param.T_init;        % isothermal
a_n           = param.a_n;
a_p           = param.a_p;
ce_c          = param.ce_c;          % Unused, might need to change to 1
ce_init       = param.ce_init;
cs_avg_c      = param.cs_avg_c;      % scaling factor for "cs_avg" (cs_avg_characteristic)
cs_max_n      = param.cs_max_n;
cs_max_p      = param.cs_max_p;
eps_n         = param.eps_n;
eps_p         = param.eps_p;
k_n           = param.k_n;
k_p           = param.k_p;
len_n         = param.len_n;
len_p         = param.len_p;
qc            = param.qc;            % scaling factor for "q" (q_characteristic)
theta_max_neg = param.theta_max_neg;
theta_max_pos = param.theta_max_pos;
theta_min_neg = param.theta_min_neg;
theta_min_pos = param.theta_min_pos;

%% Calculatons common to all four components of output voltage
% First convert to current density and compute the molar flux (assumed uniform along through-thickness direction in the model)
I_density =  U/A; % [A/m^2] U = I_load (input current in Amps)
j_p       = -I_density/(F*a_p*len_p);
j_n       =  I_density/(F*a_n*len_n);

%% Calculatons common to overpotential & OCP components of output voltage
% cs_surf_pos = cs_avg_c*X(3) + (8*R_p/35)*qc*X(1) - (R_p./(35*D_p)).*j_p; % surface concentration % surface concentration of the single pos electrode particle using the computed state-variable solution (csbar_avg)
cs_surf_pos = X(1) - (R_p./(5*D_p)).*j_p; % surface concentration % surface concentration of the single pos electrode particle using the computed state-variable solution (csbar_avg)
cs_avg_neg = cs_max_n*(theta_min_neg + (( X(1) - theta_min_pos*cs_max_p)./((theta_max_pos - theta_min_pos)*cs_max_p))*(theta_max_neg - theta_min_neg));  % average concentration in neg electrodue (analytical expn using conservation of charge/mass)
cs_surf_neg = cs_avg_neg  - (R_n./(5*D_n)).*j_n; % surface concentration of the single neg electrode particle

%% Compute overpotentials (without electrolyte dynamics) in both electrodes (1st component of output voltage)
eta_p = (2*R*T/F)*asinh(-I_density./(2*F*a_p*len_p*k_p*sqrt(ce_init*cs_surf_pos.*(cs_max_p - cs_surf_pos))));
eta_n = (2*R*T/F)*asinh(I_density./(2*F*a_n*len_n*k_n*sqrt(ce_init*cs_surf_neg.*(cs_max_n - cs_surf_neg))));

%% Compute OCPs in both electrodes (2nd component of output voltage)
theta_pos = cs_surf_pos/cs_max_p; % surface stoichiometry (pos)
theta_neg = cs_surf_neg/cs_max_n; % surface stoichiometry (neg)
Uocp_p = compute_Uocp_pos(theta_pos);
Uocp_n = compute_Uocp_neg(theta_neg);

%% Compute Electrolye Concentration & Electrolyte Potential Difference (3rd component of output voltage)
% ce_neg_cc_analytical = ce_c*X(4);
% ce_pos_cc_algebraic = -(len_n*eps_n)/(len_p*eps_p)*(ce_neg_cc_analytical - ce_init) + ce_init;
% ce_pos_cc_algebraic=-1.479*(ce_neg_cc_analytical)+2479;
phi_e_terminal_diff = 0;  % compute_phi_e_terminal_diff(param,ce_pos_cc_algebraic,ce_neg_cc_analytical,I_density);

%% Obtain Solid Potential at both Current Collectors
phi_p = eta_p + Uocp_p;
phi_n = eta_n + Uocp_n;

%% Lumped Resistance to model IR drop (4th & final component of output voltage)
R_col = 0;

%% Finally, compute cell voltage and return
cell_voltage_spm = (phi_p - phi_n) + phi_e_terminal_diff - U*R_col;

end