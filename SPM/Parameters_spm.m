function param = Parameters_spm(init_cell_soc_percent) % Cell SOC [%] to begin simulation with

param.A = 2.05268559698939; % m^2; overall active surface area
%     param.I_1C = 60; % Amps (1C current of the cell, i.e. cell capacity in disguise)

%% Solid particle radius [m]
param.R_n = 2e-6;
param.R_p = 2e-6;

%% Solid phase volume fraction
param.vol_fraction_neg = 0.4824; % unclear
param.vol_fraction_pos = 0.5900; % unclear

%% Domain  Lengths [m]
param.len_n = 88e-6;
param.len_s = 25e-6;        % Not needed?
param.len_p = 72e-6;

%% Maximum concentration of Li-ions in the solid phase [mol/m^3]
param.cs_max_n = 30555;
param.cs_max_p = 51554;

%% Solid diffusion coefficients [m^2 / s]
param.D_n = 3.9e-14;
param.D_p = 1.0e-14;

%% Reaction rate coefficients [m^2.5/(mol^0.5 s)]
param.k_n = 5.031e-11;
param.k_p = 2.334e-11;

%% Electrolyte Volume Fractions
param.eps_n = 0.485;             % All 3 not in the table
param.eps_s = 0.724;
param.eps_p = 0.385;

%% Electrolyte Conductivity Function
param.ce_init = 1000;   % Electrolyte Li-ions initial concentration [mol/m^3]
param.T_init  = 298.15; % Initial temperature of the cell [K]
param.Tref = param.T_init; % used in Electrolyte Conductivity Polynomial
param.Kappa_function = @spm_electrolyte_conductivity; % obtains the function handle to the relevant file containing the polynomials describing kappa as a function of ce (and T)

%% Other Electrolyte Parameters & coefficients
param.tplus = 0.364;
param.De = 3.2227e-10;

%% Bruggeman Coefficients
param.brugg_n = 4;
param.brugg_s = 4;
param.brugg_p = 4;

%% Stoichiometry Limits
param.theta_max_pos = 0.49550;  % at 100% cell SOC
param.theta_max_neg = 0.85510;  % at 100% cell SOC
param.theta_min_pos = 0.99174;  % at 0% cell SOC
param.theta_min_neg = 0.01429;  % at 0% cell SOC

%% Universal Constants
param.F = 96487; % Faraday Constant  [C/mol]
param.R = 8.314; % Gas constant      [J / (mol K)]

%% Cut-off Conditions
param.CutoffVoltage = 2.5;   % Cutoff voltage [V]
param.CutoverVoltage = 4.3;  % Cutover voltage [V]
param.CutoffSOC = 0;  % Cutoff SOC [%]
param.CutoverSOC = 100;  % Cutover SOC [%]
param.Tmax = 55 + 273.15;

%% Interfacial surface area Calculations [m^2 / m^3]
param.a_n = 3*param.vol_fraction_neg/param.R_n; % Negative electrode
param.a_p = 3*param.vol_fraction_pos/param.R_p; % Positive electrode

%% Computation of initial concentration of Li-ions in the solid phase [mol/m^3]
param.init_cell_soc = init_cell_soc_percent/100; % z(0)         % convert to a fraction between 0 and 1
param.cs_p_init = ((param.init_cell_soc*(param.theta_max_pos-param.theta_min_pos) + param.theta_min_pos))*param.cs_max_p;
param.cs_n_init = ((param.init_cell_soc*(param.theta_max_neg-param.theta_min_neg) + param.theta_min_neg))*param.cs_max_n;

%% Other Simulation-Specific Settings
param.TemperatureEnabled = 0;  % isothermal
param.enable_csneg_Saturation_limit = 0;
param.suppress_status_prints = 0;

%% Scaling factors for non-dimensionalising state-variables of SPM
% Might need to set all these variables to 1 or completely avoid them
param.cs_avg_c = 1e5;    % hard-coded for now (based on the order of magnitude of max concentrations in both electrodes)
param.qc = 1e4*param.cs_avg_c;  % hard-coded for now (based on open-loop simulations with SPM)
param.ce_c = param.ce_init;     % might need to set to 1

%% Current Collector Resistance for SPM
param.use_Rcol = 'n';    % used to compensate for IR drop in battery (only for EKF code)
% param.R_col = 8e-4; % combined [ohms] series resistance of the current collectors

end