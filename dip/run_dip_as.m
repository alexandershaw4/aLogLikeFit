function [mB VB Out] = run_dip_as(DCM)

% Data: log-PSD target (vector) and frequency axis
Hz    = DCM.xY.Hz;;
y_obs = DCM.xY.y{:};      % same length as Hz

% Model & priors
%DCM.M.f = @atcm.tc_twocmp_stp;
%DCM.M.x = your_state_template;           % ns×np×nk with nk=10, Vd at 8th
%DCM.M.L = your_leadfield;                % (channels×sources) or []
pE      = DCM.M.pE;                      % include gc, T, scale, pr, etc.

% Optionally supply your fast VL routine:
opts = struct('vl_fun', @fitVariationalLaplaceThermo, ...
              'm', 400, 'gen', 200, 'pop', 64, 'parallel', true);

Out = dip_hybrid_twocmp(y_obs, Hz, DCM.M, pE, opts);

% BMA posterior over θ-like vector:
mB = Out.BMA.m; VB = Out.BMA.V;
