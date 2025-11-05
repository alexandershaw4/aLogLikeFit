function Out = dip_hybrid_twocmp(y_obs, Hz, M, pE, opts)
% DIP hybrid for 2-comp TCM:
%   1) GA on a low-dim θ (global offsets/scales) with banded RMSE objectives
%   2) Make m dynamics-informed Gaussian priors (means=GA best; unit-ish vars)
%   3) Run VL from each prior -> {mi,Vi,Fi}
%   4) BMA across runs with softmax(F)

% ----- options -----
if nargin < 5, opts = struct; end
opts = setdef(opts,'m', 300);           % number of DIP priors to invert
opts = setdef(opts,'gen', 200);         % GA generations (150–300 good)
opts = setdef(opts,'pop', 48);          % GA population size
opts = setdef(opts,'bands', [6.5 12.5; 11.5 30]); % two objectives (Hz)
opts = setdef(opts,'parallel', true);
opts = setdef(opts,'vl_fun', []);       % handle to your VL inverter (optional)
opts = setdef(opts,'verbose', true);

M.f = @atcm.tc_twocmp_stp;
M.g = @g_linJ_psd;     % below; uses pE.J → vec(x) → sensor, then PSD via Laplace TF
M.Hz = Hz(:)';

% ---- parameterisation to search (compact θ) ----
% θ packs a few global knobs that move spectra without exploding dimension:
%   θ = [ dT_E, dT_I, dT_N, dT_B,  log_gc,  pr_shift,  log_GEa,log_GEn,log_GIa,log_GIb,  wE_dend_AMPA, wE_dend_NMDA ]
lb = [-1.5  -1.0  -1.0  -1.0   log(0.5)  -0.5     -1.0  -1.0  -1.0  -1.0   0.2  0.5];
ub = [ +1.0  +1.0  +1.0  +1.0   log(8.0)  +0.5     +1.0  +1.0  +1.0  +1.0   0.95 0.98];

% ---- GA setup (NSGA-II) ----
nvar = numel(lb);
fitfun = @(theta) obj_psd(theta, y_obs, M, pE, opts.bands);
optsGA = optimoptions('gamultiobj','PopulationSize',opts.pop,'MaxGenerations',opts.gen,...
    'UseVectorized',false,'Display','final');
if opts.parallel, optsGA = optimoptions(optsGA,'UseParallel',true); end

[Pop, Fvals, ~, GAoutput] = gamultiobj(fitfun, nvar, [],[],[],[], lb, ub, [], optsGA);

% rank & select top-m by Euclidean distance in objective space
fmin = min(Fvals,[],1); fmax = max(Fvals,[],1);
Fn = (Fvals - fmin)./max(1e-9,(fmax-fmin));   % normalise
dist = sqrt(sum(Fn.^2,2));
[~,ord] = sort(dist,'ascend');
Top = Pop(ord(1:min(opts.m, size(Pop,1))),:);

% ---- build dynamics-informed priors ----
Priors = arrayfun(@(i) theta2prior(Top(i,:), pE, M), 1:size(Top,1), 'uni',0);

% ---- VL inversions from each prior ----
VL = cell(numel(Priors),1);
parfor i = 1:numel(Priors)
    try
        [mi, Vi, Fi] = run_local_VL(y_obs, M, Priors{i}, opts);
        VL{i} = struct('m',mi,'V',Vi,'F',Fi,'ok',true);
    catch ME
        VL{i} = struct('m',[] ,'V',[] ,'F',-Inf,'ok',false,'err',ME.message);
    end
end

% ---- BMA across successful runs ----
ok = cellfun(@(v) ~isempty(v) && v.ok, VL);
F  = cellfun(@(v) v.F, VL(ok));
Ms = cellfun(@(v) v.m, VL(ok),'uni',0);
Vs = cellfun(@(v) v.V, VL(ok),'uni',0);

if isempty(F)
    error('DIP hybrid: all VL runs failed. Check bands, bounds, or model stability.');
end

w = exp(F - max(F));  w = w./sum(w);
mB = 0; for i=1:numel(Ms), mB = mB + w(i)*Ms{i}; end
VB = 0;
for i=1:numel(Ms)
    dm = Ms{i} - mB;
    VB = VB + w(i)*(Vs{i} + (dm*dm'));   % law of total covariance
end

% ---- package outputs ----
Out = struct;
Out.GA.population = Pop;
Out.GA.fvals      = Fvals;
Out.GA.output     = GAoutput;
Out.Priors        = Priors;
Out.VL            = VL;
Out.BMA.m         = mB;
Out.BMA.V         = VB;
Out.BMA.w         = w;
Out.M             = M;
Out.pE            = pE;
if opts.verbose
    fprintf('DIP hybrid: %d/%d VL runs succeeded. BMA complete.\n', numel(F), numel(VL));
end
end

% ----------------------- helpers -----------------------
function f = obj_psd(theta, y_obs, M, pE, bands)
    P = apply_theta(theta, pE, M);
    try
        [yhat,~,~,~] = atcm.Alex_LaplaceTFwD(P, M, []);   % returns PSD over M.Hz
        yhat = yhat(:);
    catch
        % penalise unstable / failed linearisation
        f = [1e6 1e6]; return
    end
    yhat = log(max(yhat, eps)); ydat = log(max(y_obs(:), eps));
    f = zeros(1,size(bands,1));
    for k=1:size(bands,1)
        sel = M.Hz >= bands(k,1) & M.Hz <= bands(k,2);
        d   = ydat(sel) - yhat(sel);
        f(k)= sqrt(mean(d.^2));
    end
end

function P = apply_theta(theta, pE, M)
    P = pE;
    dT = [theta(1) theta(2) theta(3) theta(4)];   % global offsets for AMPA,GABA,NMDA,GABAb
    if ~isfield(P,'T') || isempty(P.T), P.T = zeros(size(M.x,2),6); end
    P.T(:,1) = P.T(:,1) + dT(1);  % AMPA
    P.T(:,2) = P.T(:,2) + dT(2);  % GABAa
    P.T(:,3) = P.T(:,3) + dT(3);  % NMDA
    P.T(:,4) = P.T(:,4) + dT(4);  % GABAb
    P.gc     = theta(5);          % log(gc)
    P.pr     = theta(6);          % firing midpoint shift (log-scale in your model)
    P.scale  = [theta(7) theta(8) theta(9) theta(10)];  % log scales of GEa,GEn,GIa,GIb
    P.w_dend = [theta(11) theta(12)];
    % keep other fields (C, CV, STP etc.) as in pE
end

function Prior = theta2prior(theta, pE, M)
    % mean = GA solution; covariance = unit-ish on the GA-controlled dims
    m = vecP(apply_theta(theta, pE, M)); 
    S = speye(numel(m));            % simple unit diag (tune if you like)
    Prior = struct('m0',m,'S0',S,'P0',apply_theta(theta, pE, M));
end

function [m, V, F] = run_local_VL(y_obs, M, Prior, opts)
    % wraps your VL routine; falls back to SPM if not given
    f_psd = @(theta) f_from_theta(theta, M, Prior.P0);  % maps θ→ PSD
    if ~isempty(opts.vl_fun)
        [m, V, ~, F] = opts.vl_fun(y_obs(:), f_psd, Prior.m0, Prior.S0, 64, 1e-4, 0, 0.99, struct);
    else
        % SPM fallback (simple GN around Laplace TF)
        fun = @(th) f_psd(th);
        m   = Prior.m0;
        V   = Prior.S0;
        [m, V, F] = spm_nlsi_GN(struct('IS',fun,'xU',struct,'M',M), y_obs(:), m, V);
    end
end

function y = f_from_theta(theta, M, P0)
    P = unvecP(theta, P0);               % put θ back into a P-struct
    [Y,~,~,~] = atcm.Alex_LaplaceTFwD(P, M, []);  % model PSD at M.Hz
    y = Y(:);
end

function y = g_linJ_psd(x, P, M)
    % This is only used if you linearise elsewhere; here we rely on Laplace TF for PSD.
    % Keep as a placeholder for completeness.
    y = P.J * spm_vec(x);
end

% ---- param vectorisers (only touch fields we actually changed) ----
function v = vecP(P)
    v = [];
    v = [v; P.T(:,1); P.T(:,2); P.T(:,3); P.T(:,4)];
    v = [v; P.gc; P.pr; P.scale(:); P.w_dend(:)];
end
function P = unvecP(v, P)
    np = size(P.T,1);
    cut = 0;
    P.T(:,1) = v(cut+(1:np)); cut = cut+np;
    P.T(:,2) = v(cut+(1:np)); cut = cut+np;
    P.T(:,3) = v(cut+(1:np)); cut = cut+np;
    P.T(:,4) = v(cut+(1:np)); cut = cut+np;
    P.gc     = v(cut+1);      cut = cut+1;
    P.pr     = v(cut+1);      cut = cut+1;
    P.scale  = v(cut+(1:4));  cut = cut+4;
    P.w_dend = v(cut+(1:2));  cut = cut+2;
end

function S = setdef(S, fld, val)
    if ~isfield(S,fld) || isempty(S.(fld)), S.(fld) = val; end
end
