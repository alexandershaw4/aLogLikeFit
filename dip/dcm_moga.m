function [best, pareto, gaStats] = dcm_moga(DCM, opts)
% DCM_MOGA  Multiobjective GA calibration for spectral DCM (M/EEG).
%   [best, pareto, gaStats] = dcm_moga(DCM, opts)

% ---- Option defaults & merge
if nargin < 2 || ~isstruct(opts) || isempty(opts), opts = struct(); end
defaults = struct( ...
    'param_mask'     , [], ...
    'lb'             , [], ...
    'ub'             , [], ...
    'bound_scale'    , 3, ...
    'fwin1'          , [], ...
    'fwin2'          , [], ...
    'channel'        , 1, ...
    'popsize'        , 120, ...
    'ngen'           , 200, ...
    'crossoverFrac'  , 0.8, ...
    'mutationFcn'    , {@mutationgaussian, 0.2, 0.9}, ...
    'nStarts'        , 1, ...
    'lhsFrac'        , 1.0, ...
    'rngSeed'        , [], ...   % optional; if [], don't set RNG
    'useParallel'    , false, ...
    'verbose'        , 1);
fn = fieldnames(defaults);
for i = 1:numel(fn)
    f = fn{i};
    if ~isfield(opts, f) || isempty(opts.(f)), opts.(f) = defaults.(f); end
end

% Optional RNG: only set if user supplied a seed
if ~isempty(opts.rngSeed)
    rng(opts.rngSeed, 'twister');
end

% Guard: required toolbox
if ~exist('gamultiobj','file')
    error('dcm_moga:optTBX','Optimization Toolbox (gamultiobj) is required.');
end

% -------- Frequencies & data
Hz_data = DCM.xY.Hz(:);
ydata   = spm_vec(DCM.xY.y);
if ismatrix(ydata) && size(ydata,2) > 1
    ch = opts.channel;
    if isempty(ch), ydata = mean(ydata,2); else, ydata = ydata(:,ch); end
else
    ydata = ydata(:);
    ch = []; % no channel selection needed
end

% Use DCM.M.Hz if provided for predictions; else fall back to data Hz
if isfield(DCM,'M') && isfield(DCM.M,'Hz') && ~isempty(DCM.M.Hz)
    Hz = DCM.M.Hz(:);
else
    Hz = Hz_data;
end

% Objective windows (defaults if not supplied)
if isempty(opts.fwin1) || isempty(opts.fwin2)
    if max(Hz) > 60
        fwin1 = [10.7 65.8];
        fwin2 = [35.6 85.0];
    else
        fwin1 = [6.5 12.5];
        fwin2 = [11.5 30.0];
    end
else
    fwin1 = opts.fwin1;
    fwin2 = opts.fwin2;
end
I1 = Hz >= fwin1(1) & Hz <= fwin1(2);
I2 = Hz >= fwin2(1) & Hz <= fwin2(2);

% -------- Parameter vectorisation & mask
pE0  = DCM.M.pE;
vE0  = spm_vec(pE0);
if isfield(DCM.M,'pC') && ~isempty(DCM.M.pC)
    S0v = spm_vec(DCM.M.pC);
    sd  = sqrt(max(eps, S0v));
else
    sd  = 0.5*ones(size(vE0)); % fallback
end

if isempty(opts.param_mask)
    mask = sd > 0;        % free params where prior variance > 0 by default
else
    mask = logical(opts.param_mask(:));
end
theta0 = vE0(mask);

% Bounds
if ~isempty(opts.lb) && ~isempty(opts.ub)
    lb = opts.lb(:);
    ub = opts.ub(:);
else
    k  = opts.bound_scale;
    lb = theta0 - k*sd(mask);
    ub = theta0 + k*sd(mask);
end

npar = numel(theta0);

% -------- GA options
popsize   = opts.popsize;
ngen      = opts.ngen;
xoverFrac = opts.crossoverFrac;
mutFcn    = opts.mutationFcn;
usePar    = opts.useParallel;
lhsFrac   = opts.lhsFrac;
nStarts   = opts.nStarts;
verbose   = opts.verbose;

displayStr = 'off';
if verbose, displayStr = 'iter'; end

optsGA = optimoptions('gamultiobj', ...
    'PopulationSize',    popsize, ...
    'MaxGenerations',    ngen, ...
    'CrossoverFraction', xoverFrac, ...
    'MutationFcn',       mutFcn, ...
    'UseVectorized',     false, ...
    'UseParallel',       usePar, ...
    'Display',           displayStr);

% -------- LHS initial population
    function X0 = makeInit(pop, lb_, ub_, frac_)
        n = round(pop*frac_);
        if n < 1, X0 = []; return; end
        L = lhsdesign(n, numel(lb_), 'criterion','maximin','iterations',50);
        X = lb_' + L.*(ub_' - lb_');
        r = pop - n;
        if r > 0
            U = lb_' + rand(r, numel(lb_)).*(ub_' - lb_');
            X0 = [X; U];
        else
            X0 = X;
        end
    end

% -------- Fitness function: two RMSE objectives over bands
    function F = fitness(theta)
        vE = vE0; vE(mask) = theta(:);
        P  = spm_unvec(vE, pE0);

        % Predict spectra with current params
        yhat = feval(DCM.M.IS, P, DCM.M, DCM.xU);
        if isstruct(yhat) && isfield(yhat,'y'), yhat = yhat.y; end
        if ismatrix(yhat) && size(yhat,2) > 1
            if ~isempty(ch), yhat = yhat(:,ch); else, yhat = mean(yhat,2); end
        end
        yhat = spm_vec(yhat);

        % Interpolate if needed
        if numel(yhat) ~= numel(Hz)
            yhat = interp1(linspace(Hz(1), Hz(end), numel(yhat)), yhat, Hz, 'linear','extrap');
        end

        d1 = ydata(I1) - yhat(I1);
        d2 = ydata(I2) - yhat(I2);
        J1 = sqrt(mean(d1.^2));
        J2 = sqrt(mean(d2.^2));

        if any(~isfinite([J1 J2])), J1 = 1e6; J2 = 1e6; end
        F = [J1 J2];
    end

% -------- Run one or multiple starts; keep best by min ||F||
gaStats = struct('F',[],'X',[],'output',[],'start',[]);
bestAll = [];
paretoAllF = []; paretoAllX = [];

for s = 1:nStarts
    X0 = makeInit(popsize, lb, ub, lhsFrac);
    if ~isempty(X0)
        optsGA = optimoptions(optsGA, 'InitialPopulationMatrix', X0);
    else
        optsGA = optimoptions(optsGA, 'InitialPopulationMatrix', []);
    end

    [X,F,~,out] = gamultiobj(@fitness, npar, [],[],[],[], lb, ub, [], optsGA); %#ok<ASGLU>
    [~,iBest] = min(vecnorm(F,2,2)); % closest to (0,0)

    gaStats(s).F      = F;
    gaStats(s).X      = X;
    gaStats(s).output = out;
    gaStats(s).start  = s;

    bestAll     = [bestAll; X(iBest,:) F(iBest,:) s]; %#ok<AGROW>
    paretoAllF  = [paretoAllF; F];                    %#ok<AGROW>
    paretoAllX  = [paretoAllX; X];                    %#ok<AGROW>
end

% -------- Global best across starts
[~,iG]   = min(vecnorm(bestAll(:,npar+(1:2)),2,2));
thetaBest = bestAll(iG, 1:npar).';
FBest     = bestAll(iG, npar+(1:2)).';

% Rebuild full param struct and compute yhat for best
vE = vE0; vE(mask) = thetaBest;
Pbest = spm_unvec(vE, pE0);
yhat_best = feval(DCM.M.IS, Pbest, DCM.M, DCM.xU);
if isstruct(yhat_best) && isfield(yhat_best,'y'), yhat_best = yhat_best.y; end
if ismatrix(yhat_best) && size(yhat_best,2) > 1
    if ~isempty(ch), yhat_best = yhat_best(:,ch); else, yhat_best = mean(yhat_best,2); end
end
yhat_best = yhat_best(:);

% -------- Pack outputs
best = struct();
best.theta = thetaBest;
best.P     = Pbest;
best.F     = FBest;
best.Hz    = Hz;
best.yhat  = yhat_best;
best.ydata = ydata;
best.mask  = mask;

pareto = struct();
pareto.thetaSet = paretoAllX;
pareto.F        = paretoAllF;

end
