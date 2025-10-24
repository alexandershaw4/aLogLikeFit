function [m, V, D, logF, iter, sigma2, trace] = ...
    fitVariationalLaplaceThermoStruct(y, f, m0, S0, maxIter, tol, plots, varpercthresh, opts)
% fitVariationalLaplaceThermo — Thermodynamic Variational Laplace with structure learning.
%
% Adds:
%   - Active-parameter update mask I (others clamped at prior mean)
%   - Evidence gating via z-scores and fast BMR-style ΔF tests
%   - Optional ARD shrinkage on parameters (no first-level noise model needed)
%
% Signature kept compatible with older calls. New behavior via opts.
%
% Inputs (as before):
%   y        : data vector (N×1 or N×d stacked)
%   f        : model function handle. Must support:
%                yhat = f(theta)                          % predictions
%                [yhat, J] = f(theta)                     % + Jacobian d yhat / d theta
%              Optionally: f can accept extra args via opts.model_args (cell)
%   m0, S0   : prior mean and covariance for parameters (p×1, p×p)
%   maxIter  : max outer iterations
%   tol      : ELBO/free-energy convergence tolerance
%   plots    : 0/1 (optional; default 0)
%   varpercthresh : (unused here; preserved in signature)
%   opts     : struct with fields (all optional)
%       .tau_sched       (1×maxIter) annealing temperatures (default: ones)
%       .useARD          logical (default: false)
%       .lambda0         initial ARD precisions (p×1 or scalar; default: 1e-3)
%       .lambda_max      cap for ARD precision (default: 1e6)
%       .z_thresh        z-score inclusion threshold (default: 2.5)
%       .kappa_prune     ΔF threshold for pruning (default: 0)     % prune if ΔF < 0
%       .kappa_add       ΔF threshold for additions (default: 3)   % ~ strong-ish BF
%       .max_add_per_it  maximum candidates to add per outer iter (default: 3)
%       .I0              initial active set (indices; default: all)
%       .propose_fun     @(I, p) -> idx of candidates not in I (default: empty)
%       .model_args      cell array of extra args for f (default: {})
%
% Outputs:
%   m, V     : posterior mean & covariance (p×1, p×p)
%   D        : struct with bookkeeping (active sets, lambdas, ΔF history)
%   logF     : free-energy trajectory (outer iters)
%   iter     : number of outer iterations performed
%   sigma2   : residual variance estimate (scalar) (simple i.i.d. noise model)
%   trace    : struct with per-iter diagnostics (z-scores, ΔF, etc.)
%
% Notes:
% - If your original f signature is f(theta, P, M, ...), wrap it to match this.
% - If f cannot return J, we build a numerical J (costlier).
% - “BMR” here uses Laplace-style closed-form ΔF approximations for toggling
%   priors λ_j (on/off). It’s fast and works well as a heuristic gate.

% -------------------- defaults & setup --------------------
if nargin < 9 || isempty(opts), opts = struct(); end
if nargin < 8 || isempty(varpercthresh), varpercthresh = []; end
if nargin < 7 || isempty(plots), plots = 0; end
if nargin < 6 || isempty(tol), tol = 1e-3; end
if nargin < 5 || isempty(maxIter), maxIter = 64; end

y = y(:);
p = numel(m0);
S0 = forceSPD(S0);
S0i = inv(S0);

tau_sched      = getopt(opts,'tau_sched',ones(1,maxIter));
useARD         = getopt(opts,'useARD',false);
lambda         = init_lambda(getopt(opts,'lambda0',1e-3),p);
lambda_max     = getopt(opts,'lambda_max',1e6);
z_thresh       = getopt(opts,'z_thresh',2.5);
kappa_prune    = getopt(opts,'kappa_prune',0);    % prune if ΔF < 0
kappa_add      = getopt(opts,'kappa_add',3);      % add if ΔF > 3 (~BF>~20)
max_add_per_it = getopt(opts,'max_add_per_it',3);
I              = getopt(opts,'I0',(1:p).'); I = I(:);
propose_fun    = getopt(opts,'propose_fun',@(I,p) []);
model_args     = getopt(opts,'model_args',{});

% state
m = m0(:);
V = S0;
sigma2 = 1.0;
logF = -inf(1,maxIter);
trace = struct('I',cell(1,maxIter),'z',[],'dF_prune',[],'dF_add',[],'tau',[]);
best = struct('F',-inf,'m',m,'V',V,'I',I,'lambda',lambda);

% -------------------- outer loop -------------------------
for iter = 1:maxIter
    tau = tau_sched(min(iter, numel(tau_sched)));
    % 1) inference on active set (others clamped to prior)
    [m,V,Fe,sigma2] = vl_inner_step(y, f, m, V, m0, S0i, I, tau, sigma2, model_args);
    logF(iter) = Fe;
    trace(iter).tau = tau;

    % 2) optional ARD shrinkage (acts on all p, but we only update I)
    if useARD
        lambda = ard_update(m, V, lambda, lambda_max);
    end

    % 3) z-score eligibility on all params
    z = abs(m)./sqrt(max(real(diag(V)), eps));
    trace(iter).z = z;

    % 4) PRUNE pass (fast BMR-style ΔF for active params)
    [keepMask, dF_prune] = bmr_prune(m, V, lambda, I, kappa_prune);
    trace(iter).dF_prune = dF_prune;
    I = I(keepMask);

    % 5) PROPOSE/GROW pass (limited number per iter)
    CAND = setdiff(propose_fun(I, p), I);   % user domain-specific proposals
    if isempty(CAND)
        % fall back: any strong z-score outside I
        fallback = find(z > z_thresh & ~ismember((1:p).', I));
        CAND = fallback;
    end
    % score additions (ΔF if we relax prior for j)
    [accMask, dF_add] = bmr_try_add(m, V, lambda, CAND, kappa_add, max_add_per_it);
    trace(iter).dF_add = dF_add;
    I = union(I, CAND(accMask));

    % 6) track best & stopping
    trace(iter).I = I;
    if Fe > best.F
        best = struct('F',Fe,'m',m,'V',V,'I',I,'lambda',lambda);
    end
    if iter > 2 && abs(logF(iter) - logF(iter-1)) < tol
        break
    end
end

% rollback to best if needed
if best.F > logF(iter)
    m = best.m; V = best.V; I = best.I; lambda = best.lambda; logF(iter) = best.F;
end

% outputs & diagnostics
logF = logF(1:iter);
D = struct('active', I, 'lambda', lambda, 'z', trace(iter).z, ...
           'kappa_prune', kappa_prune, 'kappa_add', kappa_add, ...
           'opts', opts);

% --------- nested helpers (kept in-file for portability) ----------
function [m,V,F,s2] = vl_inner_step(y, f, m, V, m0, S0i, I, tau, s2, model_args)
    % One Gauss–Newton (Laplace) step on active set I at temperature tau.
    % Others clamped at prior mean m0.
    p = numel(m);
    I = I(:);
    Ic = setdiff((1:p).', I);
    m(Ic) = m0(Ic);  % clamp inactives to prior mean

    % predictions & Jacobian
    if nargout(f) >= 2
        [yhat, J] = f(m, model_args{:});
    else
        yhat = f(m, model_args{:});
        J = numJacobian(@(th) f(th, model_args{:}), m);
    end
    r = y - yhat(:);
    % restrict to active columns
    JA = J(:, I);
    % Gauss–Newton system (with simple i.i.d. noise variance s2)
    % posterior precision on active block:
    G = S0i(I,I) + (tau/s2) * (JA.'*JA);
    g = (tau/s2) * (JA.'*r) - S0i(I,I) * (m(I) - m0(I));
    % solve SPD
    dtheta = solveSPD(G, g);
    m(I) = m(I) + dtheta;

    % covariance update (active block)
    VA = invSPD(G);
    V = zeros(p,p);                % full V (cheap for moderate p)
    V(I,I) = VA;
    V(Ic,Ic) = invSPD(S0i(Ic,Ic)); % inactives retain prior variance
    % update noise variance (optional, robust)
    s2 = max( eps, (r.'*r) / max(1, numel(y)-numel(I)) );

    % Laplace free energy (approx) at temp tau
    F = free_energy_laplace(y, yhat, r, J, m, m0, S0i, V, s2, tau);
end

function [keepMask, dF] = bmr_prune(m, V, lambda, I, kappa)
    % Score turning OFF each active j via prior precision → ∞
    % Closed-form ΔF approx using Laplace algebra (single-parameter toggle).
    k = numel(I);
    keepMask = true(k,1);
    dF = -inf(k,1);
    for t = 1:k
        j = I(t);
        lj = max(lambda(j), 1e-12);
        vjj = max(real(V(j,j)), 1e-12);
        mj  = m(j);
        % ΔF_off ≈ (Occam penalty − signal)
        % A simple, well-behaved proxy: negative if signal is weak.
        dFj = 0.5*log(1 + lj*vjj) - 0.5*(lj * mj^2)/(1 + lj*vjj);
        % interpret: if dFj < kappa, pruning is favored (ΔF_off < 0 ⇒ remove)
        % For consistency, define ΔF_keep = -dFj (gain of keeping vs pruning)
        keepMask(t) = (-dFj) > kappa;
        dF(t) = -dFj;
    end
end

function [accMask, dF] = bmr_try_add(m, V, lambda, CAND, kappa_add, K)
    if isempty(CAND), accMask = []; dF = []; return; end
    q = numel(CAND);
    dF = -inf(q,1);
    for c = 1:q
        j = CAND(c);
        lj = max(lambda(j), 1e-8);     % prior precision if “on”
        vjj = max(real(V(j,j)), 1e-12);
        mj  = m(j);
        % ΔF_on: gain for freeing param j (inverse of prune formula)
        dF(c) = 0.5*(lj * mj^2)/(1 + lj*vjj) - 0.5*log(1 + lj*vjj);
    end
    % accept top-K above kappa_add
    [~, ord] = sort(dF, 'descend');
    acc = false(q,1);
    picked = 0;
    for idx = ord(:)'
        if dF(idx) > kappa_add && picked < K
            acc(idx) = true; picked = picked + 1;
        end
    end
    accMask = acc;
end

function lambda = ard_update(m, V, lambda, lambda_max)
    % Simple type-II ML update for ARD precisions (RVM-like; safe & bounded)
    % λ_j ← (γ_j) / m_j^2,  γ_j = 1 - λ_j v_jj
    vdiag = max(real(diag(V)), 0);
    gamma = max(0, 1 - lambda(:).*vdiag(:));
    newlam = gamma ./ max(m(:).^2, 1e-12);
    % damp & cap for stability
    alpha = 0.5;
    lambda = (1-alpha)*lambda(:) + alpha*newlam(:);
    lambda = min(lambda, lambda_max);
end

function F = free_energy_laplace(y, yhat, r, J, m, m0, S0i, V, s2, tau)
    % crude but consistent Laplace ELBO proxy (Gaussian obs)
    N = numel(y);
    p = numel(m);
    % likelihood term at temp tau
    L = -0.5*tau*(r.'*r)/max(s2,1e-12) - 0.5*tau*N*log(2*pi*max(s2,1e-12));
    % prior term
    dm = (m - m0);
    P = -0.5*(dm.'*S0i*dm) - 0.5*log((2*pi)^p * max(real(det(invSPD(S0i))),1e-300));
    % complexity (log |H|) with H ≈ S0i + (tau/s2)J'J
    H = S0i + (tau/max(s2,1e-12)) * (J.'*J);
    C = -0.5*safeLogDetSPD(H);
    F = real(L + P + C);
end

function X = solveSPD(A, b)
    % robust SPD solve
    A = 0.5*(A+A.');
    [R,flag] = chol(A);
    if flag==0
        X = R \ (R'\b);
    else
        X = pinv(A)*b; % fallback
    end
end

function Ainv = invSPD(A)
    A = 0.5*(A+A.');
    [R,flag] = chol(A);
    if flag==0
        Ri = inv(R);
        Ainv = Ri*Ri.';
    else
        Ainv = pinv(A);
    end
end

function J = numJacobian(fun, th)
    % finite-diff Jacobian of yhat wrt th
    f0 = fun(th);
    n = numel(f0); p = numel(th);
    J = zeros(n,p);
    eps0 = 1e-4*(1+abs(th));
    parfor j=1:p
        tj = th; tj(j) = th(j) + eps0(j);
        J(:,j) = (fun(tj) - f0) / eps0(j);
    end
end

function S = forceSPD(S)
    S = 0.5*(S+S.');
    [V,D] = eig((S+S')/2);
    d = diag(D);
    d(d<=0) = max(eps, 1e-8);
    S = V*diag(d)*V';
    S = 0.5*(S+S.');
end

function v = safeLogDetSPD(A)
    A = 0.5*(A+A.');
    [R,flag] = chol(A);
    if flag==0
        v = 2*sum(log(diag(R)+eps));
    else
        v = log(max(real(det(A)), 1e-300));
    end
end

function val = getopt(S, field, def)
    if isfield(S, field) && ~isempty(S.(field))
        val = S.(field);
    else
        val = def;
    end
end

function lam = init_lambda(l0, p)
    if isscalar(l0), lam = l0*ones(p,1);
    else, lam = l0(:);
    end
end
end
