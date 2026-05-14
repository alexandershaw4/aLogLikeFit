function OUT = fitVariationalLaplaceThermoPolyphonic_MO(y, f, m0, S0, OPT)
% Polyphonic ThermoVL — Multi-Objective (4-voice) version
%
% You now have *one* fitting problem, but 4 concurrently-active objectives:
%   1) Reflex / sensorimotor  : safety + step stabilisation + hard constraints
%   2) Planner / mapper       : classic ThermoVL ELBO (data fit + prior + entropy)
%   3) Explorer / epistemic   : uncertainty reduction / identifiability pressure
%   4) Social / rules         : normative constraints (sparsity, sign, bounds, etc.)
%
% Rather than minimising a weighted sum, we:
%   - compute 4 update proposals dm_k each iteration
%   - integrate them using bounded polyphonic control weights pi_ctrl
%   - update credences using leaky evidence from *multi-objective deltas*
%   - enforce non-domination via a floor
%
% Notes:
% - This is a *controller* around VL. Obj2 uses a ThermoVL step.
% - Obj1/3/4 are expressed as penalties + gradient/proposal heuristics.
% - You can swap in your own objectives with OPT.obj(1..4).(fun/stepper/veto).
%
% AS, 2026 (MO polyphonic rewrite)

% ------------------------
% defaults
% ------------------------
if nargin < 5, OPT = struct(); end
y = y(:); m0 = m0(:);

n = numel(y);
d = numel(m0);

maxIter      = getOpt(OPT,'maxIter',64);
tol_dm       = getOpt(OPT,'tol_dm',1e-4);
tol_pi       = getOpt(OPT,'tol_pi',1e-3);
tol_obj      = getOpt(OPT,'tol_obj',1e-4);
plots        = getOpt(OPT,'plots',1);

% ThermoVL noise update params (as in your routine)
nu           = getOpt(OPT,'nu',3);
epsilon      = getOpt(OPT,'epsilon',1e-6);
beta         = getOpt(OPT,'beta',1e-3);

% trust region / step control
maxStepSize  = getOpt(OPT,'maxStepSize',1.0);
maxStepMO    = getOpt(OPT,'maxStepMO',maxStepSize);
ls_max_tries = getOpt(OPT,'ls_max_tries',8);
ls_shrink    = getOpt(OPT,'ls_shrink',0.5);

% optional hard bounds on parameters
lb           = getOpt(OPT,'lb',-Inf(d,1));
ub           = getOpt(OPT,'ub', Inf(d,1));
use_bounds   = getOpt(OPT,'use_bounds',~all(isinf(lb)) || ~all(isinf(ub)));
bound_kappa  = getOpt(OPT,'bound_kappa',100.0);  % barrier weight (Obj1)

% "social/rules" penalty (Obj4): user can pass a rule_fun returning [R,gR]
% default: soft L1-ish shrink + optional sign constraints
rule_fun     = getOpt(OPT,'rule_fun',[]);
rule_w       = getOpt(OPT,'rule_w',1e-2);
l1_scale     = getOpt(OPT,'l1_scale',1.0);       % for default rule
sign_mask    = getOpt(OPT,'sign_mask',zeros(d,1)); % -1 => enforce negative, +1 => positive, 0 => none
sign_kappa   = getOpt(OPT,'sign_kappa',50.0);

% polyphonic weight dynamics
K            = 4;
rho          = getOpt(OPT,'rho',0.95);      % leaky accumulator
gamma_pi     = getOpt(OPT,'gamma_pi',1.5);  % softmax sharpness
pi_floor     = getOpt(OPT,'pi_floor',0.20); % keep all 4 alive
alpha_ctrl   = getOpt(OPT,'alpha_ctrl',0.70); % credence-control decoupling
pi_warmup    = getOpt(OPT,'pi_warmup',5);

% multi-objective evidence shaping (penalise harm)
w_self       = getOpt(OPT,'w_self',1.0);
w_harm       = getOpt(OPT,'w_harm',0.5);
harm_tol     = getOpt(OPT,'harm_tol',0.0);    % allow small worsening without punishment
reflex_veto  = getOpt(OPT,'reflex_veto',1);   % Obj1 can veto steps if violated

% explorer strength
w_explore    = getOpt(OPT,'w_explore',0.25);  % how much dm3 magnitude relative to dm2

% planner vs rule curvature usage
use_Hprior_in_rule = getOpt(OPT,'use_Hprior_in_rule',1);

% make S0 PD and pre-factorise
S0pd = makeposdef(S0);
try
    L0 = chol(S0pd,'lower');
catch
    S0pd = makeposdef(S0pd + 1e-8*eye(d));
    L0 = chol(S0pd,'lower');
end

% init state
st = struct();
st.m      = m0;
st.m_prev = m0;
st.sigma2 = ones(n,1);

voices = repmat(struct(),K,1);
names = {'reflex','planner','explorer','social'};
for k=1:K
    voices(k).name    = names{k};
    voices(k).l_accum = 0;
    voices(k).F       = Inf;   % objective value
    voices(k).dF      = 0;     % objective delta
end

pi_k     = ones(K,1)/K;
pi_ctrl  = ones(K,1)/K;

% trace
TR.pi      = nan(K,maxIter);
TR.pi_ctrl = nan(K,maxIter);
TR.F       = nan(K,maxIter);
TR.dF      = nan(K,maxIter);
TR.dm      = nan(1,maxIter);
TR.step    = nan(1,maxIter);
TR.elbo    = nan(1,maxIter);

if plots
    fw = figure('position',[200 200 1600 800]);
end

% ------------------------
% main loop
% ------------------------
for it = 1:maxIter

    m  = st.m;

    % ---------- compute "planner" (Obj2) quantities at current m ----------
    % We compute J, res, W, H, g once and reuse for Obj3/Obj4 proposals.
    Q = local_thermo_quantities(y, f, m, m0, S0pd, nu, epsilon, beta);

    % objective values at current point
    F1 = obj_reflex_value(m, st.m_prev, lb, ub, use_bounds, bound_kappa);
    F2 = obj_planner_value(Q.elbo);                        % = -ELBO
    F3 = obj_explorer_value(Q.Sigma);                      % logdet(Sigma)
    [F4, gR] = obj_social_value(m, rule_fun, rule_w, l1_scale, sign_mask, sign_kappa);

    Fvec = [F1; F2; F3; F4];
    for k=1:K, voices(k).F = Fvec(k); end

    % ---------- propose 4 updates dm_k ----------
    dm1 = obj_reflex_step(m, st.m_prev, lb, ub, use_bounds);                 % safety projection / stabiliser
    dm2 = obj_planner_step(Q, maxStepMO);                                     % ThermoVL step (GN/LM)
    dm3 = obj_explorer_step(Q, dm2, w_explore, maxStepMO);                    % epistemic projection
    dm4 = obj_social_step(Q, gR, use_Hprior_in_rule, maxStepMO);              % rule-driven correction

    DM = [dm1, dm2, dm3, dm4];

    % integrated multi-objective step (bounded, non-dominating control weights)
    dm = DM * pi_ctrl;

    % trust region (global)
    ndm = norm(dm);
    if ndm > maxStepMO
        dm = dm * (maxStepMO / (ndm + 1e-12));
    end

    % ---------- polyphonic line search / acceptance ----------
    step = 1.0;
    accepted = false;

    for ls = 1:ls_max_tries

        m_try = m + step*dm;

        % enforce bounds if requested (hard clip; reflex still tracks violation)
        if use_bounds
            m_try = min(max(m_try, lb), ub);
        end

        % evaluate objectives at candidate
        Qn = local_thermo_quantities(y, f, m_try, m0, S0pd, nu, epsilon, beta);

        %F1n = obj_reflex_value(m_try, m, lb, ub, use_bounds, bound_kappa);

        [F1_step , F1_bound , F1 ] = obj_reflex_value(m,     st.m_prev, lb, ub, use_bounds, bound_kappa);
        [F1_stepn, F1_boundn, F1n] = obj_reflex_value(m_try, m,         lb, ub, use_bounds, bound_kappa);



        F2n = obj_planner_value(Qn.elbo);
        F3n = obj_explorer_value(Qn.Sigma);
        [F4n, ~] = obj_social_value(m_try, rule_fun, rule_w, l1_scale, sign_mask, sign_kappa);

        Fvec = [F1;  F2;  F3;  F4];
        Fnew = [F1n; F2n; F3n; F4n];
        dF   = Fnew - Fvec;

        % reflex veto ONLY on safety (bounds/constraints), not on "you moved"
        if reflex_veto && ((F1_boundn - F1_bound) > 0)
            step = step * ls_shrink;
            continue;
        end

        % Fnew = [F1n; F2n; F3n; F4n];
        % dF   = Fnew - Fvec;  % negative is improvement
        % 
        % % Reflex veto: if reflex worsens a lot (bounds/step), reject
        % if reflex_veto && (dF(1) > 0.0)
        %     step = step * ls_shrink;
        %     continue;
        % end

        % Pareto-ish accept rule:
        % - accept if not "too harmful" to most objectives
        harm = max(0, dF - harm_tol);
        nBad = sum(harm > 0);
        hasGain = any(dF < -tol_obj);

        if (nBad <= 2) && hasGain
            accepted = true;
            break;
        end

        step = step * ls_shrink;
    end

    if ~accepted
        % very conservative fallback: take reflex step only (stabilise)
        m_try = m + 0.1*dm1;
        if use_bounds, m_try = min(max(m_try, lb), ub); end
        Qn = local_thermo_quantities(y, f, m_try, m0, S0pd, nu, epsilon, beta);

        F1n = obj_reflex_value(m_try, m, lb, ub, use_bounds, bound_kappa);
        F2n = obj_planner_value(Qn.elbo);
        F3n = obj_explorer_value(Qn.Sigma);
        [F4n, ~] = obj_social_value(m_try, rule_fun, rule_w, l1_scale, sign_mask, sign_kappa);

        Fnew = [F1n; F2n; F3n; F4n];
        dF   = Fnew - Fvec;
        step = 0.1;
    end

    % ---------- update state ----------
    st.m_prev = m;
    st.m      = m_try;
    st.sigma2 = Qn.sigma2; % carry forward latest (mostly for convenience)

    % ---------- evidence update (keep all voices alive) ----------
    % Reward: improve own objective; penalise harm to others
    % Convert objective deltas into per-voice scalar evidence increments s_k.
    for k=1:K
        self_gain = -dF(k); % improvement => positive
        harm_others = sum(max(0, dF([1:k-1, k+1:K]) - harm_tol));
        s_k = w_self*self_gain - w_harm*harm_others;
        voices(k).l_accum = rho * voices(k).l_accum + s_k;
        voices(k).dF = dF(k);
    end

    lvec = [voices.l_accum]';

    if it <= pi_warmup
        pi_k = ones(K,1)/K;
    else
        pi_raw = exp(gamma_pi * (lvec - max(lvec)));
        pi_raw = pi_raw / sum(pi_raw);
        pi_k = pi_floor*(1/K) + (1 - pi_floor)*pi_raw;  % non-dominating
    end

    pi_ctrl = alpha_ctrl*pi_k + (1-alpha_ctrl)*(ones(K,1)/K);

    % ---------- trace ----------
    TR.pi(:,it)      = pi_k;
    TR.pi_ctrl(:,it) = pi_ctrl;
    TR.F(:,it)       = Fvec;
    TR.dF(:,it)      = dF;
    TR.dm(it)        = norm(step*dm);
    TR.step(it)      = step;
    TR.elbo(it)      = Qn.elbo;

    % ---------- plots ----------
    if plots
        figure(fw); clf;
        tlo = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

        nexttile;
        plot(1:it, TR.pi(:,1:it)','LineWidth',2); ylim([0 1]); grid on;
        title('\pi_k (credence, non-dominating)'); xlabel('iter');

        nexttile;
        plot(1:it, TR.pi_ctrl(:,1:it)','LineWidth',2); ylim([0 1]); grid on;
        title('\pi^{ctrl}_k (control weights)'); xlabel('iter');

        nexttile;
        plot(1:it, TR.F(:,1:it)','LineWidth',2); grid on;
        title('Objectives F_k (lower better)'); xlabel('iter');

        nexttile;
        plot(1:it, TR.dF(:,1:it)','LineWidth',2); yline(0,'k-'); grid on;
        title('\Delta F_k per accepted step'); xlabel('iter');

        nexttile;
        plot(1:it, TR.dm(1:it),'LineWidth',2); grid on;
        title('||\Delta m||'); xlabel('iter');

        nexttile;
        plot(1:it, TR.elbo(1:it),'LineWidth',2); grid on;
        title('Planner ELBO (Obj2)'); xlabel('iter');

        drawnow;
    end

    % ---------- convergence ----------
    if it > 3
        dm_ok  = TR.dm(it) < tol_dm;
        dpi_ok = norm(TR.pi(:,it) - TR.pi(:,it-1)) < tol_pi;
        dF_ok  = max(abs(TR.dF(:,it))) < tol_obj;
        if dm_ok && (dpi_ok || dF_ok)
            break;
        end
    end
end

% ------------------------
% assemble outputs
% ------------------------
OUT = struct();
OUT.m        = st.m;
OUT.pi       = pi_k;
OUT.pi_ctrl  = pi_ctrl;
OUT.voices   = voices;
OUT.trace    = TR;

% Provide mixture-like view of the 4 update directions last iteration
OUT.last.DM = DM;
OUT.last.dm = dm;

% Provide final planner posterior approx
Qf = local_thermo_quantities(y, f, st.m, m0, S0pd, nu, epsilon, beta);
OUT.planner.elbo  = Qf.elbo;
OUT.planner.H     = Qf.H;
OUT.planner.Sigma = Qf.Sigma;
OUT.planner.yhat  = Qf.yhat;
OUT.planner.sigma2= Qf.sigma2;

end

% =====================================================================
% Objective definitions (defaults)
% =====================================================================

function [F_step, F_bound, F] = obj_reflex_value(m, m_prev, lb, ub, use_bounds, kappa)
dm = m - m_prev;
F_step  = (dm'*dm);

F_bound = 0;
if use_bounds
    vL = max(0, lb - m);
    vU = max(0, m - ub);
    F_bound = kappa * (sum(vL.^2) + sum(vU.^2));
end

F = F_step + F_bound;
end

function dm1 = obj_reflex_step(m, m_prev, lb, ub, use_bounds)
% Conservative stabiliser: pull back toward previous, plus projection direction
dm1 = -0.25*(m - m_prev);

if use_bounds
    % nudge back inside bounds smoothly
    vL = max(0, lb - m);
    vU = max(0, m - ub);
    dm1 = dm1 + 0.5*(vL - vU);
end
end

function F = obj_planner_value(elbo)
% We minimise F2 = -ELBO
F = -elbo;
end

function dm2 = obj_planner_step(Q, maxStep)
% Use the ThermoVL GN/LM step already computed
dm2 = Q.dm;
nd = norm(dm2);
if nd > maxStep
    dm2 = dm2 * (maxStep/(nd+1e-12));
end
end

function F = obj_explorer_value(Sigma)
% Epistemic: reduce posterior uncertainty volume
% (minimise logdet(Sigma))
try
    L = chol(makeposdef(Sigma),'lower');
    F = 2*sum(log(diag(L)));
catch
    % fallback: trace if chol fails
    F = trace(Sigma);
end
end

function dm3 = obj_explorer_step(Q, dm2, w_explore, maxStep)
% Explorer proposes: move in *uncertain* directions while remaining aligned
% with the planner gradient (avoid random drift).
%
% Implementation:
%   - take principal direction of Sigma (largest variance)
%   - project planner step into that direction
%   - scale it (w_explore)
Sigma = Q.Sigma;
d = size(Sigma,1);

dm3 = zeros(d,1);
try
    % top-eigvec of Sigma
    [V,D] = eigs((Sigma+Sigma')/2, 1, 'largestreal');
    v = V(:,1);
    dm_proj = v * (v' * dm2);
    dm3 = -w_explore * dm_proj; % negative because we want to reduce uncertainty
catch
    % fallback: diagonal weighting
    s = diag(Sigma);
    if all(isfinite(s)) && any(s>0)
        w = s / (mean(s)+1e-12);
        dm3 = -w_explore * (w .* dm2);
    else
        dm3 = zeros(d,1);
    end
end

nd = norm(dm3);
if nd > maxStep
    dm3 = dm3 * (maxStep/(nd+1e-12));
end
end

function [F4, gR] = obj_social_value(m, rule_fun, rule_w, l1_scale, sign_mask, sign_kappa)
% Social/rules objective: user-defined, else default soft L1 + sign penalties.
d = numel(m);

if ~isempty(rule_fun)
    % user-supplied: should return [R,gR] or just R (then we finite-diff gR)
    try
        [R, gR] = rule_fun(m);
        if isempty(gR), error('empty gradient'); end
    catch
        R = rule_fun(m);
        gR = local_fd_grad(rule_fun, m);
    end
    F4 = rule_w * R;
    gR = rule_w * gR(:);
    return;
end

% default rule: soft-L1 shrinkage
% R = sum sqrt(m_i^2 + a^2)
a = 1e-3 * l1_scale;
R_l1 = sum(sqrt(m.^2 + a^2));
g_l1 = m ./ sqrt(m.^2 + a^2);

% optional sign constraints:
% sign_mask(i)=+1 => enforce m(i) >= 0 ; -1 => enforce m(i) <= 0
R_sign = 0;
g_sign = zeros(d,1);
pos_idx = find(sign_mask > 0);
neg_idx = find(sign_mask < 0);
if ~isempty(pos_idx)
    v = max(0, -m(pos_idx));
    R_sign = R_sign + sum(v.^2);
    g_sign(pos_idx) = g_sign(pos_idx) + (-2)*v;
end
if ~isempty(neg_idx)
    v = max(0,  m(neg_idx));
    R_sign = R_sign + sum(v.^2);
    g_sign(neg_idx) = g_sign(neg_idx) + ( 2)*v;
end

R = R_l1 + sign_kappa*R_sign;
gR = g_l1 + sign_kappa*g_sign;

F4 = rule_w * R;
gR = rule_w * gR(:);
end

function dm4 = obj_social_step(Q, gR, use_Hprior, maxStep)
% Turn rule gradient into a conservative correction using prior/planner curvature.
H = Q.H;
Hprior = Q.H_prior;

if use_Hprior
    Hc = Hprior + 1e-6*eye(size(Hprior));
else
    Hc = H + 1e-6*eye(size(H));
end

% Solve Hc * dm = -gR
dm4 = - (Hc \ gR);

nd = norm(dm4);
if nd > maxStep
    dm4 = dm4 * (maxStep/(nd+1e-12));
end
end

% =====================================================================
% Thermo quantities (refactor of your oneThermoVLStep core)
% Returns step dm as well as H, Sigma, elbo, etc., at *current* m.
% =====================================================================
function Q = local_thermo_quantities(y, f, m, m0, S0pd, nu, epsilon, beta)

yhat = f(m);
res  = y - yhat;

% robust sigma2 update
res2 = res.^2;
sigma2 = (res2 + beta) ./ max(nu, 1);
sigma2 = max(sigma2, epsilon);
scl = median(sigma2);
sigma2 = max(sigma2, 1e-3 * scl);
W = 1 ./ sigma2;

n = numel(y);
d = numel(m);

J = computeJacobian(f, m, n);

% prior precision from Sbar = S0 + smoothcov
Sbar = S0pd + computeSmoothCovariance(m, 2);
Sbar = makeposdef(Sbar);
try
    Ls = chol(Sbar, 'lower');
catch
    Sbar = makeposdef(Sbar + eye(d)*1e-8);
    Ls = chol(Sbar, 'lower');
end
SbarInv = @(X) (Ls' \ (Ls \ X));
H_prior = SbarInv(eye(d));

% Hessian and gradient
JW = J .* W;
H_like = J' * JW;

H = H_like + H_prior;
g_like  = J' * (W .* res);
g_prior = -H_prior * (m - m0);
g = g_like + g_prior;

% LM damping to get dm
mu = 1e-6 * (trace(H)/d + 1);
mu_max = 1e8;
dm = zeros(d,1);
ok = false;

for tries = 1:12
    Hmu = H + mu*eye(d);
    try
        L = chol(Hmu,'lower');
        dm = L' \ (L \ g);
        ok = true;
        break;
    catch
        mu = min(mu*10, mu_max);
    end
end

if ~ok
    dm = g / (norm(g) + 1e-12);
    dm = 1e-3 * dm;
end

% posterior covariance approx (Sigma = inv(Hmu))
Hpd = (H + mu*eye(d));
Hpd = (Hpd + Hpd')/2;
Hpd = makeposdef(Hpd);
Lh = chol(Hpd,'lower');
I = eye(d);
X = Lh \ I;
Sigma = (Lh' \ X);

% ELBO proxy computed at current point (same flavour as your step code)
logL_like = -0.5 * sum((res2 ./ sigma2) + log(2*pi*sigma2));
logdetH   = 2*sum(log(diag(Lh)));
logL_ent  = -0.5 * logdetH;
logL_prior= -0.5 * (m - m0)' * H_prior * (m - m0);
elbo = logL_like + logL_prior + logL_ent;

Q = struct();
Q.yhat   = yhat;
Q.res    = res;
Q.J      = J;
Q.W      = W;
Q.H_like = H_like;
Q.H_prior= H_prior;
Q.H      = Hpd;
Q.Sigma  = Sigma;
Q.g      = g;
Q.dm     = dm;
Q.mu     = mu;
Q.elbo   = elbo;
Q.sigma2 = sigma2;
end

% =====================================================================
% helpers
% =====================================================================
function v = getOpt(S, name, default)
if isfield(S,name) && ~isempty(S.(name))
    v = S.(name);
else
    v = default;
end
end

function g = local_fd_grad(fun, x)
eps = 1e-6;
x = x(:);
d = numel(x);
g = zeros(d,1);
fx = fun(x);
for i=1:d
    xp = x; xm = x;
    xp(i) = xp(i) + eps;
    xm(i) = xm(i) - eps;
    g(i) = (fun(xp) - fun(xm)) / (2*eps);
end
if ~isfinite(fx), fx = 0; end %#ok<NASGU>
end

% ---- your existing helper functions (unchanged) ----

function K = computeSmoothCovariance(x, lengthScale)
n = length(x);
x = real(x);
K = exp(-pdist2(x(:), x(:)).^2 / (2 * lengthScale^2));
K = K + 1e-6 * eye(n);
end

function J = computeJacobian(f, x, m)
epsilon = 1e-6;
n = length(x);
J = zeros(m, n);
parfor i = 1:n
    x_step = x;
    x_stepb = x;
    x_step(i) = x_step(i) + epsilon;
    x_stepb(i) = x_stepb(i) - epsilon;
    J(:, i) = (f(x_step) - f(x_stepb)) / (2 * epsilon);
end
end

function A = makeposdef(A)
% light-touch PSD -> PD repair (keeps your vibe)
A = (A + A')/2;
[V,D] = eig(full(A));
d = real(diag(D));
scale = max(1, max(abs(d)));
floorv = 1e-8 * scale;
d = max(d, floorv);
A = V*diag(d)*V';
A = (A + A')/2;
end
