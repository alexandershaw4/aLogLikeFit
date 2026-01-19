function OUT = fitVariationalLaplaceThermoPolyphonic(y, f, m0, S0, OPT)
% Polyphonic Variational Laplace (Thermo-style) for y = f(m) + e
%
% Maintains K coupled Laplace/VL "voices" q_k(m) ~ N(m_k, Sigma_k)
% with non-dominating integration via soft alignment on predicted outcomes.
%
% Inspired by polyphonic inference equations:
%   F_poly = sum_k pi_k F_k + sum_{i<j} lambda_ij C(q_i,q_j)
%   pi update via leaky evidence with floor (no collapse)
%
% INPUTS
%   y    : data vector (n x 1)
%   f    : forward model handle, yhat = f(m)
%   m0   : prior mean (d x 1)
%   S0   : prior covariance (d x d)
%   OPT  : struct of options (all optional)
%
% OUTPUT
%   OUT struct with fields:
%     .voices(k): per-voice state (m, H, L_H, sigma2, elbo, l_accum, phi)
%     .pi       : final credences (K x 1)
%     .trace    : diagnostics over iterations
%     .mix      : mixture moments (moment-matched Gaussian, optional)
%
% DEPENDENCIES (expected to exist on your path, same as ThermoVL):
%   computeJacobian, computeSmoothCovariance, makeposdef
%
% AS-style, 2026

% ------------------------
% defaults
% ------------------------
if nargin < 5, OPT = struct(); end
y = y(:); m0 = m0(:);

n = numel(y);
d = numel(m0);

K            = getOpt(OPT,'K',4);
maxIter      = getOpt(OPT,'maxIter',64);
tol_dm       = getOpt(OPT,'tol_dm',1e-4);
tol_elbo     = getOpt(OPT,'tol_elbo',1e-4);

plots        = getOpt(OPT,'plots',1);

% observation noise update (same flavour as your ThermoVL)
nu           = getOpt(OPT,'nu',3);
epsilon      = getOpt(OPT,'epsilon',1e-6);
beta         = getOpt(OPT,'beta',1e-3);

% low-rank thresholding for initial k (copied vibe)
varpercthresh = getOpt(OPT,'varpercthresh',0.01);

% polyphonic coupling
lambda0      = getOpt(OPT,'lambda0',0.1);      % base coupling strength
lambdaMax    = getOpt(OPT,'lambdaMax',10.0);
eta_couple   = getOpt(OPT,'eta_couple',0.25);  % coupling step size
couple_mode  = getOpt(OPT,'couple_mode','consensus'); % 'consensus' only for now

% credence update (Eq. 20 flavour)
rho          = getOpt(OPT,'rho',0.95);     % leaky accumulator
gamma_pi     = getOpt(OPT,'gamma_pi',1.0); % sharpness
pi_floor     = getOpt(OPT,'pi_floor',0.05);% pluralism guarantee (eps in paper)

% agreement -> diplomacy (optional: scale lambda with disagreement)
use_diplomacy = getOpt(OPT,'use_diplomacy',1);
kappa_agree   = getOpt(OPT,'kappa_agree',1.0);

% feature map ϕ; default = identity on predicted yhat
phi_fun      = getOpt(OPT,'phi_fun',@(yhat) yhat(:));
% You can pass e.g. @(yhat)[bandpower(yhat,fs,[8 12]); ...] etc.

% initialisations for voices (offsets, jitter, custom seeds)
init_jitter  = getOpt(OPT,'init_jitter',0.01);
init_spread  = getOpt(OPT,'init_spread',1.0);
init_modes   = getOpt(OPT,'init_modes','priorjitter'); % 'priorjitter'|'custom'
custom_m0s   = getOpt(OPT,'custom_m0s',[]);            % (d x K) if provided

OPT.couple_warmup = getOpt(OPT,'couple_warmup',0);

% trust region
maxStepSize  = getOpt(OPT,'maxStepSize',1.0);

% ------------------------
% initialise voices
% ------------------------
voices = repmat(struct(),K,1);

% pick a sensible rank for low-rank precision approx (as in your code)
[U0, Sval0, ~] = svd(full(S0), 'econ');
eigvals0 = diag(Sval0);
thr0 = varpercthresh * max(eigvals0);
kRank = sum(eigvals0 > thr0);
kRank = max(kRank, min(d, d));  % keep your “stability” instinct

% priors (precompute a prior precision-ish object via smooth covariance)
% NOTE: your ThermoVL uses inv(S0 + computeSmoothCovariance(m,2)) each step.
% We’ll do the same per voice per step.

% init pi uniform with floor guaranteed later
pi_k = ones(K,1)/K;

for k = 1:K
    if strcmpi(init_modes,'custom') && ~isempty(custom_m0s)
        mk = custom_m0s(:,k);
    else
        mk = m0 + init_spread * init_jitter * randn(d,1);
    end

    % initialize sigma2 as ones
    sigma2k = ones(n,1);

    voices(k).m       = mk;
    voices(k).sigma2  = sigma2k;
    voices(k).elbo    = -Inf;
    voices(k).l_accum = 0;         % leaky evidence accumulator
    voices(k).H       = eye(d);    % precision approx (will update)
    voices(k).L_H     = eye(d);    % chol(H)
    voices(k).phi     = [];        % features
    voices(k).yhat    = [];        % prediction
end

% ------------------------
% trace
% ------------------------
TR.pi   = nan(K, maxIter);
TR.elbo = nan(K, maxIter);
TR.dm   = nan(K, maxIter);
TR.lam  = nan(1, maxIter);
TR.agree= nan(1, maxIter);

if plots
    fw = figure('position',[300,300,1600,800]);
end

% ------------------------
% main polyphonic loop
% ------------------------
lambda = lambda0;

for it = 1:maxIter

    % ---- local VL step per voice (independent) ----
    for k = 1:K
        [voices(k), dm_norm] = oneThermoVLStep(voices(k), y, f, m0, S0, ...
                                              nu, epsilon, beta, maxStepSize, kRank);
        TR.dm(k,it)   = dm_norm;
        TR.elbo(k,it) = voices(k).elbo;
    end

    % ---- compute features and consensus ----
    PHI = [];
    for k = 1:K
        yhatk = voices(k).yhat;
        phik  = phi_fun(yhatk);
        phik  = phik(:);
        voices(k).phi = phik;
        PHI(:,k) = phik; %#ok<AGROW>
    end

    % weighted consensus feature
    phi_bar = PHI * pi_k;

    % agreement index (negative variance across voices)
    % (simple proxy; you can swap in something richer)
    diffs = PHI - phi_bar;
    agree = -mean(sum(diffs.^2,1)); % more negative = more disagreement
    TR.agree(it) = agree;

    % diplomacy: scale coupling with agreement (optional)
    if use_diplomacy
        % when disagreement is large (agree very negative), increase coupling
        %lam_eff = lambda * exp(-kappa_agree * agree);
        %lam_eff = min(lam_eff, lambdaMax);

        lambda_base = lambda0 * min(1, (it-OPT.couple_warmup)/20);  % 20-iter ramp
        lam_eff = min(lambda_base * exp(-kappa_agree*agree), lambdaMax);
    else
        lam_eff = lambda;
    end
    TR.lam(it) = lam_eff;

    if it <= OPT.couple_warmup
        lam_eff = 0;
    end
    
    % ---- coupling correction (soft alignment; non-dominating) ----
    if lam_eff > 0
        for k = 1:K
            mk = voices(k).m;

            % gradient of feature mismatch wrt parameters:
            % phi(m) = ϕ(f(m))
            g_fun = @(x) phi_fun(f(x));  % composed map
            Jphi  = computeJacobian(g_fun, mk, numel(voices(k).phi)); % (p x d)

            % coupling gradient: J' * (phi_k - phi_bar)
            dphi  = voices(k).phi - phi_bar;
            g_cpl = Jphi' * dphi;
            dm_cpl = -(voices(k).H \ g_cpl);

            % apply a small correction (damped)
            %mk_new = mk - eta_couple * lam_eff * g_cpl;

            %voices(k).m = mk_new;

            cpl_max = 0.02;  % small
            nc = norm(dm_cpl);
            if nc > cpl_max
                dm_cpl = dm_cpl * (cpl_max/nc);
            end
            voices(k).m = voices(k).m + eta_couple * lam_eff * dm_cpl;

        end
    end

    % % ---- credence update (leaky evidence, no collapse) ----
    % for k = 1:K
    %     % using -ELBO as “energy”; bigger ELBO => better evidence
    %     voices(k).l_accum = rho * voices(k).l_accum + voices(k).elbo;
    % end
    % 
    % lvec = [voices.l_accum]';
    % % softmax with temperature gamma_pi
    % % z = gamma_pi * (lvec - max(lvec));
    % % pi_raw = exp(z);
    % % pi_raw = pi_raw / sum(pi_raw);
    % % 
    % % % pluralism floor (epsilon in the paper)
    % % pi_k = pi_floor*(1/K) + (1 - pi_floor)*pi_raw;
    % 
    % alpha_plur = 0.3;
    % rho        = 0.995;
    % gamma_pi   = 2.0;
    % 
    % pi_raw = exp(gamma_pi * (lvec - max(lvec)));
    % pi_raw = pi_raw / sum(pi_raw);
    % pi_k = (1 - alpha_plur) * pi_raw + alpha_plur * (1/K);

    % ---- credence update (rank-based, warmup, persistent pluralism) ----
    pi_warmup  = getOpt(OPT,'pi_warmup',10);
    alpha_plur = getOpt(OPT,'alpha_plur',0.3);
    rho_pi     = getOpt(OPT,'rho_pi',0.995);
    gamma_pi   = getOpt(OPT,'gamma_pi',2.0);   % OK with rank evidence

    elbo_vec = arrayfun(@(s) s.elbo, voices)';     % (K x 1)
    [~, ord] = sort(elbo_vec, 'descend');          % ord(1) best
    rank = zeros(K,1); rank(ord) = 1:K;

    % gentle rank score: best=0, worst=-1
    score = -(rank - 1) / max(1,(K - 1));

    for k = 1:K
        voices(k).l_accum = rho_pi * voices(k).l_accum + score(k);
    end

    lvec = [voices.l_accum]';

    if it <= pi_warmup
        pi_k = ones(K,1)/K;
    else
        pi_raw = exp(gamma_pi * (lvec - max(lvec)));
        pi_raw = pi_raw / sum(pi_raw);

        % persistent pluralism (non-dominating)
        pi_k = (1 - alpha_plur) * pi_raw + alpha_plur * (1/K);
    end



    % store
    TR.pi(:,it) = pi_k;

    % ---- plots ----
    if plots
        figure(fw); clf;
        t = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

        % pi
        nexttile;
        plot(1:it, TR.pi(:,1:it)','LineWidth',2);
        title('Credences \pi_k'); xlabel('iter'); ylim([0 1]); grid on;

        % elbos
        nexttile;
        plot(1:it, TR.elbo(:,1:it)','LineWidth',2);
        title('Voice ELBOs'); xlabel('iter'); grid on;

        % lambda + agreement
        nexttile;
        yyaxis left;
        plot(1:it, TR.lam(1:it),'LineWidth',2);
        ylabel('\lambda_{eff}');
        yyaxis right;
        plot(1:it, TR.agree(1:it),'LineWidth',2);
        ylabel('agreement index');
        title('Diplomacy'); xlabel('iter'); grid on;

        % current fits
        nexttile;
        hold on;
        for k = 1:K
            plot(voices(k).yhat,'LineWidth',1.5);
        end
        plot(y,'k','LineWidth',2);
        title('Predictions per voice (and data)'); grid on;
        hold off;

        drawnow;
    end

    % ---- convergence check (polyphonic) ----
    % stop when:
    %   (a) all dm norms small, and
    %   (b) pi has stabilised, and/or elbos stabilised
    if it > 3
        dm_ok   = all(TR.dm(:,it) < tol_dm);
        dpi     = norm(TR.pi(:,it) - TR.pi(:,it-1));
        dpi_ok  = dpi < 1e-3;
        delbo   = max(abs(TR.elbo(:,it) - TR.elbo(:,it-1)));
        delbo_ok= delbo < tol_elbo;

        if dm_ok && (dpi_ok || delbo_ok)
            break;
        end
    end

end

% ------------------------
% assemble outputs
% ------------------------
OUT = struct();
OUT.pi = pi_k;
OUT.trace = TR;
OUT.voices = voices;

% moment-matched mixture (optional; approximate using precisions)
% NOTE: voices store precision H (approx), not covariance. We’ll form cov via solve.
do_mix = getOpt(OPT,'do_moment_match',1);
if do_mix
    % mean: E[m] = sum pi_k m_k
    m_mix = zeros(d,1);
    for k = 1:K, m_mix = m_mix + pi_k(k) * voices(k).m; end

    % covariance: sum pi*(Sigma_k + (m_k - m_mix)(m_k - m_mix)')
    S_mix = zeros(d);
    for k = 1:K
        % Sigma_k = inv(H_k) via triangular solves (avoid full inv if you want later)
        L = voices(k).L_H;
        % inv(H) = inv(L')*inv(L)
        % compute by solving for I (OK for moderate d; you can replace with low-rank later)
        I = eye(d);
        X = L \ I;
        Sigma_k = (L' \ X);

        dm = voices(k).m - m_mix;
        S_mix = S_mix + pi_k(k) * (Sigma_k + dm*dm');
    end

    OUT.mix.m = m_mix;
    OUT.mix.S = S_mix;
end

end

% =====================================================================
% one ThermoVL-like step (based on your fitVariationalLaplaceThermo core)
% =====================================================================
function [st, dm_norm] = oneThermoVLStep(st, y, f, m0, S0, nu, epsilon, beta, maxStepSize, kRank)

m = st.m(:);
n = numel(y);
d = numel(m);

% ---------- forward pass ----------
yhat = f(m);
res  = y - yhat;

% ---------- robust sigma2 update ----------
% Student-t-ish but with a scale floor to stop huge precisions when res ~ 0
res2 = res.^2;
sigma2 = (res2 + beta) ./ max(nu, 1);            % base
sigma2 = max(sigma2, epsilon);                   % absolute floor

% scale-aware floor: prevent absurd weights if signal is tiny
scl = median(sigma2);
sigma2 = max(sigma2, 1e-3 * scl);

W = 1 ./ sigma2;                                 % weights

% ---------- Jacobian ----------
J = computeJacobian(f, m, n);

% ---------- prior precision via solve (no inv) ----------
Sbar = S0 + computeSmoothCovariance(m, 2);

% Make Sbar PD (very important if smooth cov introduces near singularity)
Sbar = makeposdef(Sbar);

% Solve for prior precision action: H_prior * v = Sbar \ v
% We'll never explicitly form inv(Sbar). Use backslash solves.
% To build H_prior term in Hessian: it’s just Sbar^{-1}.
% For modest d, we can form it via chol solves; for large d, keep operator form.
try
    Ls = chol(Sbar, 'lower');
catch
    Sbar = makeposdef(Sbar + eye(d)*1e-8);
    Ls = chol(Sbar, 'lower');
end

% Helper: apply Sbar^{-1} to a vector/matrix
SbarInv = @(X) (Ls' \ (Ls \ X));

% Prior precision matrix (explicit, because we need it in Hessian sum)
H_prior = SbarInv(eye(d));

% ---------- likelihood Hessian ----------
% Use J'*(W.*J) efficiently without diag(W)
JW = J .* W;                 % (n x d), each row weighted
H_like = J' * JW;

% ---------- total Hessian ----------
H = H_like + H_prior;

% ---------- gradient ----------
g_like  = J' * (W .* res);
g_prior = -H_prior * (m - m0);
g = g_like + g_prior;

% ---------- Levenberg–Marquardt damping ----------
% adapt mu so chol succeeds and step is well-behaved
mu = 1e-6 * (trace(H)/d + 1);   % scale-aware start
mu_max = 1e8;

dm = zeros(d,1);
ok = false;

for tries = 1:12
    Hmu = H + mu * eye(d);

    % ensure PD
    try
        L = chol(Hmu, 'lower');
        dm = L' \ (L \ g);
        ok = true;
        break;
    catch
        mu = min(mu * 10, mu_max);
    end
end

if ~ok
    % fallback: very damped gradient step (never explode)
    dm = g / (norm(g) + 1e-8);
    dm = dm * min(maxStepSize, 1e-3);
end

% ---------- trust region ----------
ndm = norm(dm);
if ndm > maxStepSize
    dm = dm * (maxStepSize / ndm);
end

% ---------- update ----------
m_new = m + dm;

% ---------- recompute ELBO proxy ----------
yhat_new = f(m_new);
res_new  = y - yhat_new;

res2n = res_new.^2;
sigma2_new = (res2n + beta) ./ max(nu, 1);
sigma2_new = max(sigma2_new, epsilon);
scln = median(sigma2_new);
sigma2_new = max(sigma2_new, 1e-3 * scln);

logL_like = -0.5 * sum((res2n ./ sigma2_new) + log(2*pi*sigma2_new));

% entropy proxy via logdet(Hmu) from the successful chol
% (use Hmu (damped) rather than raw H)
try
    Lmu = chol(H + mu*eye(d),'lower');
    logdetH = 2*sum(log(diag(Lmu)));
catch
    Hpd = makeposdef(H + mu*eye(d));
    Lmu = chol(Hpd,'lower');
    logdetH = 2*sum(log(diag(Lmu)));
end
logL_ent  = -0.5 * logdetH;

logL_prior = -0.5 * (m_new - m0)' * H_prior * (m_new - m0);

elbo = logL_like + logL_prior + logL_ent;

% ---------- stash ----------
st.m      = m_new;
st.sigma2 = sigma2_new;
st.H      = H + mu*eye(d);     % store damped precision (more stable)
st.L_H    = chol(st.H,'lower');
st.elbo   = elbo;
st.yhat   = yhat_new;

dm_norm = norm(dm);

end

% =====================================================================
% tiny option helper
% =====================================================================
function v = getOpt(S, name, default)
if isfield(S,name) && ~isempty(S.(name))
    v = S.(name);
else
    v = default;
end
end

function K = computeSmoothCovariance(x, lengthScale)
    n = length(x);
    xx = x;


    x = real(x);
    K = exp(-pdist2(x(:), x(:)).^2 / (2 * lengthScale^2));
    K = K + 1e-6 * eye(n); % Regularization for numerical stability

    %x = imag(xx);
    %Kx = exp(-pdist2(x(:), x(:)).^2 / (2 * lengthScale^2));
    %K = Kx + 1e-6 * eye(n); % Regularization for numerical stability


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

