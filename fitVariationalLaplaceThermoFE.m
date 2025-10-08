function [m, V, D, F, iter, sigma2, allm, g_elbo, L_terms] = fitVariationalLaplaceThermoFE( ...
        y, f, m0, S0, maxIter, tol, plots, varpercthresh)
% SPM-style FE with explicit L(3) hyper block (splat basis) and stable backtracking.

if nargin < 7 || isempty(plots),          plots = 1;        end
if nargin < 8 || isempty(varpercthresh),  varpercthresh = 0.01; end

thresh = 1/16; solenoidalmix = 0;

% --- init
m = m0(:);
n = length(y);

% adaptive rank for initial S0
[U0, Sval0, ~] = svd(full(S0), 'econ');
eigvals0 = diag(Sval0);
threshold = varpercthresh * max(eigvals0);
k = sum(eigvals0 > threshold); k = max(k, length(m0));
V  = U0(:, 1:k) * diag(sqrt(eigvals0(1:k)));
D  = diag(diag(S0) - sum(V.^2, 2));      % init cov approx

epsilon  = 1e-6; beta = 1e-3; nu = 3;

% --- hyperparameterised noise via 1D Gaussian "splat" basis ---
use_hyper       = true;
use_splat_hyper = true;

if use_splat_hyper
    tvec      = (1:n)';                         % 1-D coordinate (index/time/freq)
    K0        = min(64, max(8, round(n/16)));
    opts_splat = struct( ...
        'centres_mode','quantiles', ...
        'sigma_mode','silverman', ...
        'sigma',[], ...
        'add_bias',true, ...
        'ard_alpha0',1e-2, ...
        'prune',true, ...            % prune only after accepted step
        'prune_thresh',1e-3, ...
        'max_inner',8, ...
        'tol_inner',1e-6 ...
    );
    [B, centres, sigma_rbf] = build_splat_basis(tvec, K0, opts_splat); %#ok<ASGLU>
    q = size(B,2);
    a0 = zeros(q,1);
    Ca0 = eye(q) * 1e2;     % weak prior; ARD will tighten
    ihC_a = inv(Ca0);
    logdet_ihC_a = logdet_pd(ihC_a);
    a = a0;
else
    % simple global precision (kept here for completeness; unused when splat on)
    W_h = ones(n,1); q = 1; %#ok<NASGU>
    h0   = zeros(q,1); %#ok<NASGU>
    Ch0  = eye(q) * 1e2; %#ok<NASGU>
end

% logs
allm = m(:);
all_F = []; all_L1 = []; all_L2 = []; all_L3 = []; L_terms = [];

if plots, fw = figure('position',[570,659,1740,649]); end

% prior precision and logdet
[RS0, pS0] = chol(S0, 'lower');
if pS0>0, RS0 = chol(makeposdef(S0), 'lower'); end
iPC = RS0'\(RS0\eye(length(m0)));
logdetS0  = 2*sum(log(diag(RS0)));
logdet_ipC = -logdetS0;

best_F = inf; m_best = m; V_best = V; D_best = D;

% --------- initial residuals & baseline noise / iS ---------
y_pred    = f(m);
residuals = y - y_pred;
s2_base = median(residuals.^2);
s2_min  = max(1e-10, s2_base/50);
s2_max  = max(s2_min*1.0001, s2_base*50);
sigma2  = min(max((residuals.^2 + beta) ./ (nu + residuals.^2/2), s2_min), s2_max);
iS_strict = spdiags(1./sigma2, 0, n, n);

% tempering schedule
warm_iters = 2; tau0=0.25; tau1=1.0;
iS_loose  = speye(n) / max(s2_base, 1e-12);
tau = tau0; iS = (1-tau)*iS_loose + tau*iS_strict;

for iter = 1:maxIter

    c1 = 1e-4;                 % Armijo slope fraction
    eps_accept = 1e-3; 

    % smooth prior precision at current m (freeze within this iter)
    Ksm        = computeSmoothCovariance(m, 2);
    [RL,fl]    = chol(S0 + Ksm + 1e-8*eye(numel(m)), 'lower');
    if fl, RL = chol(makeposdef(S0 + Ksm), 'lower'); end
    Hpr_fixed  = RL'\(RL\eye(numel(m)));

    % Jacobian & GN Hessian with tempered iS
    J = computeJacobian(f, m, n);
    H   = J' * iS * J;
    H_elbo = H + Hpr_fixed;

    % gradient
    g_elbo = J' * (iS * residuals) - Hpr_fixed * (m - m0);

    % low-rank precision factorisation
    [U, Sval, ~] = svd(H_elbo, 'econ');
    V = U(:,1:k) * sqrt(Sval(1:k,1:k));
    dvec = max(diag(H_elbo) - sum(V.^2,2), 1e-12);
    D    = spdiags(dvec,0,numel(m),numel(m));

    % step solve
    try
        Lchol = chol(H_elbo, 'lower');
        dm = Lchol' \ (Lchol \ g_elbo);
    catch
        fprintf('Cholesky failed, using PCG.\n');
        [dm,flag,relres] = pcg(H_elbo + 1e-6*speye(size(H_elbo)), g_elbo, 1e-6, 200);
        if flag ~= 0 || relres > 1e-2
            fprintf('PCG unreliable (relres=%.2e). Falling back to damped Cholesky.\n', relres);
            Lchol = chol(makeposdef(H_elbo), 'lower');
            dm = Lchol' \ (Lchol \ g_elbo);
        end
        alpha = min(1, 1/(1+norm(dm)));
        dm = alpha * dm;
    end

    % trust region
    maxStepSize = 1.0;
    if norm(dm) > maxStepSize, dm = dm * (maxStepSize / norm(dm)); end

    % optional solenoidal mixing
    if solenoidalmix
        Q = H_elbo - H_elbo';
        gamma = 0.1;
        dm = dm - gamma * Q * dm;
    end

    % proposed step
    m_prev = m;
    F_curr = inf; L3 = 0;       % L3 is not considered during line search
    % current logdetH at base point (for reporting only)
    try
        Ltmp = chol(H_elbo,'lower');
        logdetH = 2*sum(log(diag(Ltmp)));
    catch
        Ltmp = chol(makeposdef(H_elbo),'lower');
        logdetH = 2*sum(log(diag(Ltmp)));
    end
    p = m - m0;
    [L1, L2, ~, ~] = spmFreeEnergyTerms(residuals, diag(iS_strict).^(-1), p, iPC, logdet_ipC, logdetH); %#ok<*ASGLU>
    F_curr = -(L1 + L2);  % no L3 in line search baseline

    gF = -g_elbo;              % gradient of the surrogate F wrt m
    phi0 = F_curr;             % current surrogate value

    % ---------------- BACKTRACK (fixed sigma2/iS and Hpr_fixed) ----------------
    success   = true;            % try full step first

    try
        Ltmp = chol(H_elbo,'lower');
        logdetH = 2*sum(log(diag(Ltmp)));
    catch
        Ltmp = chol(makeposdef(H_elbo),'lower');
        logdetH = 2*sum(log(diag(Ltmp)));
    end
    p = m - m0;
    [L1, L2, ~, ~] = spmFreeEnergyTerms(residuals, diag(iS_strict).^(-1), p, iPC, logdet_ipC, logdetH);
    F_curr = -(L1 + L2);
    gF = -g_elbo;

    m_trial   = m_prev + dm;
    y_pred_t  = f(m_trial);
    r_t       = y - y_pred_t;
    J_t       = computeJacobian(f, m_trial, n);
    H_t       = J_t' * iS * J_t + Hpr_fixed;
    Lt        = chol(makeposdef(H_t), 'lower');
    logdetH_t = 2*sum(log(diag(Lt)));
    p_t       = m_trial - m0;
    [L1t, L2t, ~, ~] = spmFreeEnergyTerms(r_t, diag(iS_strict).^(-1), p_t, iPC, logdet_ipC, logdetH_t);
    F_t       = -(L1t + L2t);    % no L3 in line search

    if F_t > F_curr
        fprintf('F increased. Backtracking...\n');
        success = false;
        stepScale = 0.5; maxTries = 10; attempt = 0;
        while ~success && attempt < maxTries
            attempt = attempt + 1;
            scale   = stepScale^attempt;
            m_trial = m_prev + scale * dm;

            y_pred_t  = f(m_trial);
            r_t       = y - y_pred_t;
            J_t       = computeJacobian(f, m_trial, n);
            H_t       = J_t' * iS * J_t + Hpr_fixed;
            Lt        = chol(makeposdef(H_t), 'lower');
            logdetH_t = 2*sum(log(diag(Lt)));
            p_t       = m_trial - m0;
            [L1t, L2t, ~, ~] = spmFreeEnergyTerms(r_t, diag(iS_strict).^(-1), p_t, iPC, logdet_ipC, logdetH_t);
            F_t       = -(L1t + L2t);

            fprintf('  Attempt %d: scale=%.4f | F=%.6f\n', attempt, scale, F_t);
            %if F_t <= F_curr + 1e-12
            %    success = true;
            %end
            armijo_ok = F_t <= F_curr + c1 * scale * (gF' * dm);

            % Also accept if we're "numerically flat" (tiny absolute increase)
            flat_ok   = F_t <= F_curr + eps_accept * max(1, abs(F_curr));

            if armijo_ok || flat_ok || attempt == maxTries
                success = true;
            end
        end
    end

    if success
        m         = m_trial;
        residuals = y - f(m);
        % ---------------- AFTER ACCEPT: update hyper ONCE ----------------
        if use_hyper
            if use_splat_hyper
                % do one hyper update; allow pruning here
                [a, Ca, L3, sigma2, ihC_a, B] = updateHyper_L3_splat( ...
                    residuals, B, a, a0, ihC_a, logdet_ihC_a, opts_splat); %#ok<NASGU,ASGLU>
                logdet_ihC_a = logdet_pd(ihC_a);
            else
                % (kept for completeness)
                % [h, Ch, L3, sigma2] = updateHyper_L3(residuals, W_h, h, h0, ihC, logdet_ihC);
                error('Non-splat hyper path not enabled.');
            end
            % clamp and rebuild precisions for *next* iteration
            sigma2 = min(max(sigma2, s2_min), s2_max);
            iS_strict = spdiags(1./sigma2, 0, n, n);
        else
            L3 = 0;
        end

        % refresh tempered iS for next iteration
        tau = tau0 + (tau1 - tau0) * min(1, (iter)/max(1,warm_iters));
        iS  = (1-tau)*iS_loose + tau*iS_strict;

        % recompute FE terms with NEW sigma2 and current m for logging
        J_log    = computeJacobian(f, m, n);
        H_log    = J_log' * iS * J_log + Hpr_fixed;
        Lch_log  = chol(makeposdef(H_log),'lower');
        logdetH  = 2*sum(log(diag(Lch_log)));
        p        = m - m0;
        [L1, L2, ~, ~] = spmFreeEnergyTerms(residuals, diag(iS_strict).^(-1), p, iPC, logdet_ipC, logdetH);
        F_curr   = -(L1 + L2 + L3);
    else
        % revert
        m = m_prev;
        % no hyper update; FE stays at previous value
    end

    % keep best
    if F_curr < best_F
        best_F = F_curr; m_best = m; V_best = V; D_best = D;
    end

    % logging & plots
    allm = [allm m(:)];
    all_F  = [all_F  F_curr];
    all_L1 = [all_L1 L1];
    all_L2 = [all_L2 L2];
    all_L3 = [all_L3 L3];
    L_terms(iter).L1 = L1; %#ok<AGROW>
    L_terms(iter).L2 = L2;
    L_terms(iter).L3 = L3;
    L_terms(iter).F  = F_curr;

    if plots
        figure(fw); clf;
        t = tiledlayout(2,4,'TileSpacing','compact','Padding','compact'); %#ok<NASGU>
        nexttile([1 4]);
        w = 1:length(y);
        y_pred_new = f(m);
        errorbar(w, y, sqrt(diag(iS_strict).^(-1)), 'k.', 'CapSize', 0, 'DisplayName','Observed ±σ'); hold on
        plot(w, y, 'k', 'LineWidth',1, 'DisplayName','Observed mean');
        plot(w, y_pred, '--', 'LineWidth',1.5, 'DisplayName','Prev pred');
        plot(w, y_pred_new, '-', 'LineWidth',2, 'DisplayName','Curr pred');
        plot(w, sqrt(diag(iS_strict).^(-1)), '-', 'LineWidth',1.5, 'DisplayName','σ (hetero)');
        hold off; grid on; box on; legend('Location','best');
        title('Model fit (heteroscedastic)');

        nexttile; plot(1:iter, all_L1, '.-'); grid on; title('L1 (data)');
        nexttile; plot(1:iter, all_L2, '.-'); grid on; title('L2 (params)');
        nexttile; plot(1:iter, all_L3, '.-'); grid on; title('L3 (hyper)');
        nexttile; plot(1:iter, all_F,  '.-'); grid on; title('F (SPM-style)');

        set(gcf,'Color','w'); set(findall(gcf,'-property','FontSize'),'FontSize',18);
        drawnow;
    end

    % update prev preds for plot
    y_pred = y_pred_new;

    % convergence
    if norm(dm) < tol || mean((y - f(m)).^2) <= thresh
        fprintf('Converged at iteration %d (F=%.6f)\n', iter, best_F);
        break;
    end

    fprintf('Iter %d | F: %.6f | ||dm||: %.4f\n', iter, F_curr, norm(dm));
end

% return best
m = m_best; V = V_best; D = D_best; F = best_F;

end

% ----------------- helpers -----------------

function [L1, L2, L3, F] = spmFreeEnergyTerms(residuals, sigma2, p, iPC, logdet_ipC, logdetH)
% L1, L2 as in SPM; L3 handled elsewhere.
logdet_iS = -sum(log(sigma2));                 % diag noise ⇒ log|iS|
quad_e    = sum((residuals.^2)./sigma2);       % e'iS e
L1 = 0.5*( logdet_iS - quad_e );

logdet_Cp = -logdetH;                          % log|Cp| = -log|H|
quad_p    = p' * (iPC * p);
L2 = 0.5*( logdet_ipC + logdet_Cp ) - 0.5*quad_p;

L3 = 0;
F  = -(L1 + L2 + L3);
end

function A = makeposdef(A)
A = (A + A')/2;
[V,D] = eig(A);
d = diag(D); d(d<1e-10) = 1e-10;
A = V*diag(d)*V'; A = (A + A')/2;
end

function J = computeJacobian(f, x, m)
epsilon = 1e-6;
n = length(x);
J = zeros(m, n);
parfor i = 1:n
    x_f = x; x_b = x;
    x_f(i) = x_f(i) + epsilon;
    x_b(i) = x_b(i) - epsilon;
    J(:, i) = (f(x_f) - f(x_b)) / (2 * epsilon);
end
end

function K = computeSmoothCovariance(x, lengthScale)
x = real(x(:));
K = exp(-pdist2(x, x).^2 / (2 * lengthScale^2));
K = K + 1e-6 * eye(numel(x));
end

function [B, centres, sigma_rbf] = build_splat_basis(x, K, opts)
x = x(:); n = numel(x);
xmin = min(x); xmax = max(x);
if strcmpi(opts.centres_mode,'quantiles')
    qs = linspace(0,1,K+2); qs = qs(2:end-1);
    centres = quantile(x, qs);
else
    centres = linspace(xmin, xmax, K);
end
dx = median(diff(sort(x)));
if isempty(opts.sigma) || strcmpi(opts.sigma_mode,'silverman')
    sigma_rbf = 1.06 * std(x) * n^(-1/5);
    if ~isfinite(sigma_rbf) || sigma_rbf<=0
        sigma_rbf = max(dx, (xmax-xmin)/max(K,1))/2;
    end
else
    sigma_rbf = opts.sigma;
end
B = zeros(n, K);
for j = 1:K
    B(:,j) = exp( -((x - centres(j)).^2) / (2*sigma_rbf^2) );
end
if isfield(opts,'add_bias') && opts.add_bias
    B = [ones(n,1) B];
end
end

function [a, Ca, L3, sigma2, ihC_a, B] = updateHyper_L3_splat(e, B, a, a0, ihC_a, logdet_ihC_a, opts)
% Laplace update for eta = B*a, lambda=exp(eta), sigma2=exp(-eta).
max_inner = opts.max_inner; tol_inner = opts.tol_inner;

% ---- shape guards ----
[n, qB] = size(B); %#ok<ASGLU>
if numel(a)  ~= qB, a  = [a(:); zeros(qB-numel(a),1)]; a  = a(1:qB); end
if numel(a0) ~= qB, a0 = [a0(:); zeros(qB-numel(a0),1)]; a0 = a0(1:qB); end
if ~ismatrix(ihC_a) || any(size(ihC_a) ~= [qB qB]), ihC_a = eye(qB) * 1e-2; end

% Newton/Laplace
for k = 1:max_inner
    eta = B*a; eta = max(min(eta, 30), -30);
    lam = exp(eta);
    g_eta = 0.5*(1 - (e.^2).*lam);
    w     = -0.5*(e.^2).*lam;

    g_a   = B.' * g_eta - ihC_a*(a - a0);
    H_a   = B.' * (B .* w) - ihC_a;

    [Lh,flag] = safe_chol(-(H_a));
    if flag
        H_a = H_a - 1e-8*eye(size(H_a));
        [Lh,flag] = safe_chol(-(H_a));
    end
    da = -(Lh'\(Lh\g_a));
    alpha = 1.0;
    a_try = a + alpha*da;
    for bt=1:6
        if isfinite_obj_eta(e, B*a_try, a_try, a0, ihC_a), break; end
        alpha = alpha/2; a_try = a + alpha*da;
    end
    if norm(alpha*da) < tol_inner, a = a_try; break; end
    a = a_try;
end

% Laplace covariance and (optional) pruning
eta = B*a; eta = max(min(eta, 30), -30);
lam = exp(eta);
w   = -0.5*(e.^2).*lam;
H_a = B.' * (B .* w) - ihC_a;
Ca  = inv_pd(-(H_a));

if isfield(opts,'prune') && opts.prune && isfield(opts,'prune_thresh') && opts.prune_thresh>0
    sd  = sqrt(max(eps, diag(Ca)));
    z   = abs(a) ./ max(sd, 1e-12);
    keep = z > opts.prune_thresh;
    if all(B(:,1)==1), keep(1) = true; end
    if sum(keep) < numel(keep)
        B     = B(:, keep);
        a     = a(keep);
        Ca    = Ca(keep, keep);
        alpha_mean = mean(diag(ihC_a));
        ihC_a = eye(sum(keep)) * alpha_mean;
        a0    = a0(keep);
    end
end

% simple ARD re-estimation
if isfield(opts,'ard_alpha0') && opts.ard_alpha0>0
    post_power = a.^2 + max(eps, diag(Ca));
    alpha_new  = (1 ./ max(post_power, 1e-16)) * opts.ard_alpha0;
    ihC_a      = diag(alpha_new);
end

% SPM-style L3
logdet_ihC_a = logdet_pd(ihC_a); %#ok<NASGU>
logdet_Ca    = logdet_pd(Ca);
d            = (a - a0);
L3           = 0.5*(logdet_ihC_a + logdet_Ca) - 0.5*(d'*(ihC_a*d));

% implied variance
sigma2 = exp(-B*a);
sigma2 = min(max(sigma2, 1e-12), 1e6);
end

function ok = isfinite_obj_eta(e, eta, a, a0, ihC)
lam = exp(eta);
L1  = 0.5*( sum(eta) - sum((e.^2).*lam) );
d   = a - a0;
Lh  = L1 - 0.5*(d'*(ihC*d));
ok  = isfinite(Lh);
end

function [L,flag] = safe_chol(A)
flag = 0;
try
    L = chol(A,'lower');
catch
    try
        L = chol(makeposdef(A),'lower');
    catch
        flag = 1; L = [];
    end
end
end

function v = logdet_pd(A)
L = chol(makeposdef(A),'lower');
v = 2*sum(log(diag(L)));
end

function Ainv = inv_pd(A)
L = chol(makeposdef(A),'lower');
Ainv = L'\(L\eye(size(A)));
end
