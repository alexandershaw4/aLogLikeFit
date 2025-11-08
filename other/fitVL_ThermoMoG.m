function [m, V, D, logL, iter, out] = fitVL_ThermoMoG(y, f, m0, S0, maxIter, tol, opts)
% fitVL_ThermoMoG  Variational Laplace with Mixture-of-Gaussians (MoG) observation model,
% low-rank covariance, heteroscedastic noise, and thermodynamic safeguards.
%
% Observation model:
%   y_i ~ sum_{k=1..K} pi_k * N( f_k(m)_i , sigma2_k(i) )
%   with f_k(m) = f(m) + mu_k  (default; offsets learnable)
%
% Posterior over latent parameters (Laplace):
%   q(z) ~ N(m, S),  S ~ V*V' + diagD
%
% Inputs
%   y        : (n x 1) observed data
%   f        : function handle, f(m) -> (n x 1) model prediction
%   m0, S0   : prior mean and covariance over parameters
%   maxIter  : maximum iterations
%   tol      : convergence tolerance on ||dm||
%   opts     : struct with fields (all optional)
%              .K (default 2)               - number of mixture components
%              .mu0 (n x K)                 - initial component offsets (default = 0)
%              .learn_offsets (bool)        - learn mu_k (default true)
%              .learn_weights (bool)        - learn pi_k (default true)
%              .alpha0 (1 x K)              - Dirichlet prior for pi (default = 1)
%              .share_sigma (bool)          - share sigma2 across components (default false)
%              .sigma2_init (n x 1 or n x K)- initial variances (default = ones)
%              .plots (bool)                - plotting (default true)
%              .solenoidal (bool)           - skew adjustment (default true)
%              .maxStepSize (scalar)        - trust region cap (default 1.0)
%              .gamma_mix (scalar)          - solenoidal scale (default 0.1)
%              .nu (scalar)                 - dof for robust variance update (default 3)
%              .smooth_length (scalar)      - GP-like prior smoothing scale for H_prior (default 2)
%
% Outputs
%   m, V, D  : posterior mean and low-rank covariance factors (S \approx V*V' + D)
%   logL     : final ELBO-like objective
%   iter     : number of iterations performed
%   out      : struct with fields
%              .pi (1xK), .mu (n x K), .sigma2 (n x K or n x 1), .gamma (n x K)
%              .elbo_trace, .like_trace, .prior_trace, .entropy_trace, .m_trace
%
% AS, 2025

if nargin < 7, opts = struct; end
if ~isfield(opts,'K'),            opts.K = 2; end
if ~isfield(opts,'learn_offsets'),opts.learn_offsets = true; end
if ~isfield(opts,'learn_weights'),opts.learn_weights = true; end
if ~isfield(opts,'alpha0'),       opts.alpha0 = ones(1,opts.K); end
if ~isfield(opts,'share_sigma'),  opts.share_sigma = false; end
if ~isfield(opts,'plots'),        opts.plots = true; end
if ~isfield(opts,'solenoidal'),   opts.solenoidal = true; end
if ~isfield(opts,'maxStepSize'),  opts.maxStepSize = 1.0; end
if ~isfield(opts,'gamma_mix'),    opts.gamma_mix = 0.1; end
if ~isfield(opts,'nu'),           opts.nu = 3; end
if ~isfield(opts,'smooth_length'),opts.smooth_length = 2; end

% Initialisation
m = m0(:);
n = numel(y);
K = opts.K;

% Low-rank init from S0
[U,Sv,~] = svd(full(S0),'econ');
eigvals = diag(Sv);
threshold = 0.01 * max(eigvals);
k = sum(eigvals > threshold);
k = max(k, numel(m0));
V = U(:,1:k) * diag(sqrt(eigvals(1:k)));
D = diag(diag(S0) - sum(V.^2,2));

% MoG parameters
if ~isfield(opts,'mu0') || isempty(opts.mu0)
    mu = zeros(n,K);
else
    mu = opts.mu0;
    if size(mu,1)~=n || size(mu,2)~=K, error('mu0 must be (n x K)'); end
end

if ~isfield(opts,'sigma2_init') || isempty(opts.sigma2_init)
    if opts.share_sigma
        sigma2 = ones(n,1);
    else
        sigma2 = ones(n,K);
    end
else
    sigma2 = opts.sigma2_init;
    if opts.share_sigma && ~isequal(size(sigma2),[n,1])
        error('When share_sigma=true, sigma2_init must be (n x 1).');
    end
    if ~opts.share_sigma && ~isequal(size(sigma2),[n,K])
        error('When share_sigma=false, sigma2_init must be (n x K).');
    end
end

% Symmetry-breaking init for offsets (scalar bias per component, broadcast to n)
c = 0.5*std(y(:));
mu_vals0 = linspace(-c, c, K);                 % K distinct biases
mu = repmat(mu_vals0, numel(y), 1);

% mixture weights (simple logits -> softmax)
alpha = zeros(1,K); % logits
pi_k = softmax(alpha);

% traces
m_trace = m(:);
elbo_trace = [];
like_trace = [];
prior_trace = [];
entropy_trace = [];

if opts.plots
    fw = figure('position',[570,659,1740,649]);
end

best_elbo = -inf; m_best = m; V_best = V; D_best = D;

for iter = 1:maxIter
    % Predictions
    f_m = f(m); % (n x 1)

    % Component means
    f_ks = f_m + mu; % (n x K) using offset parameterisation

    % Log component likelihoods per datum
    % Handle shared vs component-specific sigma2
    if opts.share_sigma
        % broadcast sigma2 -> (n x K)
        S2 = repmat(sigma2,1,K);
    else
        S2 = sigma2; % (n x K)
    end

    resid = y(:) - f_ks;            % (n x K)
    log_p_ik = -0.5*(resid.^2 ./ S2) - 0.5*log(2*pi) - 0.5*log(S2);

    % Responsibilities (E-step like)
    log_post_ik = log_p_ik + repmat(log(pi_k),n,1);
    [ll_i, gamma] = logsumexp_rows(log_post_ik); % ll_i: (n x 1), gamma: (n x K)
    
    if ~isfield(opts,'tau'), opts.tau = max(1.5, 1 + 1/log(iter+2)); end % example decay
    log_post_ik = log_p_ik + repmat(log(pi_k), n, 1);
    [ll_i, gamma] = logsumexp_rows(log_post_ik / opts.tau);

    % Likelihood term
    logL_likelihood = sum(ll_i);

    % Robust variance update (heteroscedastic): Student-t style
    nu = opts.nu;
    if opts.share_sigma
        num = sum(gamma .* (resid.^2 + 1e-3), 2);               % (n x 1)
        den = sum(gamma,2) .* (nu + 0.5*mean((resid.^2)./max(S2,1e-12),2));
        den(den==0) = 1; %#ok<NASGU>
        sigma2 = max(1e-6, num ./ max(sum(gamma,2) + nu, 1));   % simple stable update
    else
        sigma2 = max(1e-6, (resid.^2 + 1e-3) ./ (nu + 0.5));    % (n x K)
    end

    % Jacobian: if f_k differ by offsets only, J is shared
    J = computeJacobian_local(f, m, n);

    % Weighted Gauss-Newton terms under mixture
    % H = sum_k J' * W_k * J,  g = sum_k J' * W_k * (y - f_k)
    H = zeros(numel(m));
    g = zeros(numel(m),1);
    for kcomp = 1:K
        Wk = diag(gamma(:,kcomp) ./ S2(:,kcomp));
        rk = y(:) - f_ks(:,kcomp);
        H = H + J' * Wk * J;
        g = g + J' * (Wk * rk);
    end

    % Prior
    H_prior = inv(S0 + computeSmoothCovariance_local(m, opts.smooth_length));
    g = g - H_prior * (m - m0);
    H_elbo = H + H_prior;

    % Low-rank covariance update
    [U,Sv,~] = svd(H_elbo,'econ');
    V = U(:,1:k) * sqrt(Sv(1:k,1:k));
    D = diag(diag(H_elbo) - sum(V.^2,2));

    % Natural gradient / preconditioned step
    dm = safe_solve(H_elbo, g);

    % trust region
    if norm(dm) > opts.maxStepSize
        dm = dm * (opts.maxStepSize / norm(dm));
    end

    % solenoidal mixing
    if opts.solenoidal
        Q = H_elbo - H_elbo';
        dm = dm - opts.gamma_mix * (Q * dm);
    end

    m_prev = m; m = m + dm; m_trace(:,end+1) = m; %#ok<AGROW>

    % Update offsets (M-step like)
    if opts.learn_offsets
        r  = y(:) - f(m);        % residual (n x 1)
        Nk = sum(gamma,1);       % (1 x K)
        Nk(Nk < 1e-12) = 1e-12;

        mu_vals = zeros(1,K);
        for kcomp = 1:K
            mu_vals(kcomp) = (gamma(:,kcomp)' * r) / Nk(kcomp);
        end

        % small ridge to avoid drift
        mu_vals = mu_vals ./ (1 + 1e-2);

        % broadcast scalars to (n x K)
        mu = repmat(mu_vals, numel(y), 1);
    end
    % if opts.learn_offsets
    %     % Weighted residual mean per component: minimize \sum_i gamma_ik (y_i - f(m)_i - mu_ik)^2
    %     % => mu_ik = (\sum_i gamma_ik (y_i - f(m)_i)) / (\sum_i gamma_ik)
    %     gm = sum(gamma,1); gm(gm==0) = 1;
    %     mu = ( (y(:) - f(m)) * ones(1,K) - 0 ) .* 0; % reset
    %     for kcomp = 1:K
    %         mu(:,kcomp) = (gamma(:,kcomp) .* (y(:) - f(m))) ./ max(gamma(:,kcomp),1e-8);
    %     end
    %     % Smooth/shrink offsets: ridge regularisation toward zero
    %     lam = 1e-2; mu = mu ./ (1 + lam);
    % end

    % Update weights
    if opts.learn_weights
        Nk = sum(gamma,1) + (opts.alpha0 - 1); Nk = max(Nk, 1e-9);
        pi_k = Nk / sum(Nk);
        alpha = log(max(pi_k,1e-12)); %#ok<NASGU> % for completeness
    end

    % Entropy-ish term from covariance (diagonal approx for speed)
    logL_entropy = 0.5 * sum(log(diag(D) + 1e-6));
    logL_prior   = -0.5 * ((m - m0)' * H_prior * (m - m0));
    logL = logL_likelihood + logL_prior + logL_entropy;

    % Backtracking if ELBO decreased
    if iter > 1 && logL < elbo_trace(end)
        success = false; stepScale = 0.5; maxTries = 8; attempt = 0; m_trial = m_prev;
        while ~success && attempt < maxTries
            attempt = attempt + 1;
            m_trial = m_prev + stepScale^attempt * dm;
            f_mt = f(m_trial); f_ks_t = f_mt + mu;
            resid_t = y(:) - f_ks_t;
            if opts.share_sigma
                S2t = repmat(sigma2,1,K);
            else
                S2t = sigma2;
            end
            log_p_ik_t = -0.5*(resid_t.^2 ./ S2t) - 0.5*log(2*pi) - 0.5*log(S2t);
            [ll_i_t, gamma_t] = logsumexp_rows(log_p_ik_t + repmat(log(pi_k),n,1));
            Jt = computeJacobian_local(f, m_trial, n);
            Ht = zeros(numel(m)); gt = zeros(numel(m),1);
            for kc = 1:K
                Wk = diag(gamma_t(:,kc) ./ S2t(:,kc));
                rk = y(:) - f_ks_t(:,kc);
                Ht = Ht + Jt' * Wk * Jt; gt = gt + Jt' * (Wk * rk);
            end
            H_prior_t = H_prior; % keep same prior curvature around m0
            gt = gt - H_prior_t * (m_trial - m0);
            logL_t = sum(ll_i_t) - 0.5*((m_trial - m0)'*H_prior_t*(m_trial - m0)) + 0.5*sum(log(diag(Ht)-sum((U(:,1:k)*sqrt(Sv(1:k,1:k))).^2,2)+1e-6)); %#ok<NASGU>
            if logL_t > logL
                success = true; m = m_trial; % accept backtracked step
            end
        end
        if ~success, m = m_prev; end
    end

    entropy_trace(end+1) = logL_entropy; %#ok<AGROW>
    like_trace(end+1)    = logL_likelihood; %#ok<AGROW>
    prior_trace(end+1)   = logL_prior; %#ok<AGROW>
    elbo_trace(end+1)    = logL; %#ok<AGROW>

    if logL > best_elbo
        best_elbo = logL; m_best = m; V_best = V; D_best = D;
    end

    % Show
    if opts.plots
        w = 1:n;
        f_now = f(m);
        fks_now = f_now + mu;
        if opts.share_sigma, S2plot = repmat(sigma2,1,K); else, S2plot = sigma2; end
        [~,gamma_now] = logsumexp_rows(-0.5*((y(:)-fks_now).^2)./S2plot - 0.5*log(2*pi) - 0.5*log(S2plot) + repmat(log(pi_k),n,1));

        gamma_now = gamma;   % use responsibilities from current iteration
        imagesc(gamma_now'); colorbar;

        figure(fw); clf;
        t = tiledlayout(3,4,'TileSpacing','compact','Padding','compact');

        % ---- Row 1: Main fit (span all 4 cols) ----
        nexttile([1 4]); hold on;
        if opts.share_sigma, splot = sqrt(sigma2); else, splot = sqrt(mean(S2plot,2)); end
        errorbar(w, y, splot, 'k.', 'CapSize', 0, 'DisplayName','Observed \pm\sigma');
        plot(w, y, 'k', 'LineWidth', 1, 'DisplayName','Observed');
        plot(w, f_now, '--', 'LineWidth', 1.5, 'DisplayName','f(m)');
        for kc=1:K
            plot(w, fks_now(:,kc), '-', 'LineWidth', 1.5, 'DisplayName',sprintf('f_k (k=%d)',kc));
        end
        hold off; title('Model Fit with MoG components'); xlabel('Index'); ylabel('Value'); legend('Location','best'); grid on; box on;

        % ---- Row 2: four scalar traces ----
        nexttile; plot(1:iter, entropy_trace, 'LineWidth',2); title('Entropy'); grid on; box on;
        nexttile; plot(1:iter, like_trace,    'LineWidth',2); title('Log-Likelihood'); grid on; box on;
        nexttile; plot(1:iter, prior_trace,   'LineWidth',2); title('Log-Prior'); grid on; box on;
        nexttile; plot(1:iter, elbo_trace,    'LineWidth',2); title('ELBO'); grid on; box on;

        % ---- Row 3: Responsibilities heatmap (span all 4 cols) ----
        nexttile([1 4]); imagesc(gamma_now'); colorbar;
        xlabel('i'); ylabel('k'); title('Responsibilities \gamma_{ik}');

        set(gcf,'Color','w'); set(findall(gcf,'-property','FontSize'),'FontSize',14); drawnow;
    end
    % Convergence
    if norm(dm) < tol, break; end
end

% Return best
m = m_best; V = V_best; D = D_best; logL = best_elbo;

% Pack outputs
out = struct();
out.pi = pi_k; out.mu = mu; out.gamma = gamma; out.iter = iter;
out.sigma2 = sigma2; out.elbo_trace = elbo_trace; out.like_trace = like_trace;
out.prior_trace = prior_trace; out.entropy_trace = entropy_trace; out.m_trace = m_trace;

end

% --------- helpers ---------
function [ll_i, gamma] = logsumexp_rows(A)
% A: (n x K)
mx = max(A,[],2);
Z  = exp(A - mx);
Zs = sum(Z,2);
ll_i = mx + log(Zs + 1e-12);
gamma = Z ./ max(Zs,1e-12);
end

function s = softmax(a)
a = a - max(a); ea = exp(a); s = ea / sum(ea);
end

function K = computeSmoothCovariance_local(x, lengthScale)
% RBF kernel over parameter vector (regulariser for H_prior)
x = real(x(:));
K = exp(-pdist2(x,x).^2 / (2*lengthScale^2));
K = K + 1e-6*eye(numel(x));
end

function J = computeJacobian_local(f, x, m)
epsilon = 1e-6; nparam = numel(x); J = zeros(m, nparam);
parfor i = 1:nparam
    x_step  = x; x_stepb = x;
    x_step(i)  = x_step(i)  + epsilon;
    x_stepb(i) = x_stepb(i) - epsilon;
    J(:,i) = (f(x_step) - f(x_stepb)) / (2*epsilon);
end
end

function dm = safe_solve(H, g)
try
    L = chol(H,'lower');
    dm = L' \ (L \ g);
catch
    [dm,flag,relres] = pcg(H + 1e-6*eye(size(H)), g, 1e-6, 200);
    if flag~=0 || relres>1e-2
        dm = zeros(size(g));
    else
        alpha = min(1, 1/(1+norm(dm)));
        dm = alpha*dm;
    end
end
end