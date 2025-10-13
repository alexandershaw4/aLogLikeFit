function [m, V, D, logL, iter, sigma2, allm, g_elbo] = ...
    fitVariationalLaplaceThermo_BayesARD(y, f, m0, S0, maxIter, tol, plots, varpercthresh, useARDStep, tau_sched)
% Variational Laplace with an optional PEB-ARD step (Bayesian Gauss–Newton).
%
% The ARD step solves a linearised update r ≈ J * dθ in a Bayesian way:
%   X = J, y = r  (residual)
% with ARD shrinkage per parameter and predictive uncertainty on dθ.
% We also inject the prior as ridge-like pseudo-observations so that
% curvature from S0 is respected in the update.
%
% Inputs/Outputs mostly match your original; new args:
%   useARDStep (default true)   – use PEB-ARD for step; else fall back
%   tau_sched  (1×maxIter)      – optional annealing schedule (τ), default 1s
%
% Requires: peb_ard_novar.m, peb_ard_predict.m (on path)

if nargin < 7 || isempty(plots),           plots = 1;                 end
if nargin < 8 || isempty(varpercthresh),   varpercthresh = 0.01;      end
if nargin < 9 || isempty(useARDStep),      useARDStep = true;         end
if nargin < 10 || isempty(tau_sched),      tau_sched = ones(1,maxIter); end

thresh = 1/16;
solenoidalmix = 0;

% Initialisation
m = m0(:);
n = numel(y);
d = numel(m);

% Adaptive rank selection for low-rank approx of posterior covariance
[U0,Sv0,~] = svd(full(S0), 'econ');
eigvals = diag(Sv0);
k = sum(eigvals > varpercthresh * max(eigvals));   % keep >% of max eig
k = max(k, min(d, 1));                             % at least 1, at most d

% initial low-rank + diag
V  = U0(:,1:k) * diag(sqrt(max(eigvals(1:k), 1e-12)));
D  = diag(max(diag(S0) - sum(V.^2,2), 1e-10));     % keep PD-ish
sigma2 = ones(n,1);
epsilon = 1e-8;   % tiny floor for numerics
betaHub = 1e-3;   % robust variance term
nu = 3;           % t-like variance update

% bookkeeping
allm = m(:);
allentropy=[]; allloglike=[]; alllogprior=[]; all_elbo=[];
best_elbo = -inf; m_best = m; V_best = V; D_best = D;
if plots, fw = figure('position',[570,659,1740,649]); end

for iter = 1:maxIter
    tau = tau_sched(min(iter, numel(tau_sched)));   % temperature

    % ----- model linearisation -----
    y_pred = f(m);
    residuals = y - y_pred;              % r
    % robust, heteroscedastic variance update
    sigma2 = max(epsilon, (residuals.^2 + betaHub) ./ (nu + residuals.^2/2));

    % Jacobian J (N×d) via central differences
    J = computeJacobian(f, m, n);

    % Likelihood part
    logL_lik = -0.5 * sum((residuals.^2 ./ sigma2) + log(2*pi*sigma2));

    % Prior precision (you had a smoothed S0 option; keep it)
    H_prior = inv(S0 + computeSmoothCovariance(m, 2) + 1e-6*eye(d));
    g_prior = -H_prior * (m - m0);          % gradient from prior

    % Gauss–Newton curvature
    W = 1 ./ sigma2;                         % diag weights
    H = J' * (bsxfun(@times, J, W));        % J' * diag(W) * J
    g = J' * (W .* residuals) + g_prior;    % J' * diag(W) * r + prior grad
    H_elbo = H + H_prior;

    % ============================================================
    %  ARD STEP (data-only): X = J, y = r, then fuse with prior
    % ============================================================
    useARD = useARDStep;
    dm = []; z_cred = [];

    if useARD
        try
            % 1) Temperature scaling: scale residual variance by tau
            srt = 1/sqrt(max(tau, eps));
            X_ard = J * srt;
            y_ard = residuals * srt;

            % 2) Run ARD on DATA ONLY (no prior augmentation!)
            optsARD = struct('standardise', true, ...
                'tie_lambdas', true, ...
                'max_iter', 500, ...
                'tol', 1e-6);
            Mstep = peb_ard_novar(y_ard, X_ard, optsARD);

            % 3) Extract ARD posterior in PARAMETER units
            %    mean:
            dm_ard = Mstep.beta_ordered;                  % d×1 (already mapped to "original" parameter scale)
            dm_ard = denan(dm_ard);
            %    covariance: start from std-space, map to parameter units
            S_std = Mstep.Vbeta_ordered;                  % d×d (std-space)
            sc = (Mstep.y_std(1) ./ Mstep.x_std_ordered(:));    % d×1
            S_ard = (sc .* S_std) .* sc.';        % Σ_ARD in parameter units
            S_ard = denan(S_ard);

            % 4) Fuse with PRIOR: Σ*^{-1} = Σ_ARD^{-1} + H_prior ; μ* = Σ* Σ_ARD^{-1} μ_ARD
            %    Use robust solves (no explicit inv)
            %    Compute A = Σ_ARD^{-1} via chol or SVD-repair
            [R,flag] = chol(S_ard, 'lower');
            if flag==0
                % inv via solves
                Ainv_mu = R' \ (R \ dm_ard);         % Σ_ARD^{-1} μ_ARD
                % Build Σ*^{-1} implicitly as H_star = H_prior + Σ_ARD^{-1}
                % We need to solve H_star * dm = Ainv_mu
                H_star = @(v) H_prior*v + (R' \ (R \ v));
                % Use PCG to get dm
                [dm,pcgflag,pcgres] = pcg(H_star, Ainv_mu, 1e-6, 200);
                if pcgflag~=0
                    % Fallback: add small ridge
                    [dm,pcgflag,pcgres] = pcg(@(v) H_star(v)+1e-6*v, Ainv_mu, 1e-6, 400);
                end
                % Posterior covariance Σ*: not needed for update, but we want z
                % Get diagonal approx via probing (cheap): e_i basis
                zdiag = zeros(d,1);
                for kprobe = 1:min(d, 16)        % 16 probes is enough for z range; increase if you like
                    e = randn(d,1); e = e/norm(e);
                    w = R' \ (R \ e);            % Σ_ARD^{-1} e
                    v = pcg(H_star, w, 1e-6, 200); % (H_star)^{-1} (Σ_ARD^{-1} e) ≈ Σ* e
                    zdiag = zdiag + (v.^2);
                end
                zdiag = d * zdiag / min(d,16);   % rough diag(Σ*)
                step_sd = sqrt(max(zdiag, 1e-12));
            else
                % S_ard not SPD -> SVD repair
                [U,Sv] = svd((S_ard+S_ard')/2, 'econ');
                s = max(diag(Sv), 1e-12);
                % Σ_ARD^{-1} = U diag(1/s) U'
                Ainv_mu = U * ((U' * dm_ard) ./ s);
                H_star_mv = @(v) H_prior*v + U * ((U' * v) ./ s);
                [dm,pcgflag,pcgres] = pcg(H_star_mv, Ainv_mu, 1e-6, 200);
                % posterior sd (rough): use same probing trick
                zdiag = zeros(d,1);
                for kprobe = 1:min(d, 16)
                    e = randn(d,1); e = e/norm(e);
                    w = U * ((U' * e) ./ s);
                    v = pcg(H_star_mv, w, 1e-6, 200);
                    zdiag = zdiag + (v.^2);
                end
                zdiag = d * zdiag / min(d,16);
                step_sd = sqrt(max(zdiag, 1e-12));
            end

            % 5) Coordinate-wise credibility for logging / optional gating
            z_cred = abs(dm) ./ max(step_sd, 1e-12);

            % 6) Trust region & monotone line search
            maxStep = 0.3;                        % smaller, gentler steps
            nrm = norm(dm);
            if nrm > maxStep, dm = dm * (maxStep / nrm); end

            % quick one-step backtracking on the ELBO surrogate
            m_try = m + dm;
            y_try = f(m_try);
            r_try = y - y_try;
            sig_try = max(epsilon, (r_try.^2 + 1e-3) ./ (nu + r_try.^2/2));
            J_try = computeJacobian(f, m_try, n);
            H_try = J_try' * (bsxfun(@times, J_try, 1./sig_try));
            L_try = -0.5 * sum((r_try.^2 ./ sig_try) + log(2*pi*sig_try)) ...
                -0.5 * (m_try - m0)' * H_prior * (m_try - m0) ...
                + 0.5 * sum(log(max(diag(H_try + H_prior),1e-12))); % crude entropy proxy

            if iter>1 && L_try < logL
                dm = 0.5 * dm;   % halve once; keep it simple
            end

        catch ME
            warning('ARD step failed (%s). Falling back to linear solve.', ME.message);
            useARD = false;
        end
    end


    % Fallback: your original Cholesky/PCG solve if ARD not used
    if ~useARD
        try
            L = chol(H_elbo, 'lower');
            dm = L' \ (L \ g);
        catch
            [dm,flag,relres] = pcg(H_elbo + 1e-6*eye(d), g, 1e-6, 200);
            if flag~=0
                warning('PCG fallback had flag=%d, relres=%.2e. Damping step.', flag, relres);
                dm = g / (trace(H_elbo)/d + 1e-6);
            end
        end
        % trust region
        maxStep = 1.0;
        nrm = norm(dm);
        if nrm > maxStep, dm = dm * (maxStep / nrm); end
    end

    % Optional solenoidal mixing
    if solenoidalmix
        Q = H_elbo - H_elbo'; gamma = 0.1;
        dm = dm - gamma * Q * dm;
    end

    % ----- propose update -----
    m_prev = m;
    m = m + dm;
    allm = [allm m(:)];

    % ----- ELBO pieces (approximate entropy term) -----
    % Entropy ≈ 0.5*log|Σ|; we approximate with inverse(H_elbo)
    logL_entropy = 0.5 * sum(log(max(diag(H_elbo),1e-12)));  % crude but monotone proxy
    logL_prior    = -0.5 * (m - m0)' * H_prior * (m - m0);
    logL          = logL_lik + logL_prior + logL_entropy;

    % Backtrack if ELBO decreased
    if iter > 1 && logL < all_elbo(end)
        stepScale = 0.5; maxTries = 8;
        improved = false;
        for t = 1:maxTries
            m_try = m_prev + (stepScale^t) * dm;
            y_try = f(m_try);
            r_try = y - y_try;
            sig_try = max(epsilon, (r_try.^2 + betaHub) ./ (nu + r_try.^2/2));
            J_try = computeJacobian(f, m_try, n);
            H_try = J_try' * (bsxfun(@times, J_try, 1./sig_try));
            H_elbo_try = H_try + H_prior;

            ll_try  = -0.5 * sum((r_try.^2 ./ sig_try) + log(2*pi*sig_try));
            ent_try = 0.5 * sum(log(max(diag(H_elbo_try),1e-12)));
            lp_try  = -0.5 * (m_try - m0)' * H_prior * (m_try - m0);
            L_try   = ll_try + ent_try + lp_try;

            if L_try > logL
                m = m_try; logL = L_try; improved = true; break;
            end
        end
        if ~improved
            m = m_prev; logL = all_elbo(end);
        end
    end

    % record
    allentropy = [allentropy logL_entropy];
    allloglike = [allloglike logL_lik];
    alllogprior= [alllogprior logL_prior];
    all_elbo   = [all_elbo logL];

    if iter==1 || logL > best_elbo
        best_elbo = logL; m_best = m; V_best = V; D_best = D;
    end

    % ----- quick plots -----
    if plots
        w = (1:n).';
        y_pred_new = f(m);
        figure(fw); clf
        t = tiledlayout(2,4, 'TileSpacing','compact','Padding','compact');

        nexttile([1 4]); hold on
        errorbar(w, y, sqrt(sigma2), 'k.', 'CapSize',0, 'DisplayName','Observed ±σ');
        plot(w, y, 'k', 'LineWidth',1, 'DisplayName','Observed mean');
        plot(w, y_pred, '--', 'Color',[0 0.4 1], 'LineWidth',1.5, 'DisplayName','Prev pred');
        plot(w, y_pred_new, '-', 'Color',[0.8 0 0], 'LineWidth',2, 'DisplayName','Current pred');
        plot(w, sqrt(sigma2), '-', 'Color',[0.1 0.6 0.1], 'LineWidth',1.5, 'DisplayName','sqrt(σ^2)');
        title(sprintf('VL with %s step (\\tau=%.3g)', tern(useARD,'PEB-ARD','linear'), tau), 'FontWeight','bold');
        legend('Location','best'); grid on; box on; hold off

        nexttile; plot(1:iter, allentropy, '-o'); title('Entropy'); grid on; box on
        nexttile; plot(1:iter, allloglike, '-o'); title('Log-Likelihood'); grid on; box on
        nexttile; plot(1:iter, alllogprior,'-o'); title('Log-Prior'); grid on; box on
        nexttile; plot(1:iter, all_elbo,  '-o'); title('ELBO'); grid on; box on

        set(gcf,'Color','w'); set(findall(gcf,'-property','FontSize'),'FontSize',14);
        drawnow;
    end

    % ----- convergence -----
    y_pred_new = f(m);
    if norm(dm) < tol || norm((y - y_pred_new).^2) <= thresh
        fprintf('Converged at iter %d\n', iter);
        break;
    end

    fprintf('Iter %d | ELBO %.4f | ||dm|| %.4g | step via %s%s\n', ...
        iter, logL, norm(dm), tern(useARD,'ARD','LIN'), ...
        tern(~isempty(z_cred), sprintf(' | med z=%.2f', median(z_cred,'omitnan')), ''));
end

% return best fits
fprintf('Returning best fits...\n');
m     = m_best;
V     = V_best;
D     = D_best;
logL  = best_elbo;

end

function J = computeJacobian(f, x, m)
% central-difference Jacobian of f(x) \in R^m
    epsc = 1e-6;
    n = numel(x);
    J = zeros(m, n);
    fx = f(x);
    parfor i = 1:n
        xp = x; xm = x;
        xp(i) = xp(i) + epsc;
        xm(i) = xm(i) - epsc;
        J(:,i) = (f(xp) - f(xm)) / (2*epsc);
    end
    % fallback if any NaNs
    bad = any(~isfinite(J),1);
    if any(bad)
        J(:,bad) = repmat((fx - f(x - epsc*eye(n,1)))/epsc, 1, sum(bad));
    end
end

function K = computeSmoothCovariance(x, ell)
% simple RBF on parameters to stabilise prior precision
    if nargin<2, ell=2; end
    x = real(x(:));
    D2 = pdist2(x,x).^2;
    K = exp(-D2/(2*ell^2)) + 1e-6*eye(numel(x));
end

function s = tern(cond,a,b)
    if cond, s=a; else, s=b; end
end
