function [m, V, D, logL, iter, sigma2, allm,g_elbo] = fitVariationalLaplaceThermoStable(y, f, m0, S0, maxIter, tol,plots)
% Extended Variational Laplace with Low-Rank Approximation, Smarter Variance 
% Updates, and Thermodynamic Integration. Non extended version is fitVariationalLaplace.
%
% Fits dynamical systems models of the form;
%          y = f(m) + e 
%
% This routine estimates a Gaussian variational posterior over the latent 
% variables, refining its mean and covariance structure via a Laplace 
% approximation to a variational free-energy bound (ELBO). The covariance 
% is efficiently represented using a low-rank plus diagonal structure 
% (S ≈ VV^T + D), allowing for tractable optimization even in high dimensions.
%
% Returs the posterior: q(z) ~ N(m, VV' + D)
%
% Key Features:
% - Smarter variance updates: Dynamically adapts observation noise variances (sigma2).
% - Thermodynamic integration: Computes log evidence estimates via annealed variational inference.
% - Low-rank covariance approximation: Captures dependencies without full covariance estimation.
%
% [m, V, D, logL, iter, sigma2] = fitVariationalLaplaceThermo(y, f, m0, S0, maxIter, tol)
%
% Inputs:
%   y        - Observed data (vector)
%   f        - Model function handle (f(x) returns predicted values)
%   m0       - Initial mean for variational distribution (vector)
%   S0       - Initial covariance for variational distribution (matrix)
%   maxIter  - Maximum number of iterations
%   tol      - Convergence tolerance
%
% Outputs:
%   m        - Optimized mean of variational distribution
%   V, D     - Low-rank decomposition of covariance (S ≈ VV^T + D)
%   sigma2   - Optimized observation variances (vector)
%   logL     - Final ELBO value
%   iter     - Number of iterations performed
%   allm     - estimates from each iteration
%
% AS2025

if nargin < 7 || isempty(plots); plots = 1; end

% ---- options ----
thresh         = 1/16;
use_armijo     = 1;
c1_armijo      = 1e-4;
max_backtrack  = 10;

% ---- init ----
m = m0(:);
n = length(y);
d = length(m);
epsilon = 1e-12;

% Fixed prior precision (do NOT depend on m inside line search)
H_prior = inv(S0 + 1e-6*eye(d));

% Keep history
allm = m(:);
allentropy = [];
allloglike = [];
alllogprior = [];
all_elbo = [];

% Best-so-far trackers
best_elbo = -inf;
m_best = m; V_best = []; D_best = [];

% Plot window
if plots, fw = figure('position',[570,659,1740,649]); end

for iter = 1:maxIter
    % === Outer iteration: compute residuals, sigma2, J, H with these fixed for line search ===
    y_pred    = f(m);
    resid     = y - y_pred;

    % robust heteroscedastic sigma2 for THIS iteration (fixed during line search)
    nu = 3; beta = 1e-3;
    sigma2 = max(epsilon, (resid.^2 + beta) ./ (nu + resid.^2/2));

    % build objects at current m
    J = computeJacobian(f, m, n);
    W = diag(1./sigma2);
    H_like = J' * W * J;
    H = H_like + H_prior;

    % step (Cholesky with LM fallback)
    [L, p] = chol(H, 'lower');
    rhs = J'*(W*resid) - H_prior*(m - m0);
    if p ~= 0
        % LM damping until PD
        lambda = 1e-3;
        while p ~= 0
            [L, p] = chol(H + lambda*eye(d), 'lower');
            if p ~= 0, lambda = lambda * 10; end
            if lambda > 1e12
                warning('Failed to PD-ize H; taking zero step.'); 
                L = chol(H + 1e-6*eye(d), 'lower'); rhs = zeros(d,1);
                break;
            end
        end
    end
    dm_newton = L' \ (L \ rhs);

    % Entropy/logdet from H (log|S| = -log|H|)
    logdetH = 2*sum(log(diag(L)));
    logL_entropy = -0.5 * logdetH;
    logL_prior   = -0.5 * (m - m0)' * H_prior * (m - m0);
    logL_like    = -0.5 * sum((resid.^2)./sigma2 + log(2*pi*sigma2));
    elbo0        = logL_like + logL_prior + logL_entropy;
    g_elbo       = rhs; % gradient at m (for Armijo)

    % --- Backtracking line search on a FIXED objective (sigma2, H_prior fixed) ---
    t = 1.0;
    m_prev = m;
    f0 = elbo0;

    if use_armijo
        accepted = false;
        for attempt = 1:max_backtrack
            m_try = m_prev + t*dm_newton;

            % recompute only terms that depend on m (keep W, sigma2, H_prior fixed)
            y_pred_try = f(m_try);
            r_try = y - y_pred_try;
            J_try = computeJacobian(f, m_try, n);
            H_like_try = J_try' * W * J_try;
            H_try = H_like_try + H_prior;

            [Ltry, ptr] = chol(H_try, 'lower');
            if ptr ~= 0
                % not PD -> shrink step
                t = 0.5 * t;
                continue;
            end

            logdetH_try     = 2*sum(log(diag(Ltry)));
            logL_entropy_try= -0.5 * logdetH_try;
            logL_prior_try  = -0.5 * (m_try - m0)' * H_prior * (m_try - m0);
            logL_like_try   = -0.5 * sum((r_try.^2)./sigma2 + log(2*pi*sigma2));
            f_try           = logL_like_try + logL_prior_try + logL_entropy_try;

            % Armijo condition
            if f_try >= f0 + c1_armijo * t * (g_elbo' * dm_newton)
                % accept
                m = m_try;
                elbo = f_try; L = Ltry; H = H_try; y_pred = y_pred_try;
                accepted = true;
                break;
            else
                t = 0.5 * t;
            end
        end
        if ~accepted
            % fallback: very small step or stop
            m = m_prev; elbo = f0;
        end
    else
        % No backtracking: take full step (kept for completeness)
        m = m + dm_newton;
        % recompute H at new m for book-keeping
        y_pred = f(m);
        resid  = y - y_pred;
        J = computeJacobian(f, m, n);
        H_like = J' * W * J;
        H = H_like + H_prior;
        [L, ~] = chol(H, 'lower');
        logdetH = 2*sum(log(diag(L)));
        logL_entropy = -0.5 * logdetH;
        logL_prior   = -0.5 * (m - m0)' * H_prior * (m - m0);
        logL_like    = -0.5 * sum((resid.^2)./sigma2 + log(2*pi*sigma2));
        elbo         = logL_like + logL_prior + logL_entropy;
    end

    % === Low-rank factorization from S = inv(H) ===
    % S via solves (more stable than inv)
    S = L' \ (L \ eye(d));             % S = inv(H)
    S = (S + S')/2;                    % symmetrize
    % Choose rank adaptively from S's spectrum
    [Ue, Se] = eig(S);
    evals = max(epsilon, diag(Se));
    [evals, idx] = sort(evals, 'descend');
    Ue = Ue(:, idx);
    % retain components contributing >1% of top eigenvalue (min rank 1)
    k = max(1, sum(evals > 0.01 * evals(1)));
    k = min(k, d);
    U_k = Ue(:, 1:k);
    V   = U_k * diag(sqrt(evals(1:k)));          % low-rank factor
    D   = max(1e-12, diag(S - V*V.'));           % diagonal correction (clipped >= 0)

    % bookkeeping
    logL = elbo;
    allm        = [allm m(:)];
    allentropy  = [allentropy logL_entropy];
    allloglike  = [allloglike logL_like];
    alllogprior = [alllogprior logL_prior];
    all_elbo    = [all_elbo logL];

    % best-so-far
    if iter == 1 || logL > best_elbo
        best_elbo = logL;
        m_best = m; V_best = V; D_best = D;
    end

    % ---- plotting (compact) ----
    if plots
        w = 1:length(y);
        y_pred_new = y_pred;

        figure(fw); clf;
        tlay = tiledlayout(2,4,'TileSpacing','compact','Padding','compact');

        nexttile([1 4]);
        hold on;
        errorbar(w, y, sqrt(sigma2), 'k.', 'CapSize', 0);
        plot(w, y, 'k', 'LineWidth', 1);
        plot(w, y_pred_new, '-', 'LineWidth', 2);
        plot(w, sqrt(sigma2), '-', 'LineWidth', 1.5);
        hold off;
        title('Model Fit (heteroscedastic \sigma)'); xlabel('Index'); ylabel('Value'); grid on; box on;
        legend({'Observed \pm\sigma','Observed','Prediction','\sigma'},'Location','best');

        nexttile; plot(1:iter, allentropy,'-o'); title('Entropy'); grid on; box on;
        nexttile; plot(1:iter, allloglike,'-o'); title('Log-Likelihood'); grid on; box on;
        nexttile; plot(1:iter, alllogprior,'-o'); title('Log-Prior'); grid on; box on;
        nexttile; plot(1:iter, all_elbo,'-o'); title('ELBO'); grid on; box on;

        set(gcf,'Color','w'); set(findall(gcf,'-property','FontSize'),'FontSize',16);
        drawnow;
    end

    % ---- convergence ----
    if norm(m - m_prev) < tol*(1 + norm(m_prev)) || norm((y - y_pred_new).^2) <= thresh
        fprintf('Converged at iteration %d\n', iter);
        break;
    end

    fprintf('Iter: %d | ELBO: %.6f | step t=%.3g | ||dm||=%.3g\n', ...
        iter, logL, t, norm(m - m_prev));
end

% return best fits
fprintf('Returning best fits...\n');
m    = m_best;
V    = V_best;
D    = D_best;
logL = best_elbo;

end % main


% ---------------- helpers ----------------
function J = computeJacobian(f, x, m)
% Central differences with scale-adaptive epsilon
n = length(x);
J = zeros(m, n);
fx = f(x);                                % cache
for i = 1:n
    h = sqrt(eps) * (1 + abs(x(i)));      % scale-adaptive step
    xp = x; xm = x;
    xp(i) = xp(i) + h; xm(i) = xm(i) - h;
    fp = f(xp); fm = f(xm);
    J(:, i) = (fp - fm) / (2*h);
end
end
