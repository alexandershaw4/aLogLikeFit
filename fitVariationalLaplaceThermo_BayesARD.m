function [m, V, D, logL, iter, sigma2, allm] = ...
    fitVariationalLaplaceThermo_BayesARD(y, f, m0, S0, maxIter, tol, plots, varpercthresh, useARDStep, tau_sched)
% Variational Laplace with a temperature schedule and optional PEB-ARD step.
% Includes: prior-centred line search, prior homotopy, heteroscedastic variance cap,
% ARD warm-up + blending guard, Barzilai–Borwein step scaling, Mahalanobis trust region,
% momentum, and relative-step Jacobian for stability.

if nargin < 7 || isempty(plots),           plots = 1;                 end
if nargin < 8 || isempty(varpercthresh),   varpercthresh = 0.01;      end
if nargin < 9 || isempty(useARDStep),      useARDStep = true;         end
if nargin < 10 || isempty(tau_sched),      tau_sched = ones(1,maxIter); end

thresh = 1/16;                 % residual threshold stop
solenoidalmix = 0;             % optional skew-symmetric mix

% Initialisation
m = m0(:);
n = numel(y);
d = numel(m);

% Low-rank + diag initial posterior approx (kept for API parity)
[U0,Sv0,~] = svd(full(S0), 'econ');
eigvals = diag(Sv0);
k = sum(eigvals > varpercthresh * max(eigvals));
k = max(k, min(d, 1));
V  = U0(:,1:k) * diag(sqrt(max(eigvals(1:k), 1e-12)));
D  = diag(max(diag(S0) - sum(V.^2,2), 1e-10));

epsilon  = 1e-8;
betaHub0 = 1e-3;               % robust variance term (cooled over iters)
nu       = 3;                  % t-like variance update

allm = m(:);
allentropy=[]; allloglike=[]; alllogprior=[]; all_elbo=[];
best_elbo = -inf; m_best = m; V_best = V; D_best = D;
if plots, fw = figure('position',[570,659,1740,649]); end

% State for advanced stepping
dm_prev = zeros(d,1);
g_prev  = zeros(d,1);
mu      = 0;                   % momentum coefficient
delta   = 1.0;                 % trust-region radius (Mahalanobis)
success_streak = 0;
fail_streak    = 0;
sig_cap        = [];           % soft cap for heteroscedastic variance

for iter = 1:maxIter
    tau = tau_sched(min(iter, numel(tau_sched)));

    % --- robust variance cooling ---
    betaHub = betaHub0 * max(0.2, 0.95^(iter-1));   % decays but not to zero

    % ----- model linearisation -----
    y_pred   = f(m);
    residuals = y - y_pred;

    % Heteroscedastic variance with soft cap
    sigma2 = max(epsilon, (residuals.^2 + betaHub) ./ (nu + residuals.^2/2));
    if isempty(sig_cap)
        sig_cap = median(sigma2) * 8;
    else
        sig_cap = max(sig_cap * 0.95, median(sigma2) * 6);
    end
    sigma2 = min(sigma2, sig_cap);

    % Jacobian via relative central differences (scale-aware)
    J = computeJacobian_rel(f, m, n);

    % Likelihood pieces
    logL_lik = -0.5 * sum((residuals.^2 ./ sigma2) + log(2*pi*sigma2));

    % Prior precision with homotopy (let data lead early)
    H_prior_full = inv(S0 + computeSmoothCovariance(m, 2) + 1e-6*eye(d));
    w0 = 0.05; T_prior = 20;
    if iter <= T_prior
        w = w0 + (1 - w0) * (iter-1) / max(T_prior-1,1);
    else
        w = 1.0;
    end
    H_prior = w * H_prior_full;
    g_prior = -H_prior * (m - m0);

    % Gauss–Newton curvature/gradient
    W = 1 ./ sigma2;                         % diag weights (as vector)
    H = J' * bsxfun(@times, J, W);          % J' * diag(W) * J
    g = J' * (W .* residuals) + g_prior;    % gradient of (lik + prior)
    H_elbo = H + H_prior;

    % Diagonal preconditioner (cheap, stabilises norms)
    Hdiag = max(diag(H_elbo), 1e-12);
    Mprec = spdiags(1./Hdiag, 0, d, d);

    % ============================================================
    %  ARD STEP (data-only) with warm-up + blending guard
    % ============================================================
    dm = []; z_cred = [];
    useARD = useARDStep && (iter > 3);   % warm-up GN for 3 iterations

    if useARD
        try
            % Temperature scaling
            srt   = 1/sqrt(max(tau, eps));
            X_ard = J * srt;
            y_ard = residuals * srt;

            optsARD = struct('standardise', true, ...
                             'tie_lambdas', true, ...
                             'max_iter', 500, ...
                             'tol', 1e-6);
            Mstep = peb_ard_novar(y_ard, X_ard, optsARD);

            dm_ard = denan(Mstep.beta_ordered);
            S_std  = Mstep.Vbeta_ordered;
            sc     = (Mstep.y_std(1) ./ Mstep.x_std_ordered(:));
            S_ard  = denan((sc .* S_std) .* sc.');

            % Fuse ARD posterior with prior via linear solve (SPD repair if needed)
            [R,flag] = chol((S_ard+S_ard')/2, 'lower');
            if flag==0
                Ainv_mu = R' \ (R \ dm_ard);
                H_star  = @(v) H_prior*v + (R' \ (R \ v));
                [dm,pcgflag] = pcg(@(v) H_star(v), Ainv_mu, 1e-6, 200, Mprec);
                if pcgflag~=0
                    [dm,~] = pcg(@(v) H_star(v)+1e-6*v, Ainv_mu, 1e-6, 400, Mprec);
                end
            else
                [U,Sv] = svd((S_ard+S_ard')/2, 'econ');
                s      = max(diag(Sv), 1e-12);
                Ainv_mu = U * ((U' * dm_ard) ./ s);
                H_star  = @(v) H_prior*v + U*((U'*v)./s);
                [dm,~]  = pcg(H_star, Ainv_mu, 1e-6, 200, Mprec);
            end

            % Rough credibility (diag approx)
            z_cred = abs(dm) ./ sqrt(max(1./Hdiag,1e-12));

        catch ME
            warning('ARD step failed (%s). Falling back to linear solve.', ME.message);
            useARD = false;
        end
    end

    % Linear GN fallback (and also GN direction for blending)
    try
        L = chol(H_elbo, 'lower');
        dm_lin = L' \ (L \ g);
    catch
        [dm_lin,flag_lin] = pcg(H_elbo + 1e-6*speye(d), g, 1e-6, 200, Mprec);
        if exist('flag_lin','var') && flag_lin~=0
            dm_lin = g ./ (mean(Hdiag)+1e-6);
        end
    end

    % Blend or pick GN if ARD misaligned with gradient
    if isempty(dm) || ~useARD
        dm = dm_lin;
    else
        cg_ard = (g'*dm)     / (norm(g)*max(norm(dm),1e-12));
        cg_lin = (g'*dm_lin) / (norm(g)*max(norm(dm_lin),1e-12));
        if cg_ard < max(-0.1, 0.6*cg_lin)
            dm = dm_lin;
        else
            alpha = 0.5 + 0.5 * tanh(3*(cg_ard - cg_lin));
            dm = alpha*dm + (1-alpha)*dm_lin;
        end
    end

    % Barzilai–Borwein scaling (clamped wider)
    if iter > 1
        s  = dm_prev;
        yk = g - g_prev;
        denom = (s'*yk);
        if abs(denom) > 1e-16
            bb = (s'*s) / denom;
            bb = min(max(bb, 1e-3), 50);   % was 10
            dm = bb * dm;
        end
    end

    % Mahalanobis trust region: dm' H_elbo dm <= delta^2
    qdm = dm' * (H_elbo * dm);
    if ~isfinite(qdm) || qdm <= 0
        qdm = dm' * (spdiags(Hdiag,0,d,d) * dm);
    end
    scaleTR = min(1, delta / sqrt(max(qdm,1e-12)));
    dm = dm * scaleTR;

    % Optional solenoidal mixing
    if solenoidalmix
        Q = H_elbo - H_elbo'; gamma = 0.1;
        dm = dm - gamma * Q * dm;
    end

    % Nesterov-like momentum
    m_prev = m;
    m = m + dm + mu*(m - (m_prev - dm));

    allm = [allm m(:)];

    % ----- ELBO (proxy entropy) -----
    logL_entropy = 0.5 * sum(log(max(diag(H_elbo),1e-12)));
    logL_prior   = -0.5 * (m - m0)' * H_prior * (m - m0);
    logL         = logL_lik + logL_prior + logL_entropy;

    % Backtracking Armijo with correct prior centring
    if iter > 1 && logL < all_elbo(end)
        step = 1.0; c = 1e-4; shrink = 0.5; improved=false;
        dir  = (m - m_prev);
        for tbt = 1:8
            step = step * shrink;
            m_try = m_prev + step * dir;
            L_try = local_ELBO(m_try, m0, f, y, H_prior, betaHub, nu); % centred at m0
            if L_try >= all_elbo(end) + c*step*(g'*dir)
                m = m_try; logL = L_try; improved = true; break;
            end
        end
        if ~improved
            m = m_prev; logL = all_elbo(end);
            fail_streak    = fail_streak + 1; success_streak = 0;
            delta = max(0.5*delta, 1e-3);  % shrink trust-region
            mu    = max(mu*0.5, 0);        % reduce momentum
        else
            success_streak = success_streak + 1; fail_streak = 0;
            delta = min(delta*1.2, 10);    % widen gently
            mu    = min(0.9, 0.4 + 0.05*success_streak);
        end
    else
        % Accepted without backtracking: optionally expand a bit
        if iter > 1 && (logL - all_elbo(end)) > 1e-3
            dir = (m - m_prev);
            m_try = m_prev + 1.2 * dir;
            % quick check (ll+lp only; entropy proxy unchanged)
            r2   = y - f(m_try);
            sig2 = max(1e-8,(r2.^2 + betaHub) ./ (nu + r2.^2/2));
            ll2  = -0.5 * sum((r2.^2 ./ sig2) + log(2*pi*sig2));
            lp2  = -0.5 * (m_try - m0)' * H_prior * (m_try - m0);
            L2   = ll2 + lp2 + 0.5 * sum(log(max(diag(H_elbo),1e-12)));
            if L2 >= logL
                m = m_try; logL = L2;
            end
        end
        success_streak = success_streak + 1; fail_streak = max(fail_streak-1,0);
        delta = min(delta*1.1, 10);
        mu    = min(0.9, 0.4 + 0.05*success_streak);
    end

    % record
    allentropy = [allentropy logL_entropy];
    allloglike = [allloglike logL_lik];
    alllogprior= [alllogprior logL_prior];
    all_elbo   = [all_elbo logL];

    if iter==1 || logL > best_elbo
        best_elbo = logL; m_best = m; V_best = V; D_best = D;
    end

    % quick plots
    if plots
        w = (1:n).';
        y_pred_new = f(m);
        figure(fw); clf
        tiledlayout(2,4, 'TileSpacing','compact','Padding','compact');

        nexttile([1 4]); hold on
        errorbar(w, y, sqrt(sigma2), 'k.', 'CapSize',0);
        plot(w, y, 'k', 'LineWidth',1);
        plot(w, y_pred, '--', 'Color',[0 0.4 1], 'LineWidth',1.5);
        plot(w, y_pred_new, '-', 'Color',[0.8 0 0], 'LineWidth',2);
        plot(w, sqrt(sigma2), '-', 'Color',[0.1 0.6 0.1], 'LineWidth',1.5);
        title(sprintf('VL %s (\\tau=%.3g) | TR=%.2f | mu=%.2f', tern(useARD,'+ ARD',''), tau, delta, mu), 'FontWeight','bold');
        legend({'Obs \pmσ','Obs','Prev pred','Current pred','sqrt(σ²)'}, 'Location','best'); grid on; box on; hold off

        nexttile; plot(1:iter, allentropy, '-o'); title('Entropy'); grid on; box on
        nexttile; plot(1:iter, allloglike, '-o'); title('Log-Likelihood'); grid on; box on
        nexttile; plot(1:iter, alllogprior,'-o'); title('Log-Prior'); grid on; box on
        nexttile; plot(1:iter, all_elbo,  '-o'); title('ELBO'); grid on; box on

        set(gcf,'Color','w'); set(findall(gcf,'-property','FontSize'),'FontSize',14);
        drawnow;
    end

    % ----- convergence -----
    y_pred_new = f(m);
    if norm(m - m_prev) < tol || norm((y - y_pred_new).^2) <= thresh
        fprintf('Converged at iter %d\n', iter);
        break;
    end

    fprintf('Iter %d | ELBO %.4f | ||dm|| %.4g | TR %.2f | mu %.2f | step %s%s\n', ...
        iter, logL, norm(m - m_prev), delta, mu, tern(useARD,'ARD-blend','LIN'), ...
        tern(~isempty(z_cred), sprintf(' | med z=%.2f', median(z_cred,'omitnan')), ''));

    % carry state for BB
    dm_prev = (m - m_prev);
    g_prev  = g;
end

fprintf('Returning best fits...\n');
m     = m_best;
V     = V_best;
D     = D_best;
logL  = best_elbo;

end

% ===================== Helpers =====================

function J = computeJacobian_rel(f, x, m)
% Relative-step central-difference Jacobian of f(x) \in R^m
    n = numel(x);
    J = zeros(m, n);
    fx = f(x);
    parfor i = 1:n
        h = 1e-6 * (1 + abs(x(i)));           % relative step
        xp = x; xm = x; xp(i)=xp(i)+h; xm(i)=xm(i)-h;
        J(:,i) = (f(xp) - f(xm)) / (2*h);
    end
    bad = any(~isfinite(J),1);
    if any(bad)
        for i=find(bad)
            h = 1e-6*(1+abs(x(i)));
            ei = zeros(n,1); ei(i)=1;
            J(:,i) = (f(x + h*ei) - fx)/h;
        end
    end
end

function [L_try] = local_ELBO(m_try, m0, f, y, H_prior, betaHub, nu)
% ELBO surrogate used in line search (ll + prior; entropy proxy omitted)
    epsilon = 1e-8;
    r   = y - f(m_try);
    sig = max(epsilon, (r.^2 + betaHub) ./ (nu + r.^2/2));
    ll  = -0.5 * sum((r.^2 ./ sig) + log(2*pi*sig));
    lp  = -0.5 * (m_try - m0)' * H_prior * (m_try - m0);  % centre at m0
    L_try = ll + lp;
end

function K = computeSmoothCovariance(x, ell)
% Simple RBF prior smoothness on parameters to stabilise prior precision
    if nargin<2, ell=2; end
    x = real(x(:));
    D2 = pdist2(x,x).^2;
    K = exp(-D2/(2*ell^2)) + 1e-6*eye(numel(x));
end

function s = tern(cond,a,b)
    if cond, s=a; else, s=b; end
end

function v = denan(v)
    v(~isfinite(v)) = 0;
end
