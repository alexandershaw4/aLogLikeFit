function [m, V, D, logL, iter, sigma2, allm, g_elbo, active_log] = ...
    fitVariationalLaplaceThermo_active( ...
        y, f, m0, S0, maxIter, tol, plots, varpercthresh, ...
        effect_mode, selection, sel_param, reeval_every, lambda_soft)

% Variational Laplace with "active-set" updates: only move parameters with big effects.
% - effect_mode: 'zstep' (default) | 'grad' | 'zpost'
% - selection:   'topK' | 'threshold'
% - sel_param:   K (for 'topK') OR tau (for 'threshold')
% - reeval_every: re-evaluate the active set every N iters (default 1 = every iter)
% - lambda_soft: optional lasso-like soft-threshold toward prior mean (default 0 = off)
%
% Returns active_log.active_idx history and effect scores per iteration.

if nargin < 9  || isempty(effect_mode),  effect_mode  = 'zstep'; end
if nargin < 10 || isempty(selection),    selection    = 'topK';  end
if nargin < 11 || isempty(sel_param),    sel_param    = 25;      end % K=25 default
if nargin < 12 || isempty(reeval_every), reeval_every = 1;       end
if nargin < 13 || isempty(lambda_soft),  lambda_soft  = 0;       end

% --- call the base routine structure but inline its core loop so we can gate updates
thresh = 1/32;              % your convergence criterion
solenoidalmix = 0;
m = m0(:); n = length(y);

% Low-rank init from S0 (same as your code)
[U0,Sval0,~] = svd(full(S0),'econ');
eigvals = diag(Sval0);
threshold = (nargin>=8 && ~isempty(varpercthresh)) * varpercthresh; 
if isempty(threshold) || threshold==0, threshold = 0.01; end
k = sum(eigvals > threshold*max(eigvals));
k = max(k, length(m0)); 
V  = U0(:,1:k) * diag(sqrt(eigvals(1:k)));
D  = diag(diag(S0) - sum(V.^2,2));

sigma2   = ones(n,1);
epsilon  = 1e-6; beta = 1e-3; nu = 3;
allm = m(:);
allentropy = []; allloglike = []; alllogprior = []; all_elbo = [];
active_log = struct('idx',{},'score',{});
best_elbo = -inf; m_best = m; V_best = V; D_best = D;

if plots
    fw = figure('position',[570,659,1740,649]);
end

for iter = 1:maxIter
    % Predictions and residuals
    y_pred = f(m);
    r = y - y_pred;
    sigma2 = max(epsilon, (r.^2 + beta) ./ (nu + r.^2/2));

    % Jacobian and curvatures
    J = computeJacobian(f, m, n);
    H_like  = J' * diag(1./sigma2) * J;
    H_prior = inv(S0 + computeSmoothCovariance(m,2));
    H_elbo  = H_like + H_prior;
    g_elbo  = J' * diag(1./sigma2) * r - H_prior * (m - m0);

    % --- Build an effect score s_i to decide which params to move
    Hdiag = max(real(diag(H_elbo)), 1e-12);  % ensure positive
    switch lower(effect_mode)
        case 'grad'
            % preconditioned gradient magnitude ~ |g_i| / sqrt(H_ii)
            score = abs(g_elbo) ./ sqrt(Hdiag);
        case 'zpost'
            % posterior shift relative to prior scale: |m - m0| / sqrt(Var_i)
            % approximate Var_i by 1 / H_ii (diagonal precision)
            vdiag = 1 ./ Hdiag;
            score = abs(m - m0) ./ sqrt(vdiag);
        otherwise % 'zstep'
            % predicted step size standardised by posterior var:
            % dm ≈ H^{-1} g, but we approximate component-wise: dm_i ~ g_i / H_ii
            dm_diag = g_elbo ./ Hdiag;
            vdiag   = 1 ./ Hdiag;
            score   = abs(dm_diag) ./ sqrt(vdiag);
    end

    % --- Select active set
    switch lower(selection)
        case 'topk'
            K = max(1, min(round(sel_param), numel(m)));
            [~,ord] = sort(score,'descend');
            active_idx = false(numel(m),1);
            active_idx(ord(1:K)) = true;
        case 'threshold'
            tau = sel_param;
            active_idx = (score >= tau);
            if ~any(active_idx)
                % keep at least one to avoid stall
                [~,jmax] = max(score);
                active_idx(jmax) = true;
            end
        otherwise
            error('selection must be ''topK'' or ''threshold''');
    end

    % Optionally keep the same active set for a few iterations
    if iter>1 && reeval_every>1 && mod(iter-1,reeval_every)~=0
        active_idx = active_log(end).idx;
    end
    active_log(end+1).idx = active_idx; %#ok<AGROW>
    active_log(end).score = score;

    % --- Solve the Newton step ONLY on the active block
    idx = find(active_idx);
    gA  = g_elbo(idx);
    HA  = H_elbo(idx,idx);

    try
        LA = chol(HA,'lower');
        dmA = LA' \ (LA \ gA);
    catch
        [dmA,flag,relres] = pcg(HA + 1e-6*eye(numel(idx)), gA, 1e-6, 200);
        if flag~=0 || relres>1e-2
            dmA = zeros(numel(idx),1);
        end
        % small damping
        dmA = dmA * min(1, 1/(1+norm(dmA)));
    end

    % Optional solenoidal mixing
    if solenoidalmix
        QA = HA - HA';
        dmA = dmA - 0.1 * QA * dmA;
    end

    % Build full dm with zeros for inactive params
    dm = zeros(size(m));
    dm(idx) = dmA;

    % Trust region on full step
    maxStepSize = 1.0;
    nrm = norm(dm);
    if nrm > maxStepSize, dm = dm * (maxStepSize/nrm); end

    m_prev = m;
    m = m + dm;

    % --- Optional proximal soft-thresholding toward prior mean (lasso-like)
    if lambda_soft > 0
        % soft-threshold only the INACTIVE params (keeps “few big effects” vibe)
        inact = ~active_idx;
        delta = m(inact) - m0(inact);
        m(inact) = m0(inact) + sign(delta) .* max(abs(delta) - lambda_soft, 0);
    end

    % Bookkeeping
    allm = [allm m(:)];

    % ELBO terms (using diagonal entropy approx for speed)
    logL_lik   = -0.5 * sum((r.^2 ./ sigma2) + log(2*pi*sigma2));
    logL_prior = -0.5 * ((m - m0)' * H_prior * (m - m0));
    logL_ent   = 0.5 * sum(log(Hdiag + 1e-6)); % proxy (precision diag)
    logL       = logL_lik + logL_prior + logL_ent;

    allentropy = [allentropy logL_ent];
    allloglike = [allloglike logL_lik];
    alllogprior= [alllogprior logL_prior];
    all_elbo   = [all_elbo logL];

    if logL > best_elbo
        best_elbo = logL; m_best = m; V_best = V; D_best = D;
    end

    % ---- Plot (copied layout, minimal edits)
    if plots
        w = 1:length(y);
        y_pred_new = f(m);
        figure(fw); clf;
        t = tiledlayout(2,4,'TileSpacing','compact','Padding','compact');
        nexttile([1 4]);
        hold on;
        errorbar(w, y, sqrt(sigma2), 'k.', 'CapSize',0);
        plot(w, y, 'k', 'LineWidth',1);
        plot(w, y_pred, '--', 'Color', [0 0.4 1], 'LineWidth',1.5);
        plot(w, y_pred_new, '-', 'Color', [0.8 0 0], 'LineWidth',2);
        plot(w, sqrt(sigma2), '-', 'Color', [0.1 0.6 0.1], 'LineWidth',1.5);
        % visualise active set along x-axis (tiny rug plot on top)
        aidx = find(active_idx);
        yl = ylim; 
        scatter(aidx, yl(2)*ones(size(aidx)), 12, 'k', 'filled', 'MarkerFaceAlpha',0.3, 'MarkerEdgeAlpha',0.3);
        hold off; title('Fit (+ active params as rug)'); grid on; box on;

        lineColor = [1 0.7 0.7]; scatterColor='k'; scatterSize=30; lineWidth=2;
        nexttile; plot(1:iter, allentropy, 'Color',lineColor, 'LineWidth',lineWidth); hold on;
        scatter(1:iter, allentropy, scatterSize, scatterColor, 'filled'); title('Entropy'); grid on; box on;

        nexttile; plot(1:iter, allloglike, 'Color',lineColor, 'LineWidth',lineWidth); hold on;
        scatter(1:iter, allloglike, scatterSize, scatterColor, 'filled'); title('LogLik'); grid on; box on;

        nexttile; plot(1:iter, alllogprior, 'Color',lineColor, 'LineWidth',lineWidth); hold on;
        scatter(1:iter, alllogprior, scatterSize, scatterColor, 'filled'); title('LogPrior'); grid on; box on;

        nexttile; plot(1:iter, all_elbo, 'Color',lineColor, 'LineWidth',lineWidth); hold on;
        scatter(1:iter, all_elbo, scatterSize, scatterColor, 'filled'); title('ELBO'); grid on; box on;

        set(gcf,'Color','w'); set(findall(gcf,'-property','FontSize'),'FontSize',18); drawnow;
    end

    % Convergence
    y_pred_new = f(m);
    if norm(dm) < tol || norm((y - y_pred_new).^2) <= thresh
        fprintf('Converged at iteration %d\n', iter);
        break;
    end

    % Backtracking if ELBO worsens (simple)
    if iter>1 && all_elbo(end) < all_elbo(end-1)
        fprintf('ELBO decreased, backtracking...\n');
        stepScale = 0.5;
        improved = false;
        for a=1:8
            m_try = m_prev + stepScale^a * dm;
            r_try = y - f(m_try);
            s2_try = max(epsilon, (r_try.^2 + beta) ./ (nu + r_try.^2/2));
            J_try  = computeJacobian(f, m_try, n);
            H_try  = J_try' * diag(1./s2_try) * J_try + H_prior;
            Hdiag_try = max(real(diag(H_try)),1e-12);
            logL_try = -0.5*sum((r_try.^2./s2_try)+log(2*pi*s2_try)) ...
                       -0.5*((m_try-m0)'*H_prior*(m_try-m0)) ...
                       +0.5*sum(log(Hdiag_try+1e-6));
            if logL_try > logL
                m = m_try; logL = logL_try; improved=true; break;
            end
        end
        if ~improved
            m = m_prev; logL = all_elbo(end-1);
        else
            all_elbo(end) = logL;
        end
    end

    fprintf('Iter: %d | ELBO: %.4f | ||dm||: %.4f | active=%d/%d\n', ...
        iter, logL, norm(dm), nnz(active_idx), numel(active_idx));
end

% Return best found
if best_elbo > logL
    m = m_best; V = V_best; D = D_best; logL = best_elbo;
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
