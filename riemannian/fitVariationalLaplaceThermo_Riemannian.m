function [m, V, D, logL, iter, sigma2, allm, g_elbo] = ...
    fitVariationalLaplaceThermo_Riemannian( ...
        y, f, m0, S0, maxIter, tol, plots, varpercthresh, varargin)
% fitVariationalLaplaceThermo_Riemannian
%
% Riemannian / natural-gradient extension of fitVariationalLaplaceThermo.
%
% Same core inputs/outputs as fitVariationalLaplaceThermo, with optional
% extra args:
%   metricFun (optional) : @(m, H_elbo, H_prior, J, sigma2, info) -> G
%                          where G is a positive-definite metric
%   eta       (optional) : step size for natural-gradient update
%
% The only conceptual change vs the Euclidean version is the update step
% for m:
%
%   dm_nat = G(m) \ g_elbo;
%   m_new  = m + eta * dm_nat;
%
% Everything else (heteroscedastic sigma^2, low-rank factorisation of
% H_elbo, plotting, backtracking) is kept as in the original routine.
%
% AS2025 (+ Riemannian extension)

% ---------------------------
% Optional Riemannian args
% ---------------------------
metricFun = [];
eta       = [];

if ~isempty(varargin)
    if numel(varargin) >= 1 && ~isempty(varargin{1})
        metricFun = varargin{1};
    end
    if numel(varargin) >= 2 && ~isempty(varargin{2})
        eta = varargin{2};
    end
end

if isempty(metricFun)
    % Default: diagonal-of-H metric + ridge
    metricFun = @defaultMetricDiagonalH;
end
if isempty(eta)
    eta = 1.0;   % default natural-gradient step size
end

% ---------------------------
% Defaults & initialisation
% ---------------------------
if nargin < 7 || isempty(plots)
    plots = 1;
end
if nargin < 8 || isempty(varpercthresh)
    varpercthresh = 0.01;
end

thresh        = 1/16;
solenoidalmix = 0;

% Initial mean
m = m0(:);
n = length(y);

% Adaptive rank selection for low-rank approximation
[U0, Sval0, ~] = svd(full(S0), 'econ');
eigvals0 = diag(Sval0);
threshold = varpercthresh * max(eigvals0); % Retain components > % of max eig
k = sum(eigvals0 > threshold);
k = max(k, length(m0)); % Ensure a minimum rank for stability

% Initial V, D (for covariance-like structure)
V  = U0(:, 1:k) * diag(sqrt(eigvals0(1:k)));
D  = diag(diag(S0) - sum(V.^2, 2));
V0 = (V * V') + D; %#ok<NASGU>

sigma2  = ones(n, 1);
epsilon = 1e-6;
beta    = 1e-3;   % used in heteroscedastic update

% Tracking
allm        = m(:);
allentropy  = [];
allloglike  = [];
alllogprior = [];
all_elbo    = [];
best_elbo   = -Inf;
m_best      = m;
V_best      = V;
D_best      = D;

if plots
    fw = figure('position',[570,659,1740,649]);
end

for iter = 1:maxIter

    % -------------------------
    % 1) Forward model & noise
    % -------------------------
    y_pred    = f(m);
    residuals = y - y_pred;

    % Heteroscedastic / t-like variance update
    nu     = 3; % df for t
    sigma2 = max(epsilon, (residuals.^2 + beta) ./ (nu + residuals.^2 / 2));

    % Log-likelihood
    logL_likelihood = -0.5 * sum((residuals.^2 ./ sigma2) + log(2 * pi * sigma2));

    % Jacobian
    J = computeJacobian(f, m, n);

    % -------------------------
    % 2) Hessian / precision
    % -------------------------
    % Likelihood Hessian
    H = J' * diag(1 ./ sigma2) * J;

    % Prior precision
    H_prior = inv(S0 + computeSmoothCovariance(m, 2)); % GP-ish smooth prior

    % ELBO precision
    H_elbo = H + H_prior;

    % Gradient of ELBO wrt m (up to sign convention)
    % g_elbo is ∂F/∂m (we’ll treat it as "gradient direction" for ascent)
    g_elbo = J' * diag(1 ./ sigma2) * residuals - H_prior * (m - m0);

    % -------------------------
    % 3) Low-rank factorisation of H_elbo
    % -------------------------
    [U, Sval, ~] = svd(H_elbo, 'econ');
    V = U(:, 1:k) * sqrt(Sval(1:k, 1:k));
    D = diag(diag(H_elbo) - sum(V.^2, 2));

    % -------------------------
    % 4) Natural-gradient update using G(m)
    % -------------------------
    % Build info struct to feed the metric if desired
    info.iter      = iter;
    info.H_elbo    = H_elbo;
    info.H_prior   = H_prior;
    info.J         = J;
    info.sigma2    = sigma2;
    info.m_current = m;

    % Riemannian metric G(m)
    G = metricFun(m, H_elbo, H_prior, J, sigma2, info);

    % Solve G * dm_nat = g_elbo  (natural gradient)
    % i.e., dm_nat = G^{-1} g_elbo
    try
        Lg = chol(G, 'lower');
        dm_nat = Lg' \ (Lg \ g_elbo);
    catch
        fprintf('Metric Cholesky failed, falling back to PCG on G.\n');
        [dm_nat, flagG, relresG] = pcg(G + eye(size(G))*1e-6, g_elbo, 1e-6, 100);
        if flagG ~= 0
            fprintf('PCG on metric failed (flag %d, relres=%.2e). Using zero step.\n', ...
                flagG, relresG);
            dm_nat = zeros(size(m));
        end
    end

    % Apply step size (for F-ascent; if you regard g_elbo as ∇F)
    dm = eta * dm_nat;

    % Trust-region style damping
    maxStepSize = 1.0;
    if norm(dm) > maxStepSize
        dm = dm * (maxStepSize / norm(dm));
    end

    % Optional solenoidal mixing wrt H_elbo (unchanged)
    if solenoidalmix
        Q = H_elbo - H_elbo';       % skew-symmetric part of Hessian
        gamma = 0.1;                % scaling factor for solenoidal adjustment
        dm = dm - gamma * Q * dm;
    end

    m_prev = m;
    m      = m + dm;
    allm   = [allm m(:)]; %#ok<AGROW>

    % -------------------------
    % 5) ELBO / free energy
    % -------------------------
    % Recompute y_pred_new for diagnostics only
    y_pred_new = f(m);

    % Entropy-ish term: here you were using diag(D) as proxy
    logL_entropy = 0.5 * sum(log(diag(D) + 1e-6));

    % Prior term
    logL_prior = -0.5 * ((m - m0)' * H_prior * (m - m0));

    % ELBO (approx)
    logL = logL_likelihood + logL_prior + logL_entropy;

    % -------------------------
    % 6) Backtracking on ELBO
    % -------------------------
    if iter > 1 && logL < all_elbo(end)
        fprintf('ELBO decreased. Initiating backtracking...\n');
        success   = false;
        stepScale = 0.5;
        maxTries  = 10;
        attempt   = 0;
        m_trial   = m_prev;

        while ~success && attempt < maxTries
            attempt  = attempt + 1;
            alpha_bt = stepScale^attempt;
            m_trial  = m_prev + alpha_bt * dm;

            y_pred_trial    = f(m_trial);
            residuals_trial = y - y_pred_trial;
            sigma2_trial    = max(epsilon, (residuals_trial.^2 + beta) ./ ...
                                        (nu + residuals_trial.^2 / 2));
            J_trial = computeJacobian(f, m_trial, n);

            H_trial      = J_trial' * diag(1 ./ sigma2_trial) * J_trial + H_prior;
            logL_like_bt = -0.5 * sum((residuals_trial.^2 ./ sigma2_trial) + ...
                                      log(2 * pi * sigma2_trial));
            logL_prior_bt   = -0.5 * ((m_trial - m0)' * H_prior * (m_trial - m0));
            logL_entropy_bt = 0.5 * sum(log(diag(H_trial) + 1e-6)); % rough
            logL_trial      = logL_like_bt + logL_prior_bt + logL_entropy_bt;

            fprintf(' Attempt %d: scaled step %.4f | ELBO: %.4f\n', ...
                    attempt, alpha_bt, logL_trial);

            if logL_trial > logL
                success = true;
                m      = m_trial;
                logL   = logL_trial;
                % Optionally update V,D from H_trial if you want to be strict
            end
        end

        if ~success
            fprintf('Backtracking failed to improve ELBO. Reverting to previous state.\n');
            m    = m_prev;
            logL = all_elbo(end);
        end
    end

    % -------------------------
    % 7) Tracking & diagnostics
    % -------------------------
    allentropy  = [allentropy  logL_entropy];  %#ok<AGROW>
    allloglike  = [allloglike  logL_likelihood];
    alllogprior = [alllogprior logL_prior];
    all_elbo    = [all_elbo    logL];

    if iter == 1 || logL > best_elbo
        best_elbo = logL;
        m_best    = m;
        V_best    = V;
        D_best    = D;
    end

    % Plots
    if plots
        w = 1:length(y);   % x vals

        figure(fw); clf;
        t = tiledlayout(2,4, 'TileSpacing', 'compact', 'Padding', 'compact'); %#ok<NASGU>

        % --- Main model fit ---
        nexttile([1 4]);
        hold on;
        errorbar(w, y, sqrt(sigma2), 'k.', 'DisplayName', 'Observed ±σ', 'CapSize', 0);
        plot(w, y, 'k', 'LineWidth', 1, 'DisplayName', 'Observed Mean');
        plot(w, y_pred, '--', 'Color', [0 0.4 1], 'LineWidth', 1.5, ...
            'DisplayName', 'Previous Prediction');
        plot(w, y_pred_new, '-', 'Color', [0.8 0 0], 'LineWidth', 2, ...
            'DisplayName', 'Current Prediction');
        plot(w, sqrt(sigma2), '-', 'Color', [0.1 0.6 0.1], 'LineWidth', 1.5, ...
            'DisplayName', 'Heteroscedastic σ');
        hold off;

        title('Model Fit: Riemannian VL with Heteroscedastic Variance', 'FontWeight', 'bold');
        xlabel('Data Index'); ylabel('Value');
        legend('Location', 'best'); grid on; box on;

        % Common style
        lineColor   = [1 0.7 0.7];
        scatterColor= 'k';
        scatterSize = 30;
        lineWidth   = 2;

        % --- Entropy ---
        nexttile;
        plot(1:iter, allentropy, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
        scatter(1:iter, allentropy, scatterSize, scatterColor, 'filled'); hold off;
        title('Entropy', 'FontWeight', 'bold');
        xlabel('Iteration'); grid on; box on;

        % --- Log-likelihood ---
        nexttile;
        plot(1:iter, allloglike, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
        scatter(1:iter, allloglike, scatterSize, scatterColor, 'filled'); hold off;
        title('Log-Likelihood', 'FontWeight', 'bold');
        xlabel('Iteration'); grid on; box on;

        % --- Log-prior ---
        nexttile;
        plot(1:iter, alllogprior, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
        scatter(1:iter, alllogprior, scatterSize, scatterColor, 'filled'); hold off;
        title('Log-Prior', 'FontWeight', 'bold');
        xlabel('Iteration'); grid on; box on;

        % --- ELBO ---
        nexttile;
        plot(1:iter, all_elbo, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
        scatter(1:iter, all_elbo, scatterSize, scatterColor, 'filled'); hold off;
        title('ELBO', 'FontWeight', 'bold');
        xlabel('Iteration'); grid on; box on;

        set(gcf, 'Color', 'w');
        set(findall(gcf,'-property','FontSize'),'FontSize',18);
        drawnow;
    end

    % -------------------------
    % 8) Convergence check
    % -------------------------
    if norm(dm) < tol || norm((y - y_pred_new).^2) <= thresh
        fprintf('Converged at iteration %d\n', iter);
        break;
    end

    fprintf('Iter: %d | ELBO: %.4f | ||dm||: %.4f\n', iter, logL, norm(dm));

end % iter loop

% Return best fits
fprintf('Returning best fits (Riemannian)...\n');
m    = m_best;
V    = V_best;
D    = D_best;
logL = best_elbo;

end % main function

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

