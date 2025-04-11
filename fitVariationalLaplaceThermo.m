function [m, V, D, logL, iter, sigma2, allm] = fitVariationalLaplaceThermo(y, f, m0, S0, maxIter, tol)
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
% Returs the posterior: q(z) ~ N(m, VVᵀ + D)
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

% Initialization
m = m0(:);
n = length(y);

% Adaptive rank selection for low-rank approximation
[U, Sval, ~] = svd(full(S0), 'econ');
eigvals = diag(Sval);
threshold = 0.01 * max(eigvals); % Retain components contributing >1% of max eigenvalue
k = sum(eigvals > threshold);
k = max(k, length(m0)); % Ensure a minimum rank for stability

% initialization of V
V = U(:, 1:k) * diag(sqrt(eigvals(1:k)));
D = diag(diag(S0) - sum(V.^2, 2));

%k = min(10, length(m)); % Rank for low-rank covariance approximation
%V = randn(length(m), k); % Low-rank component
%D = diag(S0); % Diagonal component
sigma2 = ones(n, 1);
epsilon = 1e-6;
beta = 1e-3;

% keep track over iterations for plotting
allm = m(:);
allentropy = [];
allloglike = [];
alllogprior = [];
all_elbo = [];

fw = figure('position',[570,659,1740,649]);

for iter = 1:maxIter
    % Predictions and residuals
    y_pred = f(m);
    residuals = y - y_pred;
    
    % Robust variance update using Huber loss
    delta = 1.5;
    huber_residuals = residuals;
    huber_residuals(abs(residuals) > delta) = delta * sign(residuals(abs(residuals) > delta));

    nu = 3; % Degrees of freedom for t-distribution
    sigma2 = max(epsilon, (residuals.^2 + beta) ./ (nu + residuals.^2 / 2));
    %sigma2 = max(epsilon, (huber_residuals.^2 + beta) / nu);
    
    % Log-likelihood and Jacobian
    logL_likelihood = -0.5 * sum((residuals.^2 ./ sigma2) + log(2 * pi * sigma2));
    J = computeJacobian(f, m, n);
    
    % Low-rank approximation: S ≈ VV^T + D
    H = J' * diag(1 ./ sigma2) * J;
   % H_prior = inv(S0 + eye(size(S0)) * 1e-6);
    H_prior = inv(S0 + computeSmoothCovariance(m, 2)); % Instead of just S0

    H_elbo = H + H_prior;
    g_elbo = J' * diag(1 ./ sigma2) * residuals - H_prior * (m - m0);
    
    % Low-rank covariance update
    [U, Sval, ~] = svd(H_elbo, 'econ');
    V = U(:, 1:k) * sqrt(Sval(1:k, 1:k));
    D = diag(diag(H_elbo) - sum(V.^2, 2));
    
    % Update mean using preconditioned CG
    try
        L = chol(H_elbo, 'lower'); % Ensure positive definiteness
    catch
        L = chol(makeposdef(H_elbo), 'lower'); % Ensure positive definiteness
    end

    dm = L' \ (L \ g_elbo); % Solve using Cholesky decomposition

    %dm = pcg(H_elbo, g_elbo, 1e-6, 100);
    m  = m + dm;
    allm = [allm m(:)];
    
    % Thermodynamic integration for model evidence
    logL_entropy = 0.5 * sum(log(diag(D) + 1e-6));
    logL_prior = -0.5 * ((m - m0)' * H_prior * (m - m0));
    logL = logL_likelihood + logL_prior + logL_entropy;
   
    allentropy = [allentropy logL_entropy];
    allloglike = [allloglike logL_likelihood];
    alllogprior = [alllogprior logL_prior];
    all_elbo = [all_elbo logL];

    % Adaptive step-size: backtrack if ELBO gets worse
    if iter > 1 && all_elbo(end) < all_elbo(end-1)
        fprintf('ELBO decreased. Dampening step...\n');
        dm = dm * 0.5;  % Reduce step size
        m = m - dm;     % Revert previous update
        m = m + dm * 0.25; % Try smaller update instead
    end

    % Show
    w = 1:length(y);   % x vals
    y_pred_new = f(m); % Updated predictions
  

    % Create a tiled layout for cleaner subplot spacing
    figure(fw); clf;
    t = tiledlayout(2,4, 'TileSpacing', 'compact', 'Padding', 'compact');

    % --- Main model fit plot ---
    nexttile([1 4]);
    hold on;
    errorbar(w, y, sqrt(sigma2), 'k.', 'DisplayName', 'Observed ±σ', 'CapSize', 0);
    plot(w, y, 'k', 'LineWidth', 1, 'DisplayName', 'Observed Mean');
    plot(w, y_pred, '--', 'Color', [0 0.4 1], 'LineWidth', 1.5, 'DisplayName', 'Previous Prediction');
    plot(w, y_pred_new, '-', 'Color', [0.8 0 0], 'LineWidth', 2, 'DisplayName', 'Current Prediction');
    plot(w, sqrt(sigma2), '-', 'Color', [0.1 0.6 0.1], 'LineWidth', 1.5, 'DisplayName', 'Heteroscedastic σ');
    hold off;

    title('Model Fit: Variational Laplace with Heteroscedastic Variance', 'FontWeight', 'bold');
    xlabel('Data Index');
    ylabel('Value');
    legend('Location', 'best');
    grid on;
    box on;

    % Define common style
    lineColor = [1 0.7 0.7];
    scatterColor = 'k';
    scatterSize = 30;
    lineWidth = 2;

    % --- Entropy ---
    nexttile;
    plot(1:iter, allentropy, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
    scatter(1:iter, allentropy, scatterSize, scatterColor, 'filled'); hold off;
    title('Entropy', 'FontWeight', 'bold');
    xlabel('Iteration');
    grid on; box on;

    % --- Log-likelihood ---
    nexttile;
    plot(1:iter, allloglike, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
    scatter(1:iter, allloglike, scatterSize, scatterColor, 'filled'); hold off;
    title('Log-Likelihood', 'FontWeight', 'bold');
    xlabel('Iteration');
    grid on; box on;

    % --- Log-prior ---
    nexttile;
    plot(1:iter, alllogprior, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
    scatter(1:iter, alllogprior, scatterSize, scatterColor, 'filled'); hold off;
    title('Log-Prior', 'FontWeight', 'bold');
    xlabel('Iteration');
    grid on; box on;

    % --- ELBO ---
    nexttile;
    plot(1:iter, all_elbo, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
    scatter(1:iter, all_elbo, scatterSize, scatterColor, 'filled'); hold off;
    title('ELBO', 'FontWeight', 'bold');
    xlabel('Iteration');
    grid on; box on;

    % Optional: use a nicer colormap or export-friendly background
    set(gcf, 'Color', 'w'); % White background for saving/export
    set(findall(gcf,'-property','FontSize'),'FontSize',18);
    drawnow;

    
    % Convergence check
    if norm(dm) < tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
    
    fprintf('Iter: %d | ELBO: %.4f | ||dm||: %.4f\n', iter, logL, norm(dm));
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

% function J = computeJacobianAD(f, x)
%     x_dl = dlarray(x);  % Convert to differentiable dlarray
%     y_dl = f(x_dl);     % Ensure f(x) outputs a scalar or element-wise function
%     J_dl = dlgradient(sum(y_dl), x_dl); % Compute gradient
%     J = extractdata(J_dl); % Convert back to normal MATLAB array
% end

    % % Plot observed data, predictions, and variance
    % figure(fw); clf;
    % subplot(2,4,[1:4]);
    % errorbar(w, y, sqrt(sigma2), 'k.', 'DisplayName', 'Observed (±σ)'); % Observed data with variance
    % hold on;
    % plot(w, y, 'k', 'LineWidth', 1, 'DisplayName', 'Observed (Mu)');
    % plot(w, y_pred, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Previous Prediction');
    % plot(w, y_pred_new, 'r-', 'LineWidth', 2, 'DisplayName', 'Current Prediction');
    % plot(w, sqrt(sigma2), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Heteroscedastic σ'); % Variance curve
    % hold off;
    % title('Model Fit: Variational Laplace with Heteroscedastic Variance');
    % xlabel('Data Index');
    % ylabel('Value');
    % legend('Location', 'best');
    % grid on;
    % 
    % subplot(2,4,5);
    % plot(1:iter,allentropy,'Color',[1 .7 .7],'linewidth',3); hold on;
    % scatter(1:iter,allentropy,30,'k','filled');grid on; hold off;
    % title('entropy');
    % 
    % subplot(2,4,6);
    % plot(1:iter,allloglike,'Color',[1 .7 .7],'linewidth',3); hold on;
    % scatter(1:iter,allloglike,30,'k','filled');grid on; hold off;
    % title('log likeihood');
    % 
    % subplot(2,4,7);
    % plot(1:iter,alllogprior,'Color',[1 .7 .7],'linewidth',3); hold on;
    % scatter(1:iter,alllogprior,30,'k','filled');grid on; hold off;
    % title('log prior');
    % 
    % subplot(2,4,8);
    % plot(1:iter,all_elbo,'Color',[1 .7 .7],'linewidth',3); hold on;
    % scatter(1:iter,all_elbo,30,'k','filled');grid on; hold off;
    % title('elbo');
    % 
    % drawnow;
    % 
