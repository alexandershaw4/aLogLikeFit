function [m, V, D, logL, iter, sigma2] = fitVariationalLaplaceThermo(y, f, m0, S0, maxIter, tol)
% Extended Variational Laplace with Low-Rank Approximation, Smarter Variance 
% Updates, and Thermodynamic Integration. Non extended version is fitVariationalLaplace.
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
k = max(k, 5); % Ensure a minimum rank for stability

% initialization of V
V = U(:, 1:k) * diag(sqrt(eigvals(1:k)));
D = diag(diag(S0) - sum(V.^2, 2));

%k = min(10, length(m)); % Rank for low-rank covariance approximation
%V = randn(length(m), k); % Low-rank component
%D = diag(S0); % Diagonal component
sigma2 = ones(n, 1);
epsilon = 1e-6;
beta = 1e-3;
nu = 1.0;

for iter = 1:maxIter
    % Predictions and residuals
    y_pred = f(m);
    residuals = y - y_pred;
    
    % Robust variance update using Huber loss
    delta = 1.5;
    huber_residuals = residuals;
    huber_residuals(abs(residuals) > delta) = delta * sign(residuals(abs(residuals) > delta));
    sigma2 = max(epsilon, (huber_residuals.^2 + beta) / nu);
    
    % Log-likelihood and Jacobian
    logL_likelihood = -0.5 * sum((residuals.^2 ./ sigma2) + log(2 * pi * sigma2));
    J = computeJacobian(f, m, n);
    
    % Low-rank approximation: S ≈ VV^T + D
    H = J' * diag(1 ./ sigma2) * J;
    H_prior = inv(S0 + eye(size(S0)) * 1e-6);
    H_elbo = H + H_prior;
    g_elbo = J' * diag(1 ./ sigma2) * residuals - H_prior * (m - m0);
    
    % Low-rank covariance update
    [U, Sval, ~] = svd(H_elbo, 'econ');
    V = U(:, 1:k) * sqrt(Sval(1:k, 1:k));
    D = diag(diag(H_elbo) - sum(V.^2, 2));
    
    % Update mean using preconditioned CG
    dm = pcg(H_elbo, g_elbo, 1e-6, 100);
    m  = m + dm;
    
    % Thermodynamic integration for model evidence
    logL_entropy = 0.5 * sum(log(diag(D) + 1e-6));
    logL_prior = -0.5 * ((m - m0)' * H_prior * (m - m0));
    logL = logL_likelihood + logL_prior + logL_entropy;
   
    % Show
    w = 1:length(y);   % x vals
    y_pred_new = f(m); % Updated predictions
    
    % Plot observed data, predictions, and variance
    figure(1); clf;
    errorbar(w, y, sqrt(sigma2), 'k.', 'DisplayName', 'Observed (±σ)'); % Observed data with variance
    hold on;
    plot(w, y, 'k', 'LineWidth', 1, 'DisplayName', 'Observed (Mu)');
    plot(w, y_pred, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Previous Prediction');
    plot(w, y_pred_new, 'r-', 'LineWidth', 2, 'DisplayName', 'Current Prediction');
    plot(w, sqrt(sigma2), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Heteroscedastic σ'); % Variance curve
    hold off;
    title('Model Fit: Variational Laplace with Heteroscedastic Variance');
    xlabel('Data Index');
    ylabel('Value');
    legend('Location', 'best');
    grid on;
    drawnow;


    % Convergence check
    if norm(dm) < tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
    
    fprintf('Iter: %d | ELBO: %.4f | ||dm||: %.4f\n', iter, logL, norm(dm));
end
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