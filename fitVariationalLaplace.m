function [m, S, logL, iter,sigma2] = fitVariationalLaplace(y, f, m0, S0, maxIter, tol)
% Variational Laplace with iterative optimization of heteroscedastic variance.
%
% This function uses Variational Laplace to approximate the posterior over parameters
% given observed data and a generative model.
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
%   S        - Optimized covariance of variational distribution
%   sigma2   - Optimized observation variances (vector)
%   logL     - Final ELBO value
%   iter     - Number of iterations performed
%
% AS2025

% Initialization
m = m0(:); % Mean of variational distribution
S = S0;    % Covariance of variational distribution
n = length(y);

sigma2  = ones(n, 1);   % Initialize variances (homoscedastic starting point)
epsilon = 1e-6;         % Small value to ensure positivity
beta    = 1e-3;         % Smoothing parameter for variance updates
nu      = 1.0;          % Scaling factor for variance updates

for iter = 1:maxIter
    % Predictions and residuals
    y_pred = f(m);
    residuals = y - y_pred;
    
    % Update heteroscedastic variance
    sigma2 = max(epsilon, (residuals.^2 + beta) / nu);

    % Log-likelihood and Jacobian
    logL_likelihood = -0.5 * sum((residuals.^2 ./ sigma2) + log(2 * pi * sigma2));
    J = computeJacobian(f, m, n);

    % ELBO components [using just diag(1.sigma)]
     H = J' * diag(1 ./ sigma2) * J; % Likelihood Hessian
     g = J' * diag(1 ./ sigma2) * residuals; % Gradient

    % Smooth weights
    %W = radialPD(sigma2,2)*diag(1./sigma2)*radialPD(sigma2,2)';

    % ELBO components [using full radially-smoothed W]
    %H = J' * (W) * J; % Likelihood Hessian
    %g = J' * (W) * residuals; % Gradient

    H_prior = inv(S0 + eye(size(S0)) * 1e-6);
    g_prior = H_prior * (m - m0);
    
    H_elbo = H + H_prior;
    g_elbo = g - g_prior;

    % Update mean and covariance ( could just inv(H)*g )
    dm = pcg(H_elbo, g_elbo, 1e-6, 100);  % Conjugate Gradient method

    m  = m + dm; 
    S  = pinv(H_elbo + eye(size(H_elbo)) * 1e-6);

    % Compute ELBO
    cholS = chol(S + eye(size(S)) * 1e-6, 'lower');
    logL_entropy = sum(log(diag(cholS))) - 0.5 * length(m);
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
    title('Model Fit with Heteroscedastic Variance');
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

    % Display diagnostics
    fprintf('Iter: %d | ELBO: %.4f | ||dm||: %.4f\n', iter, logL, norm(dm));
end

end

function J = computeJacobian(f, x, m)
% computeJacobian - Numerical computation of Jacobian matrix
%   f: Function handle that computes model values f(x)
%   x: Current parameter estimates
%   m: Length of observed data vector y
%
% Outputs:
%   J: Jacobian matrix (m x length(x))

epsilon = 1e-6;  % Small step size for finite differences
n = length(x);
J = zeros(m, n);

parfor i = 1:n
    x_step = x;
    x_stepb = x;
    x_step(i) = x_step(i) + epsilon;
    x_stepb(i) = x_stepb(i) - epsilon;
    

    J(:, i) = (f(x_step) - f(x_stepb)) / epsilon;
end
end

function elbo = lineSearchObjective(alpha, m, delta_m, y, f, H_prior, m0, sigma2)
    % Update the parameter estimate
    m_new = m + alpha * delta_m;
    % Predictions and residuals
    y_pred_new = f(m_new);
    residuals_new = y - y_pred_new;
    % Log-likelihood
    logL_likelihood = -0.5 * sum((residuals_new.^2 ./ sigma2) + log(2 * pi * sigma2));
    % Prior contribution
    logL_prior = -0.5 * ((m_new - m0)' * H_prior * (m_new - m0));
    % Total ELBO
    elbo = logL_likelihood + logL_prior;
end