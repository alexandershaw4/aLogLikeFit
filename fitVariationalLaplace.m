function [m, S, logL, iter] = fitVariationalLaplace(y, f, m0, S0, sigma, maxIter, tol)
% Variational Laplace approximation for parameter estimation.
%
% This function uses Variational Laplace to approximate the posterior over parameters
% given observed data and a generative model.
%
% Inputs:
%   y        - Observed data (vector)
%   f        - Model function handle (f(x) returns predicted values)
%   m0       - Initial mean for variational distribution (vector)
%   S0       - Initial covariance for variational distribution (matrix)
%   sigma    - Standard deviations of observations (vector or scalar)
%   maxIter  - Maximum number of iterations
%   tol      - Convergence tolerance
%
% Outputs:
%   m        - Optimized mean of variational distribution
%   S        - Optimized covariance of variational distribution
%   logL     - Final ELBO value
%   iter     - Number of iterations performed
%
% AS:12/2024


% Initialization
m = m0(:); % Mean of variational distribution
S = S0;    % Covariance of variational distribution
n = length(y);

% Precompute observation variance
sigma2 = sigma.^2;

for iter = 1:maxIter
    % Predictions and residuals
    y_pred = f(m);
    residuals = y - y_pred;

    % Log-likelihood and Jacobian
    logL_likelihood = -0.5 * sum((residuals ./ sigma).^2 + log(2 * pi * sigma2));
    J = computeJacobian(f, m, n);

    % ELBO components
    H = J' * diag(1 ./ sigma2) * J; % Likelihood Hessian
    g = J' * diag(1 ./ sigma2) * residuals; % Gradient

    % GN components
    H_prior = inv(S0 + eye(size(S0)) * 1e-6);
    g_prior = H_prior * (m - m0);
    
    H_elbo = H + H_prior;
    g_elbo = g - g_prior;

    % Update mean and covariance
    dm = pinv(H_elbo + eye(size(H_elbo)) * 1e-6) * g_elbo;
    m  = m +  dm; 
    S  = pinv(H_elbo + eye(size(H_elbo)) * 1e-6);

    % Compute ELBO
    cholS = chol(S + eye(size(S)) * 1e-6, 'lower');
    logL_entropy = sum(log(diag(cholS))) - 0.5 * length(m);
    logL_prior = -0.5 * ((m - m0)' * H_prior * (m - m0));
    logL = logL_likelihood + logL_prior + logL_entropy;

    % Show
    w = 1:length(y);
    y_pred_new = f(m);
    plot(w,y,':k',w,y_pred,w,y_pred_new,'linewidth',2);
    drawnow;

    % Convergence check
    if norm(dm) < tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end

    % Display diagnostics
    fprintf('Iter: %d | ELBO: %.4f | ||dm||: %.4f\n', iter, logL, norm(dm));
end


% for iter = 1:maxIter
%     % Compute predictive mean and residuals
%     y_pred = f(m);
%     residuals = y - y_pred;
% 
%     % Compute log-likelihood term
%     logL_likelihood = -0.5 * sum((residuals ./ sigma).^2 + log(2 * pi * sigma2));
% 
%     % Compute the Jacobian of the model
%     J = computeJacobian(f, m, n);
% 
%     % Compute the Hessian and gradient for the ELBO
%     H = J' * diag(1 ./ sigma2) * J; % Approximation to the Hessian of the log-likelihood
%     g = J' * diag(1 ./ sigma2) * residuals; % Gradient
% 
%     % Add prior terms
%     H_prior = inv(S0); % Prior precision matrix
%     g_prior = H_prior * (m - m0); % Prior gradient
% 
%     % ELBO components
%     H_elbo = H + H_prior; % Effective Hessian
%     g_elbo = g - g_prior; % Effective gradient
% 
%     % Update variational parameters
%     dm = pinv(H_elbo) * g_elbo; % Update for the mean
%     m = m + dm;
% 
%     % Update covariance (Laplace approximation)
%     S = pinv(H_elbo);
% 
%     % Compute ELBO
%     logL_prior = -0.5 * ((m - m0)' * H_prior * (m - m0));
%     logL_entropy = 0.5 * log(det(S)) - 0.5 * length(m);
%     logL = logL_likelihood + logL_prior + logL_entropy;
% 
%     % Show
%     w = 1:length(y);
%     y_pred_new = f(m);
%     plot(w,y,':k',w,y_pred,w,y_pred_new,'linewidth',2);
%     drawnow;
% 
% 
%     % Convergence check
%     if norm(dm) < tol
%         break;
%     end
% 
%     % Display iteration info
%     fprintf('Iter: %d | ELBO: %.4f\n', iter, logL);
% end
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

for i = 1:n
    x_step = x;
    x_step(i) = x_step(i) + epsilon;
    J(:, i) = (f(x_step) - f(x)) / epsilon;
end
end