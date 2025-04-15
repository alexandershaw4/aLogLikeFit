function [beta, lambda_vals, t_stats, p_values, posterior_means, posterior_covs] = ...
    peb_ard_with_stats_LM(theta, X, Sigma_theta_prior, max_iter, tol)
% This function implements a Parametric Empirical Bayes (PEB) method for estimating 
% group-level parameters while incorporating individual-level priors. The 
% method combines ridge regression with Bayesian regularisation, using prior 
% covariance information about individual parameters to shrink the group-level 
% estimates. It also includes Automatic Relevance Determination (ARD) to determine 
% the importance of each predictor. This version of the method incorporates the 
% Levenberg-Marquardt (LM) algorithm to optimise the parameter estimation, 
% adjusting the update step to improve convergence and stability.
%
% The function returns the group-level parameter estimates, ARD hyperparameters, 
% t-statistics, p-values, and the individual-level posterior means and covariances.
%
%  [b,l,t,p,pos_mu,pos_cov] = peb_ard_with_stats_LM(theta, X, Sigma_theta_prior, max_iter, tol)
%
% Inputs:
% - theta: Individual-level parameters (N x d)
% - X: Design matrix (N x p)
% - Sigma_theta_prior: Prior covariance of individual-level parameters (d x d)
% - max_iter: Maximum number of iterations
% - tol: Convergence tolerance
%
% Outputs:
% - beta: Estimated beta coefficients (p x d)
% - lambda_vals: ARD hyperparameters (p x 1)
% - t_stats: t-statistics for each beta coefficient
% - p_values: p-values for each beta coefficient
% - posterior_means: Individual-level posterior means (N x d)
% - posterior_covs: Posterior covariances (d x d)
%

if nargin < 4
    max_iter = 100; % Default maximum iterations
end
if nargin < 5
    tol = 1e-6; % Default convergence tolerance
end

[N, d] = size(theta);
p = size(X, 2); % Number of predictors

% Initialize lambda and beta
lambda_vals = ones(p, 1);
beta = zeros(p, d);

% Inverse of prior covariance
Sigma_theta_prior = Sigma_theta_prior + 1e-6 * eye(d); % Add small regularization to avoid numerical issues
Sigma_theta_prior_inv = inv(Sigma_theta_prior);

% Initialize individual parameter covariances
sigma = diag(Sigma_theta_prior); % Start with prior variances for each parameter

% Levenberg-Marquardt parameters
lambda_LM = 1e-3;  % Damping factor (adjust as needed)
I = eye(p);

for iter = 1:max_iter
    beta_old = beta;

    % Compute the residuals for the full model
    residuals = theta - X * beta;  % Residuals based on current beta

    % Compute the loss (squared error)
    loss = sum(residuals(:).^2);

    % Compute the Jacobian (gradient of the loss function with respect to beta)
    J = -2 * X' * residuals;  % Jacobian of the least squares function
    J = J';

    % Levenberg-Marquardt update step (beta update)
    H = J * J' + lambda_LM * I;  % Hessian approximation with damping
    grad = J' * residuals';  % Gradient
    delta_beta = pinv(H)*J;% H \ grad;  % Solve for the update step

    % Update beta (Levenberg-Marquardt step)
    beta = beta + reshape(delta_beta, p, d);

    % Update ARD hyperparameters
    lambda_vals = 1 ./ (sum(beta.^2, 2) + 1e-6);

    % Update residuals and parameter-specific variances
    for j = 1:d
        residuals_j = theta(:, j) - X * beta(:, j);
        sigma(j) = (residuals_j' * residuals_j + Sigma_theta_prior(j, j)) / (N + 1);
    end

    % Check for convergence
    if norm(beta - beta_old, 'fro') < tol
        fprintf('Converged after %d iterations.\n', iter);
        break;
    end

    if iter == max_iter
        fprintf('Did not converge after %d iterations.\n', max_iter);
    end
end

% Compute posterior covariance and means
Sigma_residual = cov(theta - X * beta);
posterior_covs = Sigma_residual;
posterior_means = zeros(N, d);

for j = 1:d
    posterior_covs(j, j) = 1 / (1 / Sigma_theta_prior(j, j) + N / sigma(j));
    for i = 1:N
        posterior_means(i, j) = posterior_covs(j, j) * ...
            (theta(i, j) / sigma(j) + sum(X(i, :) .* beta(:, j)') / Sigma_theta_prior(j, j));
    end
end

% Compute variance of beta
beta_variance = zeros(p, d);
for j = 1:p
    beta_variance(j, :) = diag(Sigma_theta_prior) ./ (sum(X(:, j).^2) + lambda_vals(j)^-1);
end

% Compute t-statistics
t_stats = beta ./ sqrt(beta_variance);

% Compute p-values (two-tailed)
df = N - p; % Degrees of freedom
p_values = 2 * (1 - tcdf(abs(t_stats), df));

end
