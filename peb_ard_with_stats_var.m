function [beta, lambda_vals, t_stats, p_values, posterior_means, posterior_covs] = ...
    peb_ard_with_stats_var(theta, X, Sigma_theta_prior, max_iter, tol)
% This function implements a Parametric Empirical Bayes (PEB) method for 
% estimating group-level parameters while incorporating individual-level 
% priors on the parameters. The approach combines ridge regression with 
% Bayesian regularisation, where prior covariance information about individual 
% parameters is used to shrink the group-level estimates. It also includes 
% Automatic Relevance Determination (ARD) to determine the importance of each 
% predictor. Returns the full individual level posteriors.
%
% This version accepts individual level covariances (i.e. derived from the
% first level fits).
%
%  [b,l,t,p,pos_mu,pos_cov] = peb_ard_with_stats(theta, X, Sigma_theta_prior, max_iter, tol)
%
% Inputs:
% - theta: Individual-level parameters (N x d)
% - X: Design matrix (N x p)
% - Sigma_theta_prior: Prior covariance of individual-level parameters (N x d x d)
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
% AS2024

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

% Initialize residual variance
sigma_squared = ones(1, d); 

% Precompute inverses of Sigma_theta_prior for efficiency
Sigma_theta_prior_inv = zeros(N, d, d);
for i = 1:N
    Sigma_theta_prior_inv(i, :, :) = inv(squeeze(Sigma_theta_prior(i, :, :)) + 1e-6 * eye(d));
end

for iter = 1:max_iter
    beta_old = beta;

    % Update beta using parameter-specific priors
    for j = 1:p
        % Compute weighted sum for beta(j, :)
        beta(j, :) = (X(:, j)' * (theta - X * beta + X(:, j) * beta(j, :))) ./ ...
            (X(:, j)' * X(:, j) + lambda_vals(j));
    end

    % Update ARD hyperparameters
    lambda_vals = 1 ./ (sum(beta.^2, 2) + 1e-6);

    % Update residual variance sigma_squared
    residuals = theta - X * beta;
    sigma_squared = var(residuals, 0, 1);

    % Check for convergence
    if norm(beta - beta_old, 'fro') < tol
        fprintf('Converged after %d iterations.\n', iter);
        break;
    end

    if iter == max_iter
        fprintf('Did not converge after %d iterations.\n', max_iter);
    end
end

% Compute individual posterior means and covariances
posterior_means = zeros(N, d);
posterior_covs = zeros(N, d, d);

for i = 1:N
    % Subject-specific posterior covariance
    Sigma_post_inv = squeeze(Sigma_theta_prior_inv(i, :, :)) + ...
                     (1 / sigma_squared(i)) * eye(d);
    posterior_covs(i, :, :) = inv(Sigma_post_inv);

    % Subject-specific posterior mean
    posterior_means(i, :) = squeeze(posterior_covs(i, :, :)) *...
        (squeeze(Sigma_theta_prior_inv(i, :, :)) * theta(i, :)' + (1 / sigma_squared(i)) * (X(i, :) * beta)');
end

% Compute variance of beta
beta_variance = zeros(p, d);
for j = 1:p
    beta_variance(j, :) = sigma_squared ./ (sum(X(:, j).^2) + lambda_vals(j)^-1);
end

% Compute t-statistics
t_stats = beta ./ sqrt(beta_variance);

% Compute p-values (two-tailed)
df = N - p; % Degrees of freedom
p_values = 2 * (1 - tcdf(abs(t_stats), df));

end
