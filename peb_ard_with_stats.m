function [beta, lambda_vals, t_stats, p_values, posterior_means, posterior_covs] = ...
    peb_ard_with_stats(theta, X, Sigma_theta_prior, max_iter, tol)
% This function implements a Parametric Empirical Bayes (PEB) method for 
% estimating group-level parameters while incorporating individual-level 
% priors on the parameters. The approach combines ridge regression with 
% Bayesian regularization, where prior covariance information about individual 
% parameters is used to shrink the group-level estimates. It also includes 
% Automatic Relevance Determination (ARD) to determine the importance of each 
% predictor. Returns the full individual level posteriors.
%
%  [b,l,t,p,pos_mu,pos_cov] = peb_ard_with_stats(theta, X, Sigma_theta_prior, max_iter, tol)
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

% Inverse of prior covariance
Sigma_theta_prior = Sigma_theta_prior + 1e-6 * ones(d, d);
Sigma_theta_prior_inv = inv(Sigma_theta_prior);

% Initialize individual parameter covariances
sigma = diag(Sigma_theta_prior); % Start with prior variances for each parameter

for iter = 1:max_iter
    beta_old = beta;

    % Update beta using parameter-specific variances
    for j = 1:p
        beta(j, :) = (X(:, j)' * (theta - X * beta + X(:, j) * beta(j, :)) * diag(1 ./ sigma)) / ...
            (X(:, j)' * X(:, j) + lambda_vals(j));
    end

    % Update ARD hyperparameters
    lambda_vals = 1 ./ (sum(beta.^2, 2) + 1e-6);

    % Update residuals and parameter-specific variances
    for j = 1:d
        residuals_j = theta(:, j) - X * beta(:, j);
        sigma(j) = (residuals_j' * residuals_j + Sigma_theta_prior(j, j)) / (N + 1);
    end

    % Check for convergence
    if norm(beta - beta_old, 'fro') < tol
        break;
    end
end

Sigma_residual = cov(theta - X * beta);

% Compute posterior covariances and means
posterior_covs = Sigma_residual;%zeros(d, d);
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

% OLD
%------------------------------------------------------------------------
% function [beta, lambda_vals, t_stats, p_values, posterior_means, posterior_covs] = ...
%     peb_ard_with_stats(theta, X, Sigma_theta_prior, max_iter, tol)
% % This function implements a Parametric Empirical Bayes (PEB) method for 
% % estimating group-level parameters while incorporating individual-level 
% % priors on the parameters. The approach combines ridge regression with 
% % Bayesian regularization, where prior covariance information about individual 
% % parameters is used to shrink the group-level estimates. It also includes 
% % Automatic Relevance Determination (ARD) to determine the importance of each 
% % predictor. Returns the full individual level posteriors.
% %
% %  [b,l,t,p,pos_mu,pos_cov] = peb_ard_with_stats(theta, X, Sigma_theta_prior, max_iter, tol)
% %
% % Inputs:
% % - theta: Individual-level parameters (N x d)
% % - X: Design matrix (N x p)
% % - Sigma_theta_prior: Prior covariance of individual-level parameters (d x d)
% % - max_iter: Maximum number of iterations
% % - tol: Convergence tolerance
% %
% % Outputs:
% % - beta: Estimated beta coefficients (p x d)
% % - lambda_vals: ARD hyperparameters (p x 1)
% % - t_stats: t-statistics for each beta coefficient
% % - p_values: p-values for each beta coefficient
% % - posterior_means: Individual-level posterior means (N x d)
% % - posterior_covs: Posterior covariances (d x d)
% %
% % AS2024
% 
% if nargin < 4
%     max_iter = 100; % Default maximum iterations
% end
% if nargin < 5
%     tol = 1e-6; % Default convergence tolerance
% end
% 
% 
% [N, d] = size(theta);
% p = size(X, 2); % Number of predictors
% 
% % Initialize lambda and beta
% lambda_vals = ones(p, 1);
% beta = zeros(p, d);
% 
% % Inverse of prior covariance
% Sigma_theta_prior_inv = inv(Sigma_theta_prior);
% 
% for iter = 1:max_iter
%     beta_old = beta;
% 
%     % Update beta using prior covariance
%     for j = 1:p
%         beta(j, :) = (X(:, j)' * (theta - X * beta + X(:, j) * beta(j, :)) * Sigma_theta_prior_inv) / ...
%             (X(:, j)' * X(:, j) + lambda_vals(j));
%     end
% 
%     % Update ARD hyperparameters
%     lambda_vals = 1 ./ (sum(beta.^2, 2) + 1e-6);
% 
%     % Check for convergence
%     if norm(beta - beta_old, 'fro') < tol
%         break;
%     end
% end
% 
% 
% 
% 
% % Compute residuals and variance
% residuals = theta - X * beta;
% sigma_squared = var(residuals(:));
% 
% % Compute variance of beta
% beta_variance = zeros(p, d);
% for j = 1:p
%     beta_variance(j, :) = diag(Sigma_theta_prior) ./ (sum(X(:, j).^2) + lambda_vals(j)^-1);
% end
% 
% % Compute t-statistics
% t_stats = beta ./ sqrt(beta_variance);
% 
% % Compute p-values (two-tailed)
% df = N - p; % Degrees of freedom
% p_values = 2 * (1 - tcdf(abs(t_stats), df));
% 
% % Compute individual-level posterior means and covariances
% posterior_means = zeros(N, d);
% posterior_covs = zeros(d, d); % Common posterior covariance for all individuals
% 
% posterior_covs = inv(Sigma_theta_prior_inv + (1 / sigma_squared) * eye(d));
% for i = 1:N
%     posterior_means(i, :) = posterior_covs * ...
%         (Sigma_theta_prior_inv * theta(i, :)' + ((1 / sigma_squared) * X(i, :) * beta)');
% end
% end

