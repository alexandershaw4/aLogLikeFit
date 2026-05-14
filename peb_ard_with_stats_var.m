function [beta, lambda_vals, t_stats, p_values, posterior_means, posterior_covs, free_energy, ...
          beta_sd, Pp_pos, Pp_sign] = ...
    peb_ard_with_stats_var(theta, X, Sigma_theta_prior, max_iter, tol)
% This function implements a Parametric Empirical Bayes (PEB) method for 
% estimating group-level parameters while incorporating individual-level 
% priors on the parameters. The approach combines ridge regression with 
% Bayesian regularisation, where prior covariance information about individual 
% parameters is used to shrink the group-level estimates. It also includes 
% Automatic Relevance Determination (ARD) to determine the importance of each 
% predictor. Returns the full individual level posteriors.
%
% This version accepts individual level parameter covariances (i.e. derived from the
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

% Initialize residual variance ver
%sigma_squared = ones(1, d); 
sigma_squared = ones(N, d);


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
    %lambda_vals = 1 ./ (sum(beta.^2, 2) + 1e-6);
    lambda_vals = (1 + 1e-6) ./ (sum(beta.^2,2) + 1e-6);
    lambda_vals = min(lambda_vals, 1e6);     % cap
    lambda_vals = max(lambda_vals, 1e-6);    % floor

    % Update subject- and parameter-specific residual variance
    for i = 1:N
        residuals_i = (theta(i, :) - (X(i, :) * beta))';
        sigma_squared(i, :) = (residuals_i.^2 + diag(squeeze(Sigma_theta_prior(i, :, :)))) / (1 + 1);
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

% Compute individual posterior means and covariances
posterior_means = zeros(N, d);
posterior_covs = zeros(N, d, d);

for i = 1:N
    % Subject-specific posterior covariance
    Sigma_post_inv = squeeze(Sigma_theta_prior_inv(i, :, :)) + ...
                     diag(1 ./ sigma_squared(i, :)); % Per parameter
    posterior_covs(i, :, :) = inv(Sigma_post_inv);

    % Subject-specific posterior mean
    posterior_means(i, :) = (squeeze(posterior_covs(i, :, :)) * ...
        (squeeze(Sigma_theta_prior_inv(i, :, :)) * theta(i, :)' + ...
        diag(1 ./ sigma_squared(i, :)) * (X(i, :) * beta)'))';
end

% Aggregate sigma_squared across subjects to get a single variance per parameter
sigma_squared_mean = mean(sigma_squared, 1); % (1 x d)

% ------------------------------------------------------------
% Approx posterior SD for beta and posterior probabilities
% ------------------------------------------------------------

% Use mean residual variance per parameter across subjects as a plug-in noise
sigma2 = mean(sigma_squared, 1);   % 1 x d

% Effective ridge term for each predictor from ARD:
% your coordinate update uses denom (X'X + lambda_j)
% so treat lambda_j as ridge precision-ish stabiliser
% XtX = sum(X.^2, 1);                % 1 x p
% 
% beta_var = zeros(p, d);
% for j = 1:p
%     denom = XtX(j) + lambda_vals(j);   % matches your update denominator
%     beta_var(j, :) = sigma2 ./ max(denom, 1e-12);
% end
% 
% beta_sd = sqrt(beta_var);

XtX = X' * X;                 % p x p
Lambda = diag(lambda_vals);   % p x p

A = XtX + Lambda;
A = 0.5*(A+A') + 1e-10*eye(p);    % sym + jitter
Ainv = inv(A);

beta_var = zeros(p,d);
for k = 1:d
    beta_var(:,k) = sigma2(k) * diag(Ainv);   % marginal var for each regressor
end
beta_sd = sqrt(max(beta_var, 1e-18));


% Frequentist stats (keep if you want)
t_stats = beta ./ max(beta_sd, 1e-12);
df = N - p;
p_values = 2 * (1 - tcdf(abs(t_stats), df));

% Bayesian posterior probabilities (Gaussian approx)
% P(beta > 0 | data)
Pp_pos  = 1 - normcdf(0, beta, beta_sd);

% "probability of the estimated sign" = max(P(beta>0), P(beta<0))
Pp_sign = max(Pp_pos, 1 - Pp_pos);


% % Compute variance of beta
% beta_variance = zeros(p, d);
% for j = 1:p
%     beta_variance(j, :) = sigma_squared_mean ./ (sum(X(:, j).^2) + lambda_vals(j)^-1);
% end
% 
% % Compute t-statistics
% t_stats = beta ./ sqrt(beta_variance);
% 
% % Compute p-values (two-tailed)
% df = N - p; % Degrees of freedom
% p_values = 2 * (1 - tcdf(abs(t_stats), df));

% Compute Free Energy 
log2pi = log(2 * pi);
F = 0;
for i = 1:N
    mu_q = posterior_means(i, :)';
    Sigma_q = squeeze(posterior_covs(i, :, :));
    mu_p = (X(i,:) * beta)';
    Sigma_p = diag(sigma_squared(i,:)); % Likelihood approx

    % KL[q || p]
    KL = 0.5 * ( trace(squeeze(Sigma_theta_prior_inv(i,:,:)) * Sigma_q) + ...
        (theta(i,:)' - mu_q)' * squeeze(Sigma_theta_prior_inv(i,:,:)) * (theta(i,:)' - mu_q) - ...
        d + log(det(squeeze(Sigma_theta_prior(i,:,:))) + 1e-10) - log(det(Sigma_q) + 1e-10) );

    % Expected log-likelihood
    diff = theta(i,:)' - mu_p;
    ELL = -0.5 * (d * log2pi + sum(log(sigma_squared(i,:))) + sum((diff.^2) ./ sigma_squared(i,:)'));

    F = F + ELL - KL;
end

free_energy = F;

end
