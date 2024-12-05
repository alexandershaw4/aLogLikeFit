function [x_est, freeE, iter] = fitFreeEnergyLM(y, f, x0, sigma, sigma_prior, maxIter, tol, lambda0)
% Estimate parameters using free energy with Levenberg-Marquardt optimisation.
%
% Includes online updating of hyperparameters (sigma & lambda).
%
%   [X, freeE, it] = fitLogLikelihoodLM(y, f, x0, sigma, maxIter, tol, lambda0)
%
% Inputs:
%   y          - Observed data (vector)
%   f          - Model function handle (f(x) returns predicted values)
%   x0         - Initial guess for parameters (vector)
%   sigma      - Standard deviations of observations (vector or scalar)
%   mu_prior   - Prior mean for parameters (vector)
%   sigma_prior - Prior standard deviation for parameters (vector or scalar)
%   maxIter    - Maximum number of iterations for LM
%   tol        - Convergence tolerance for stopping criterion
%   lambda0    - Initial damping parameter (scalar)
%
% Outputs:
%   x_est    - Estimated parameters
%   freeE    - Final negative free energy value
%   iter     - Number of iterations performed
%
% AS2024

% Initialize parameters
x = x0(:);
lambda = lambda0;  % Initial damping factor
n = length(y);
figure('position',[888   744   847   564]);

% Initialization of priors
mu_prior = x(:);
sigma_prior = sqrt(sigma_prior(:));

for iter = 1:maxIter
    % Predicted values from model
    y_pred = f(x);
    
    % Residuals
    residuals = y - y_pred;
    
    % Update sigma (dynamic estimate of residual variance)
    sigma = max(sqrt(mean(residuals.^2)), 1e-6) * ones(size(sigma));
    
    % Compute Accuracy (Data Term)
    accuracy = -0.5 * sum((residuals ./ sigma).^2 + log(2 * pi * sigma.^2));
    
    % Compute Complexity (Prior Regularization Term)
    complexity = -0.5 * sum(((x - mu_prior) ./ sigma_prior).^2 + log(2 * pi * sigma_prior.^2));
    
    % Compute Precision term
    %precision = -0.5 * sum(log(diag(P))); % Assuming diagonal precision matrix

    % Compute Free Energy
    freeE =  accuracy + complexity;
    %freeE = beta(1) * accuracy + beta(2) * complexity + beta(3) * precision;
    
    % Compute the Jacobian matrix J
    J = computeJacobian(f, x, length(y));
    
    % Weight residuals by the inverse variances
    W = diag(1 ./ sigma.^2);
    
    % Gauss-Newton components
    H = J' * W * J;    % Approximate Hessian
    g = J' * W * residuals - diag(1 ./ sigma_prior.^2) * (x - mu_prior); % Gradient with prior
    %g = J' * W * residuals - beta(2) * diag(1 ./ sigma_prior.^2) * (x - mu_prior);

    % Levenberg-Marquardt adjustment to Hessian
    H_lm = H + lambda * diag(diag(H)) + eye(size(H)) * 1e-6;
    
    % Parameter update
    dx = pinv(H_lm) * g;
    
    % Evaluate new parameters
    x_new = x + dx;
    
    % Recalculate residuals and free energy
    y_pred_new = f(x_new);
    residuals_new = y - y_pred_new;
    accuracy_new = -0.5 * sum((residuals_new ./ sigma).^2 + log(2 * pi * sigma.^2));
    complexity_new = -0.5 * sum(((x_new - mu_prior) ./ sigma_prior).^2 + log(2 * pi * sigma_prior.^2));
    %precision_new = -0.5 * sum(log(diag(P)));
    freeE_new =  accuracy_new + complexity_new;
    %freeE_new = beta(1) * accuracy_new + beta(2) * complexity_new + beta(3) * precision_new;

    % Adaptive damping parameter update
    if (accuracy_new > accuracy) && (complexity_new <= complexity)
        % Accept step, reduce damping factor
        x = x_new;
        lambda = lambda / 2;
    else
        % Reject step, increase damping factor
        lambda = lambda * 2;
        sigma_prior = sigma_prior + sqrt(1/8);
        sigma = sigma + sqrt(1/8);
    end

    % Dynamic beta update based on term ratios
    %beta(1) = 1 / (1 + exp(-alpha * abs(accuracy_new / complexity_new)));
    %beta(2) = 1 - beta(1);
    %beta(3) = min(max(0.1, 1 - beta(1) - beta(2)), 0.8); % Keep beta(3) constrained
    

    fprintf('It: %d | f = %d | F.E. = %d\n',iter,sum(residuals.^2),freeE_new);
    fprintf('It: %d | accuracy = %d | complexity = %d\n',iter,accuracy_new,complexity_new);

    % Show
    w = 1:length(y);
    plot(w,y,':k',w,y_pred,'b',w,y_pred_new,'r','linewidth',2);
    drawnow;
    
    % Check for convergence
    if norm(dx) < tol
        break;
    end
end

% Final parameter estimates
x_est = x;

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