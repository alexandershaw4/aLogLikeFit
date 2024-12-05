function [x_est, logL, iter, CP] = fitLogLikelihoodLMFE(y, f, x0, sigma, maxIter, tol, lambda0,sigma_prior)
% Estimate parameters using log-likelihood and free energy approach
% with Levenberg-Marquardt optimisation.
%
% Includes online updating of hyperparameters (sigma & lambda).
%
%  [X, LogL, it] = fitLogLikelihoodLM(y, f, x0, sigma, maxIter, tol, lambda0)
%
% Inputs:
%   y        - Observed data (vector)
%   f        - Model function handle (f(x) returns predicted values)
%   x0       - Initial guess for parameters (vector)
%   sigma    - Standard deviations of observations (vector or scalar)
%   maxIter  - Maximum number of iterations for LM
%   tol      - Convergence tolerance for stopping criterion
%   lambda0  - Initial damping parameter (scalar)
%   sigma_prior - prior variances on parameters
%
% Outputs:
%   x     - Estimated parameters
%   LogL  - Final log-likelihood value
%   it    - Number of iterations performed
%
% AS2024
    
% Initialize parameters
x = x0(:);
lambda = lambda0;  % Initial damping factor
n = length(y);
figure('position',[888   744   847   564]);

y_pred = f(x);
residuals = y - y_pred;
mu_prior = x0(:);
lr = 1/32;
sigma_prior = full(sigma_prior);

fprintf('Initialise\n');
fprintf('It: %d | f = %d\n',0,sum(residuals.^2));

for iter = 1:maxIter
    % Predicted values from model
    y_pred = f(x);

    % Residuals
    residuals = y - y_pred;

    % Update sigma (dynamic estimate of residual variance)
    sigma = max(sqrt(mean(residuals.^2)), 1e-6) * ones(size(sigma));

    % Compute the log-likelihood
    logL = -0.5 * sum((residuals ./ sigma).^2 + log(2 * pi * sigma.^2));

    %complexity = -0.5 * sum(abs((x - mu_prior) ./ sigma_prior).^2 + log(2 * pi * sigma_prior.^2));
    %complexity = mvgkl(x,diag(sigma_prior),mu_prior,diag(sigma_prior));
    complexity = -0.5 * (sum(((x - mu_prior)./sigma_prior).^2 ) + log(2 * pi * prod(sigma_prior)));

    free_en = logL + complexity;

    % Compute the Jacobian matrix J
    J = computeJacobian(f, x, length(y));
    
    %for i = 1:size(J,2)
    %    J(:,i) = J(:,i)./norm(J(:,i));
    %end

    % Weight residuals by the inverse variances
    W = diag(1 ./ sigma.^2);

    % Gauss-Newton components
    H = J' * W * J;    % Approximate Hessian
    g = J' * W * residuals; % Gradient

    %sigma_prior = sigma_prior - diag(pinv(H)./norm(pinv(H)));

    %g = J' * W * residuals - diag(1 ./ sigma_prior) * (x - mu_prior); % Gradient with prior
    g = J' * W * residuals + pinv(diag(sigma_prior)) * (mu_prior - x);  % Correct direction of prior regularization

    % Levenberg-Marquardt adjustment to Hessian
    %H_lm = H + lambda * diag(diag(H));
    %H_lm = H + lambda * diag(diag(H));% + eye(size(H)) * 1e-6;
    H_lm = H + lambda * eye(size(H)) + 1e-8 * diag(diag(H));

    % Parameter update
    dx = pinv(H_lm)*g;
    %dx = -spm_dx(H_lm,g,{-4});

    % Evaluate new parameters
    x_new = x + lr * dx;
    y_pred_new = f(x_new);
    residuals_new = y - y_pred_new;
    logL_new = -0.5 * sum((residuals_new ./ sigma).^2 + log(2 * pi * sigma.^2));

    %complexity_new = -0.5 * sum(abs((x_new - mu_prior) ./ sigma_prior).^2 + log(2 * pi * sigma_prior.^2));
    %complexity_new = mvgkl(x_new,diag(sigma_prior),mu_prior,diag(sigma_prior));
    complexity_new = -0.5 * (sum(((x_new - mu_prior)./sigma_prior).^2 ) + log(2 * pi * prod(sigma_prior)));

    free_en_new = logL_new + complexity_new;

    % Adaptive damping parameter update
    if free_en_new > free_en
        % Accept step, reduce damping factor
        x = x_new;
        lr = lr * 2;
        %lambda = lambda / 2;
    else
        % Reject step, increase damping factor
            lr = lr / 4;
           
        %lambda = lambda / 2;
        %lr = lr / 2;
    end

    % clamp lr
    lr = min(max(lr, 1e-8), 1);

    fprintf('It: %d | sse = %d | logLik = %d | complexity = %d\n',iter,sum(residuals.^2),logL_new,complexity_new);

    % Show
    w = 1:length(y);
    plot(w,y,':k',w,y_pred,w,y_pred_new,'linewidth',2);
    drawnow;


    % Check for convergence
    if norm(dx) < tol
        break;
    end

end

% Final parameter estimates
x_est = x;

% Compute FIM and posterior covariance
J = computeJacobian(f, x, length(y));  % Jacobian at current estimate
W = diag(1 ./ sigma.^2);               % Weight matrix

% Fisher Information Matrix
FIM = J' * W * J + inv(diag(sigma_prior)); 


% Posterior covariance
CP = pinv(FIM);

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