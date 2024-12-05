function [x_est, logL, iter, CP] = fitLogLikelihoodLM(y, f, x0, sigma, maxIter, tol, lambda0)
%Parameter estimation using log-likelihood optimisation with Levenberg-Marquardt.
%
% This function estimates model parameters by maximising the log-likelihood
% through iterative optimisation. The algorithm employs the Levenberg-Marquardt 
% method with dynamic updates of hyperparameters (observation variance and damping factor).
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

    % Compute the Jacobian matrix J
    J = computeJacobian(f, x, length(y));
    %J = J./norm(J);

    % Weight residuals by the inverse variances
    W = diag(1 ./ sigma.^2);

    % Gauss-Newton components
    H = J' * W * J;    % Approximate Hessian
    g = J' * W * residuals; % Gradient

    % Levenberg-Marquardt adjustment to Hessian
    %H_lm = H + lambda * diag(diag(H));
    H_lm = H + lambda * diag(diag(H)) + eye(size(H)) * 1e-6;

    % Parameter update
    dx = pinv(H_lm)*g;

    % Evaluate new parameters
    x_new = x + (1/8)*dx;
    y_pred_new = f(x_new);
    residuals_new = y - y_pred_new;
    logL_new = -0.5 * sum((residuals_new ./ sigma).^2 + log(2 * pi * sigma.^2));

    % Adaptive damping parameter update
    if logL_new > logL
        % Accept step, reduce damping factor
        x = x_new;
        lambda = lambda / 2;
    else
        % Reject step, increase damping factor
        lambda = lambda * 2;
    end

    fprintf('It: %d | f = %d | logLik = %d\n',iter,sum(residuals.^2),logL_new);

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
FIM = J' * W * J;


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