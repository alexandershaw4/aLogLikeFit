function FIM = condFIM(J,W)

% Initial settings
regularization_factor = 1e-6;   % Starting regularization factor
maxCond = 10;                 % Maximum acceptable condition number
minCond = 1;                   % Minimum acceptable condition number
increase_factor = 2;            % Factor to increase regularization if needed
decrease_factor = 0.5;          % Factor to decrease regularization if needed
maxIter = 100;                  % Maximum number of iterations
tolerance = 1e-4;               % Tolerance for stopping criterion

for iter = 1:maxIter
    % Compute Fisher Information Matrix (FIM)
    FIM = J' * W * J;            % Compute the Fisher Information Matrix
    
    % Regularize the FIM to improve conditioning
    FIM = FIM + eye(size(FIM)) * regularization_factor;
    
    % Compute the condition number of the FIM
    cond_FIM = cond(FIM);
    
    % Check if condition number is too high
    if cond_FIM > maxCond
        % If condition number is too high, increase the regularization factor
        regularization_factor = regularization_factor * increase_factor;
        fprintf('Condition number %.2f too large. Increasing regularization factor to %.2e\n', cond_FIM, regularization_factor);
    elseif cond_FIM < minCond
        % If condition number is reasonable, try reducing the regularization factor
        regularization_factor = regularization_factor * decrease_factor;
        fprintf('Condition number %.2f is good. Decreasing regularization factor to %.2e\n', cond_FIM, regularization_factor);
    end
    
    % Output condition number for monitoring
    fprintf('Iteration %d: cond(FIM) = %.4f, regularization_factor = %.2e\n', iter, cond_FIM, regularization_factor);
    
    % Stop if the condition number is within a reasonable range
    if cond_FIM <= maxCond && cond_FIM >= minCond
        fprintf('Condition number is acceptable at iteration %d.\n', iter);
        break;
    end
    
    % If regularization factor becomes too large, break the loop to avoid excessive regularization
    if regularization_factor > 1e3
        fprintf('Regularization factor too large. Breaking loop.\n');
        break;
    end
end
