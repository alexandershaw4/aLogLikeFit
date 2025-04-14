function covQ = makeposdef(covQ)
% Ensure a covariance matrix is strictly positive definite

tol = 1e-10;

if size(covQ,1) == size(covQ,2)

    %tol = 1e-10;  % small initial floor
    boost = 1e-6;
    maxBoost = 10;
    
    % Force symmetry
    covQ = (covQ + covQ') / 2;
    
    % Clean up any NaNs/Infs
    covQ(~isfinite(covQ)) = tol;
    
    % Try Cholesky and boost if needed
    while true
        [~, p] = chol(covQ);
        if p == 0
            break;  % Success!
        end
        covQ = covQ + boost * eye(size(covQ));
        boost = boost * 10;  % Increase boost exponentially
        if boost > maxBoost
            error('Could not make matrix positive definite after boosting');
        end
    end
    
    % Final symmetry just in case
    covQ = (covQ + covQ') / 2;

else
    % fallback for non-square input
    Q = covQ;
    covQ = sqrt(covQ * covQ');
    
    covQ(isnan(covQ)) = tol;
    covQ(isinf(covQ)) = tol;

    [V, D] = eig(covQ);
    D = diag(D);
    D(D < tol) = tol;
    covQ = V * diag(D) * V';
    covQ = Q + covQ;  % Adjust original with PD component
end
