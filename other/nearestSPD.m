function Ahat = nearestSPD(A)
% nearestSPD - Find the nearest Symmetric Positive Definite matrix to A
%
% If A is not positive definite, this function finds the nearest symmetric
% positive definite matrix using Higham’s method.
%
% Reference: Higham, N. J. (1988). Computing a nearest symmetric positive semidefinite matrix.
%
% AS2025

    if issymmetric(A) && all(eig(A) > 0)
        Ahat = A;
        return;
    end

    % Step 1: Make symmetric
    B = (A + A') / 2;

    % Step 2: Project to positive semidefinite via eigenvalue decomposition
    [V, D] = eig(B);
    D = diag(D);
    D(D < 0) = 0;  % Zero out negative eigenvalues
    Ahat = V * diag(D) * V';

    % Step 3: Ensure symmetry again
    Ahat = (Ahat + Ahat') / 2;

    % Step 4: Add jitter if needed
    k = 0;
    while ~isPositiveDefinite(Ahat)
        k = k + 1;
        minEig = min(eig(Ahat));
        Ahat = Ahat + (10^(-6) - minEig) * eye(size(A));
        if k > 5
            warning('nearestSPD: matrix adjusted multiple times');
            break;
        end
    end
end

function tf = isPositiveDefinite(A)
    [~, p] = chol(A);
    tf = (p == 0);
end
