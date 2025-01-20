function [G] = radialPD(Q, sigma)
% Generates a Gaussian Radial Basis Function (RBF) Kernel Matrix
% Converts a vector Q (length n) into a smooth, radial, positive-definite n-by-n matrix.
%
% Inputs:
%   Q     - Input vector of length n.
%   sigma - Scale parameter for the Gaussian kernel (controls smoothness).
%
% Output:
%   G     - Smooth, radial, positive-definite n-by-n matrix.
%
% AS2025

% Length of the input vector
n = length(Q);

% Pairwise radial distances
[X, Y] = meshgrid(1:n, 1:n); % Create coordinate grids
D = abs(X - Y); % Compute pairwise distances

% Gaussian RBF kernel
G = exp(-D.^2 ./ (2 * sigma.^2)); % Apply Gaussian function

% Positive-definite scaling (optional)
% Ensure matrix G is symmetric and positive-definite
G = (G + G') / 2; % Symmetrize
G = G + eye(n) * 1e-6; % Add small diagonal for numerical stability

% eigendecomposition
%[V,D] = eig(G);
%D     = diag(D);



end