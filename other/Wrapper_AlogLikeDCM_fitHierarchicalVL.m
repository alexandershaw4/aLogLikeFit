function results = Wrapper_AlogLikeDCM_fitHierarchicalVL(DCMs,X,maxIter, tol, nIter)
% Run hierarchical variational Laplace fitting on multiple DCMs
%
% Usage:
%   results = Wrapper_AlogLikeDCM_fitHierarchicalVL(DCMs, X, maxIter, tol, nIter)
%
% Inputs:
%   DCMs     - Cell array of file paths to DCM .mat files (each containing a 'DCM' struct)
%   X        - Design matrix (between-subject covariates or regressors)
%   maxIter  - Maximum number of outer iterations for the hierarchical loop (default: 4)
%   tol      - Convergence tolerance for model evidence change (default: 1e-6)
%   nIter    - Number of iterations for individual fits in each hierarchical step (default: 4)
%
% Output:
%   results  - Struct containing the output of fitHierarchicalVL, including fitted parameters
%              and model evidence estimates across levels
%
% Description:
%   This function wraps a hierarchical Bayesian fitting routine for a group of Dynamic Causal Models (DCMs).
%   It extracts data from each DCM, initializes an aFitDCM model, constructs the likelihood function,
%   and applies a low-rank variational Laplace procedure using `fitHierarchicalVL`.
%
% Dependencies:
%   Requires SPM for spm_vec, and the presence of 'aFitDCM' and 'fitHierarchicalVL' functions.
%
% Example:
%   results = Wrapper_AlogLikeDCM_fitHierarchicalVL({'subj1.mat','subj2.mat'}, X);
%
% AS2025

if nargin < 5 || isempty(nIter)
    nIter = 4;
end

if nargin < 4 || isempty(tol)
    tol = 1e-6;
end

if nargin < 3|| isempty(maxIter)
    maxIter = 4;
end

% get data
for i = 1:length(DCMs)
    load(DCMs{i},'DCM');
    y{i} = spm_vec(DCM.xY.y);
end

% initiate an aFitDCM object
load(DCMs{1},'DCM');
M = aFitDCM(DCM);

% get initial points, function handle etc
x0  = M.opts.x0(:);
fun = @(varargin)M.wrapdm(varargin{:});

%x0 = spm_vec(obj.DCM.M.pE);
V  = diag(M.opts.V );

% run fitHierarchicalVL (calls fitVL_LowRankNoise)
results = fitHierarchicalVL(y, fun, x0, V, maxIter, tol, nIter, X)