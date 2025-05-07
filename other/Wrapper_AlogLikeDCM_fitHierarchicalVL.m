function results = Wrapper_AlogLikeDCM_fitHierarchicalVL(DCMs,X,maxIter, tol, nIter)

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