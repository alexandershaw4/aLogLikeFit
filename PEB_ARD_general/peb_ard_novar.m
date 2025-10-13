function M = peb_ard_novar(y, X, opts)
% PEB/ARD Bayesian ridge without first-level variances (robust).
% - Prunes collinear / zero-variance features via rank-revealing QR
% - Solves with Cholesky + jitter; falls back to SVD SPD-repair if needed
% - Multi-output regression supported (shared λ by default)
%
% Usage:
%   M = peb_ard_novar(y, X);
%   M = peb_ard_novar(y, X, struct('standardise',true,'tie_lambdas',true));
%
% Inputs:
%   y  : N×d targets (vector OK -> N×1)
%   X  : N×p design (include column of ones if you want an intercept)
%   opts (optional fields):
%       .max_iter      (500)
%       .tol           (1e-6)
%       .standardise   (true)
%       .tie_lambdas   (true)   % share sparsity across outputs
%       .lambda_floor  (1e-9)
%       .lambda_ceil   (1e9)
%       .jitter0       (1e-9)
%
% Outputs (key):
%   M.beta        : p_kept×d posterior means (original scale, pruned cols only)
%   M.beta_std    : p_kept×d posterior means (standardised X/y space)
%   M.Vbeta       : p_kept×p_kept posterior covariance (standardised space)
%   M.lambda      : p_kept×1 ARD precisions
%   M.sigma2      : 1×d noise variances (original y units)
%   M.gamma       : p_kept×1 effective dof
%   M.logev       : scalar (approx) log-evidence
%   M.pvals       : p_kept×d Wald p-values (Normal approx)
%   M.kept_columns: indices of X columns kept (after pruning)
%   M.x_mean/x_std, M.y_mean/y_std, flags for standardisation
%
% Alexander D. Shaw / 2025, robust ARD PEB variant

if nargin < 3, opts = struct; end
if isvector(y), y = y(:); end
[N, d] = size(y);
p0 = size(X, 2);

% --- options
max_iter   = get_opt(opts,'max_iter',500);
tol        = get_opt(opts,'tol',1e-6);
do_std     = get_opt(opts,'standardise',true);
tie_lam    = get_opt(opts,'tie_lambdas',true);
lam_floor  = get_opt(opts,'lambda_floor',1e-9);
lam_ceil   = get_opt(opts,'lambda_ceil',1e9);
jitter0    = get_opt(opts,'jitter0',1e-9);

% --- remove bad rows
bad = any(~isfinite(X),2) | any(~isfinite(y),2);
if any(bad), X(bad,:) = []; y(bad,:) = []; [N,~] = size(X); end

% --- detect intercept, prune zero-variance columns (but keep intercept)
is_onecol = all(abs(X - 1) < 1e-12, 1);
sx = std(X,0,1);
keep0 = is_onecol | (sx > 0);
if ~all(keep0)
    X  = X(:, keep0);
    is_onecol = is_onecol(keep0);
end
kept_idx = find(keep0);
p = size(X,2);

% --- standardise (leave intercept as ones)
x_mean = mean(X,1);
x_std  = std(X,0,1); x_std(x_std==0) = 1;
if do_std
    Xs = (X - x_mean)./x_std;
    Xs(:, is_onecol) = 1;                 % keep exact 1 intercept
    y_mean = mean(y,1);
    y_std  = std(y,0,1); y_std(y_std==0)=1;
    ys = (y - y_mean)./y_std;
else
    Xs = X; ys = y;
    y_mean = zeros(1,d); y_std = ones(1,d);
end

% --- rank-revealing QR prune (keeps an intercept if present)
[Qqr, Rqr, Eqr] = qr(Xs, 0);
rdiag = abs(diag(Rqr));
tolQR = max(size(Xs)) * eps(max(rdiag));
keepQR = rdiag > tolQR;
% ensure an intercept is kept if present
is_onecol_std = all(abs(Xs - 1) < 1e-12, 1);
if any(is_onecol_std)
    % find position in permutation Eqr corresponding to an intercept col
    onepos = find(is_onecol_std(Eqr), 1, 'first');
    if ~isempty(onepos), keepQR(onepos) = true; end
end
Xs = Xs(:, Eqr(keepQR));
% update kept mapping to original X columns
kept_idx = kept_idx(Eqr(keepQR));
p = size(Xs,2);

% --- precompute
XT  = Xs.';
XTX = XT*Xs;      % p×p
XTy = XT*ys;      % p×d

% --- init params
lambda = ones(p,1);
sigma2 = max(var(ys,0,1), 1e-6);   % 1×d
beta_std = zeros(p,d);
logev = -inf;

% --- coordinate ascent
for it = 1:max_iter
    lambda_old = lambda;
    sigma2_old = sigma2;

    if tie_lam
        s2 = mean(sigma2);
        A = diag(lambda) + (1/s2)*XTX;
        [beta_std, diagV] = spd_solve(A, XTy / s2); % p×d, p×1
    else
        beta_std = zeros(p,d);
        diagV = zeros(p,1);
        for j = 1:d
            A = diag(lambda) + (1/sigma2(j))*XTX;
            [bj, dj] = spd_solve(A, XTy(:,j) / sigma2(j));
            beta_std(:,j) = bj;
            diagV = diagV + dj;
        end
        diagV = diagV / d; % average
    end

    % ARD gamma and updates
    gamma = max(0, 1 - lambda .* diagV);              % p×1
    denom = max(1, N - sum(gamma));                   % scalar
    resid = ys - Xs*beta_std;                         % N×d
    sigma2 = sum(resid.^2, 1) ./ denom;               % 1×d
    sigma2(~isfinite(sigma2)) = sigma2_old(~isfinite(sigma2));
    sigma2 = max(sigma2, 1e-10);

    b2 = mean(beta_std.^2, 2);                        % p×1
    lambda = gamma ./ max(b2, 1e-12);
    lambda = min(max(lambda, lam_floor), lam_ceil);
    if any(~isfinite(lambda)), lambda = lambda_old; end

    % convergence
    dlam = norm((lambda - lambda_old)./(lambda_old+1e-12));
    dsig = norm((sigma2 - sigma2_old)./(sigma2_old+1e-12));
    if max(dlam, dsig) < tol, break; end
end

% --- final covariance & evidence (in std space)
s2 = mean(sigma2);
A = diag(lambda) + (1/s2)*XTX;
[beta_std, ~, Vbeta_full] = spd_solve(A, XTy / s2);   % p×d, p×p

% approximate log-evidence sum across outputs
logdetA = safe_logdet(A);
logev = 0;
for j = 1:d
    rj2 = sum((ys(:,j) - Xs*beta_std(:,j)).^2);
    logev = logev + 0.5*( p*mean(log(lambda+1e-12)) + N*log(1/sigma2(j)) ...
        - (1/sigma2(j))*rj2 - beta_std(:,j)'*diag(lambda)*beta_std(:,j) ...
        - logdetA - N*log(2*pi) );
end

% --- unstandardise β back to original scale
beta = zeros(p,d);
for j = 1:d
    %beta(:,j) = (y_std(j)./x_std(kept_idx(:))) .* beta_std(:,j);
    scale = (y_std(j) ./ x_std(kept_idx)).';   % p×1 column
    beta(:,j) = scale .* beta_std(:,j);        % p×1
end

% implied intercept (if standardised)
intercept = [];
if do_std
    b0 = y_mean - ((x_mean(kept_idx)./x_std(kept_idx)) * beta_std) .* y_std;
    intercept = b0; %#ok<NASGU> % (not stored, but computed if needed)
end

% Wald z & p (std scale)
se = sqrt(max(diag(Vbeta_full), 1e-16));
z  = beta_std ./ se;
pvals = 2*normcdf(-abs(z));

% --- pack model
M.beta         = beta;                     % original scale, kept cols only
M.beta_std     = beta_std;                 % std scale
M.Vbeta        = Vbeta_full;               % std scale covariance
M.lambda       = lambda(:);
M.sigma2       = sigma2 .* (y_std.^2);     % original y units
M.gamma        = gamma(:);
M.logev        = logev;
M.z            = z;
M.pvals        = pvals;
M.used_intercept = any(is_onecol);
M.kept_columns = kept_idx(:);
M.x_mean = x_mean(kept_idx);
M.x_std  = x_std(kept_idx);
M.y_mean = y_mean;
M.y_std  = y_std;
M.standardised = do_std;
M.tie_lambdas  = tie_lam;

M = peb_reorder_to_original(M, size(X,2));


% % make ordering more explicit
% M.beta_ordered = zeros(size(X,2),1);
% M.beta_ordered(M.kept_columns) = M.beta;
% 
% M.beta_ordered_std = zeros(size(X,2),1);
% M.beta_ordered_std(M.kept_columns) = M.beta_std;

end

% ----------------- helpers -----------------
function v = get_opt(S,f,def)
if isfield(S,f) && ~isempty(S.(f)), v = S.(f); else, v = def; end
end

function [X, diagInvA, Ainv] = spd_solve(A, B)
% Robust SPD solve: try chol+jitter; else SVD repair.
% escalate jitter for Cholesky
jitter = 1e-9; R = []; flag = 1;
for k = 1:10
    [R, flag] = chol(A + jitter*eye(size(A)));
    if flag == 0, break; end
    jitter = jitter * 10;
end
if flag == 0
    % chol path
    X = R \ (R' \ B);
    Ri = R \ eye(size(A));
    if nargout >= 2
        diagInvA = sum(Ri.^2, 2);
    end
    if nargout >= 3
        Ainv = Ri * Ri.';
    end
    return;
end
% SVD SPD-repair
As = (A + A')/2;
[U,S,V] = svd(As,'econ');
s = diag(S);
s = max(s, 1e-8);
Ainv = V * (bsxfun(@rdivide, U', s'));   % inv(A) ≈ V * diag(1/s) * U'
X = Ainv * B;
if nargout >= 2
    diagInvA = sum((V.^2) .* (1./s'), 2);
end
end

function L = safe_logdet(A)
% Stable log|A| using chol if possible, else SVD
[R,flag] = chol(A);
if flag==0
    L = 2*sum(log(diag(R)+eps));
else
    s = svd((A + A')/2);
    s = max(s, 1e-12);
    L = sum(log(s));
end
end
