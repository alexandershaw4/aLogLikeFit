function [m, V, D, logL, iter, sigma2, allm, all_elbo] = ...
    fitVariationalLaplaceThermo_GC(y, f, m0, S0, maxIter, tol, opts)
% Thermodynamic Variational Laplace in *Generalised Coordinates*.
% See fitVariationalLaplaceThermo.m for original implementation.
%
% This version lifts residuals and Jacobians into generalised coordinates of motion:
%   r^(0:q) = [r, Dr, D^2 r, ...], with D approximated via finite-difference
%   (or supplied by the model). The cost becomes weighted across orders via Γ.
%
% Inputs:
%   y      : observed vector (length T) OR matrix (T×d stacked columnwise*). We treat it as vec.
%   f      : model handle. Must support:
%            yhat = f(m)
%            [yhat, dY] = f(m)  % OPTIONAL: dY is [T×(q)] or cell{q} of derivatives along sample axis
%   m0, S0 : prior mean and covariance
%   maxIter, tol : as before
%   opts.gc : struct with fields:
%       .order   (default 2)   – highest generalised order q
%       .dt      (default 1)   – sample spacing along the axis to differentiate
%       .lambda  (default [])  – per-order weights Γ_k; if empty, uses dt-scaled [1, 1/dt^2, 1/dt^4, ...]
%       .op      ('fd2')       – derivative operator: 'fd1' (first-order), 'fd2' (central, order 2/4 mix)
%       .boundary('rep')       – boundary handling: 'rep' (replicate), 'mirror', 'zero'
%   opts.varpercthresh : relative eigen threshold for low-rank precision (default 0.01)
%   opts.plots         : 0/1 quick progress plots (default 0)
%
% Outputs:
%   m, V, D, logL, iter, sigma2, allm, all_elbo 
%
% AS2025 http://cpnslab.com


% ---------------- Defaults ----------------
if nargin < 7 || isempty(opts), opts = struct; end
if ~isfield(opts,'plots'),            opts.plots = 1;            end
if ~isfield(opts,'varpercthresh'),    opts.varpercthresh = 0.01; end
if ~isfield(opts,'gc'),               opts.gc = struct;          end

gc = opts.gc;

if ~isfield(gc,'order'),    gc.order = 3;        end
if ~isfield(gc,'dt'),       gc.dt    = 1;        end
if ~isfield(gc,'lambda'),   gc.lambda = [];      end
if ~isfield(gc,'op'),       gc.op    = 'fd2';    end
if ~isfield(gc,'boundary'), gc.boundary = 'rep'; end

q   = max(0, round(gc.order));
dt  = gc.dt;

% per-order weights Γ_k: [1, 1/dt^2, 1/dt^4, ...]
if isempty(gc.lambda)
    lam = (1 ./ (dt.^(2*(0:q)))).';
else
    lam = gc.lambda(:);
    if numel(lam) ~= (q+1)
        error('opts.gc.lambda must have q+1 entries (order 0..q).');
    end
end

% ---------------- Setup ----------------
y   = y(:);                 % vectorise
T   = numel(y);
m   = m0(:);
n   = T*(q+1);              % stacked residual length

% Build GC derivative operators (E{1}=I, E{2}=D, ..., E{q+1}=D^q)
E = build_gc_operators(T, q, dt, gc.op, gc.boundary);

% Weights across orders (Γ = I_T ⊗ diag(lam))
Gamma     = kron(spdiags(lam,0,q+1,q+1), speye(T));             % (q+1)T × (q+1)T
sqrtGamma = spdiags(sqrt(repmat(lam,T,1)),0,(q+1)*T,(q+1)*T);

% Low-rank init from S0 (on parameters)
[U0, S0val] = svd(full(S0), 'econ');
eigs0       = diag(S0val);
k           = max(sum(eigs0 > opts.varpercthresh*max(eigs0)), numel(m));
V           = U0(:,1:k) * diag(sqrt(eigs0(1:k)));
D           = diag(diag(S0) - sum(V.^2,2));

% Noise & tracking
epsilon = 1e-6; beta = 1e-3; nu = 3;
sigma2  = ones(n,1);               % heteroscedastic *in GC space*
allm = m; all_elbo = []; best_elbo = -Inf;
best_iter = 1; sigma2_best = sigma2;
if opts.plots, fw = figure('position',[140 140 1550 820]); end

% Prior precision helper
P = size(S0,1);
K = computeSmoothCovariance(P, 2);          % K is P×P now
A = (S0 + K);                                % keep symmetric
% robust PD solve for prior precision without forming inv()
try
    R = chol(A,'lower');
catch
    R = chol(A + 1e-8*eye(P),'lower');
end
H_prior = R'\(R\eye(P));          

% ---------------- Main loop ----------------
for iter_i = 1:maxIter
    % Model predictions (optionally derivatives from model)
    try
        [yhat, dY_model] = f(m);
    catch
        yhat = f(m);
        dY_model = [];
    end
    yhat = yhat(:);
    if numel(yhat) ~= T
        error('Model f(m) must return a vector of length T to align with y.');
    end

    % Stack residuals & retain per-order y / yhat for plotting
    r0 = y - yhat;
    R  = r0;
    y_orders    = cell(q+1,1); y_orders{1} = y;
    yhat_orders = cell(q+1,1); yhat_orders{1} = yhat;

    for kOrd = 1:q
        if ~isempty(dY_model) && size(dY_model,1)==T && size(dY_model,2)>=kOrd
            yhat_k = dY_model(:,kOrd);    % model predicted derivative order k
            y_k    = E{kOrd+1} * y;       % numeric derivative of data (for comparison)
            rk     = y_k - yhat_k;
        else
            y_k    = E{kOrd+1} * y;
            yhat_k = E{kOrd+1} * yhat;
            rk     = y_k - yhat_k;
        end
        y_orders{kOrd+1}    = y_k;
        yhat_orders{kOrd+1} = yhat_k;
        R = [R; rk]; %#ok<AGROW>
    end

    % Heteroscedastic variance (GC-space t-like)
    sigma2 = max(epsilon, (R.^2 + beta) ./ (nu + R.^2/2));

    % Weighted residuals & log-likelihood
    Wsqrt  = spdiags(1./sqrt(max(epsilon,sigma2)), 0, n, n);
    RW     = Wsqrt * R;
    RWg    = sqrtGamma * RW;
    logL_lik = -0.5 * (sum((R.^2)./sigma2) + sum(log(2*pi*sigma2))) ...
               -0.5 * sum(log(lam + eps));  % constant in params; ok to include/exclude consistently

    % Jacobian in GC: J0 = dyhat/dm (T×P). Then stack [J0; E2*J0; ...; E{q+1}*J0]
    J0 = computeJacobian(@(mm) f(mm), m, T);   % helper vectorises outputs
    J_gc = J0;
    for kOrd = 1:q
        J_gc = [J_gc; E{kOrd+1}*J0]; %#ok<AGROW>
    end

    % ---- Sanity checks on shapes ----
    if size(J_gc,1) ~= (q+1)*T
        error('GC Jacobian has wrong number of rows: got %d, expected %d.', ...
              size(J_gc,1), (q+1)*T);
    end
    if size(Wsqrt,1) ~= (q+1)*T
        error('Wsqrt dimension mismatch: got %d, expected %d.', size(Wsqrt,1), (q+1)*T);
    end

    % Weighted Jacobian
    JW  = Wsqrt * J_gc;          % (n×P)
    JWG = sqrtGamma * JW;        % ((q+1)T × P)

    % Precision (Gauss–Newton) with prior
    H    = JWG' * JWG;
    g    = JWG' * RWg  -  H_prior*(m - m0);

    % Low-rank precision factor update
    [U,Sv] = svd(H + H_prior, 'econ');
    Vr     = U(:,1:k) * sqrt(Sv(1:k,1:k));
    Dr     = diag(diag(H + H_prior) - sum(Vr.^2,2));
    Dr     = max(Dr, 1e-8);

    H_elbo = Vr*Vr' + diag(Dr);
    dm = solve_pd(H_elbo, g);

    % Trust region
    maxStep = 1.0;
    if norm(dm) > maxStep, dm = dm * (maxStep/norm(dm)); end

    m_prev = m;
    m      = m + dm;
    allm   = [allm m]; %#ok<AGROW>

    % ELBO components
    logL_prior   = -0.5 * ((m - m0)' * H_prior * (m - m0));
    % try
    %     Lchol = chol(H_elbo,'lower');
    %     logdetH = 2*sum(log(diag(Lchol)));
    % catch
    %     Lchol = chol(H_elbo + 1e-6*eye(size(H_elbo)),'lower');
    %     logdetH = 2*sum(log(diag(Lchol)));
    % end
    % logL_entropy = -0.5 * logdetH;


    % --- Entropy term: log |H_elbo| with robust PD handling ---

    % Symmetrise to remove tiny asymmetries
    H_elbo = (H_elbo + H_elbo')/2;

    P = size(H_elbo,1);
    I = eye(P);

    success = false;
    logdetH = NaN;

    % 1) Try plain Cholesky
    try
        Lchol   = chol(H_elbo, 'lower');
        logdetH = 2*sum(log(diag(Lchol)));
        success = true;
    catch
        % fall through
    end

    % 2) Try Cholesky with adaptive jitter if needed
    if ~success
        diagH   = diag(H_elbo);
        baseJit = 1e-6 * max(1, max(abs(diagH)));
        jitter  = baseJit;

        for attempt = 1:5
            try
                Lchol   = chol(H_elbo + jitter*I, 'lower');
                logdetH = 2*sum(log(diag(Lchol)));
                success = true;
                break;
            catch
                jitter = jitter * 10;
            end
        end
    end

    % 3) Final fallback: eigendecomposition with eigenvalue clamping
    if ~success
        [Q, Lambda] = eig(H_elbo);
        lam         = diag(Lambda);

        % Clamp negatives / zeros
        lam = max(lam, 1e-8);

        % log |H_elbo| = sum log lambda_i
        logdetH = sum(log(lam));

        % (We don't actually need Lchol downstream, only logdetH)
    end

    logL_entropy = -0.5 * logdetH;



    logL = logL_lik + logL_prior + logL_entropy;
    all_elbo = [all_elbo, logL]; %#ok<AGROW>

    % Backtracking if ELBO drops
    if numel(all_elbo) > 1 && logL < all_elbo(end-1)
        step = 0.5; improved = false; tries = 0;
        while ~improved && tries < 8
            tries = tries+1;
            m_try = m_prev + (step^tries)*dm;
            [logL_try] = quick_elbo_gc(y, f, m_try, T, E, Wsqrt, sqrtGamma, H_prior, q);
            if logL_try > logL
                m = m_try; logL = logL_try; improved = true;
                all_elbo(end) = logL;
            end
        end
        if ~improved
            m = m_prev; logL = all_elbo(end-1);
            all_elbo(end) = logL;
        end
    end

    % ====== Track best posterior ======
    if iter_i == 1 || logL > best_elbo
        best_elbo   = logL;
        m_best      = m;
        V_best      = Vr;
        D_best      = Dr;
        sigma2_best = sigma2;
        best_iter   = iter_i;
        y_orders_best    = y_orders;    %#ok<NASGU>
        yhat_orders_best = yhat_orders; %#ok<NASGU>
    end

    % ====== Per-order plots (0..q) ======
    if opts.plots
        figure(fw); clf;

        nCols = max(q+1, 2);
        nRows = 2;
        t = 1:T;

        % Per-order fits
        for kOrd = 0:q
            subplot(nRows, nCols, kOrd+1);
            yo  = y_orders{kOrd+1};
            yho = yhat_orders{kOrd+1};
            plot(t, yo, 'k', 'LineWidth', 1); hold on;
            plot(t, yho, 'r', 'LineWidth', 1.5); hold off;
            if kOrd==0
                ttl = 'Order 0 (signal)';
            else
                ttl = sprintf('Order %d (D^{%d}y)', kOrd, kOrd);
            end
            title(ttl);
            grid on; box on;
            xlabel('Sample'); ylabel('Value');
            legend({'Observed','Model'},'Location','best');
        end

        % ELBO trajectory
        subplot(nRows, nCols, nCols+1);
        plot(1:numel(all_elbo), all_elbo, '-o'); title('ELBO'); grid on; box on;
        xlabel('Iteration');

        % Step size
        subplot(nRows, nCols, nCols+2);
        semilogy(abs(dm),'o'); title('|dm|'); grid on; box on;
        xlabel('Parameter index');

        drawnow;
    end

    % Convergence
    if norm(dm) < tol
        fprintf('Converged @ %d\n', iter_i);
        break;
    end

    fprintf('it %3d | ELBO %.4f | ||dm||=%.3e\n', iter_i, logL, norm(dm));
end

% ====== Return the best posterior ======
m     = m_best;
V     = V_best;
D     = D_best;
sigma2= sigma2_best;
logL  = best_elbo;
iter  = best_iter;

end

% ---------- helpers ----------

function E = build_gc_operators(T, q, dt, scheme, bnd)
% Builds E{1}=I, E{2}=D, ..., E{q+1}=D^q with chosen finite-difference scheme.
I = speye(T);
E = cell(q+1,1); E{1} = I;   % order 0
if q==0, return; end
D = fd_operator(T, dt, scheme, bnd);  % first derivative
E{2} = D;
for k=3:q+1
    E{k} = D*E{k-1};
end
end

function D = fd_operator(T, dt, scheme, bnd)
% Central differences (default) with simple boundary handling.
e = ones(T,1);
switch scheme
    case 'fd1' % forward 1st order
        D = spdiags([-e e],[0 1],T,T)/(dt);
    otherwise  % 'fd2' : central diff
        D = spdiags([-0.5*e zeros(T,1) 0.5*e],[-1 0 1],T,T)/dt;
end
D = apply_boundary(D,bnd);
end

function A = apply_boundary(A,bnd)
% crude boundary fixes (keep sparse)
[T,~] = size(A);
switch bnd
    case 'rep'
        if T>1
            A(1,1)   = -1; A(1,2)   = 1;
            A(T,T-1) = -1; A(T,T)   = 1;
        end
    case 'mirror'
        if T>2
            A(1,1)= -1; A(1,2)= 1;
            A(T,T-1)= -1; A(T,T)= 1;
        end
    otherwise % 'zero' leave as-is
end
end

function dm = solve_pd(H, g)
% Robust PD solve with multiple fallbacks.
%
% 1) Symmetrise H.
% 2) Try Cholesky.
% 3) If that fails, add increasing jitter.
% 4) Final fallback: eigendecomposition with eigenvalue clamping.

    % Ensure symmetry (protect against tiny asymmetries)
    H = (H + H')/2;

    % First attempt: direct Cholesky
    try
        L = chol(H, 'lower');
        dm = L'\(L\g);
        return;
    catch
        % fall through
    end

    % Second: adaptive jitter on the diagonal
    diagH   = diag(H);
    baseJit = 1e-6 * max(1, max(abs(diagH)));
    jitter  = baseJit;
    I       = eye(size(H));

    for attempt = 1:5
        try
            L = chol(H + jitter*I, 'lower');
            dm = L'\(L\g);
            return;
        catch
            jitter = jitter * 10; % increase jitter and try again
        end
    end

    % Final fallback: eigendecomposition with eigenvalue clamping
    [V, D] = eig(H);
    lam    = diag(D);
    lam    = max(lam, 1e-8);      % clamp negative / tiny eigenvalues
    dm     = V * ((V' * g) ./ lam);
end


function [logL] = quick_elbo_gc(y, f, m, T, E, Wsqrt, sqrtGamma, H_prior, q)
% fast ELBO proxy used only for backtracking comparison
yhat = f(m); yhat = yhat(:);
r0   = y - yhat;
R    = r0;
for kOrd=1:q, R=[R; E{kOrd+1}*r0]; end
sigma2 = (R.^2 + 1e-3) ./ (3 + R.^2/2);  % same form as main
RW  = Wsqrt * R;
RWg = sqrtGamma * RW;
logL_lik   = -0.5 * (sum((R.^2)./max(1e-6,sigma2)) + sum(log(2*pi*max(1e-6,sigma2))));
logL_prior = -0.5 * ((m - m)') * H_prior * (m - m); % zero; structurally here for completeness
logL = logL_lik + logL_prior; % entropy omitted for speed (ok for comparison)
end

function J = computeJacobian(f, x, T)
% central difference Jacobian on model output vector (length T)
epsi = 1e-6;
P    = numel(x);
J    = zeros(T,P);
parfor p = 1:P
    xp = x; xm = x;
    xp(p) = xp(p) + epsi;
    xm(p) = xm(p) - epsi;
    yp = f(xp); ym = f(xm);
    J(:,p) = (yp(:) - ym(:)) / (2*epsi);
end
end

% function K = computeSmoothCovariance(~, lengthScale)
% % simple GP-like prior over parameter indices 
% % note: not a function of m; acts like a smoothness stabiliser
% n = 256; % if you want it to match numel(m), pass m in and use length(m)
% K = exp(-pdist2((1:n)',(1:n)').^2/(2*lengthScale^2));
% K = K + 1e-6*eye(n);
% end

function K = computeSmoothCovariance(n, lengthScale)
% Size-aware GP-like smooth covariance over parameters (P×P).
% Uses parameter index as a proxy coordinate; tweak if you have structure.
idx = (1:n)';                     % parameter index as "location"
D2  = pdist2(idx, idx).^2;        % squared distance on index line
K   = exp(-D2/(2*lengthScale^2));
K   = K + 1e-6*eye(n);            % jitter for PD
end


