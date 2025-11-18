function [m, V, D, logL, iter, sigma2, allm, all_elbo] = ...
    fitVariationalLaplaceThermo_GClam(y, f, m0, S0, maxIter, tol, opts)
% Thermodynamic Variational Laplace in *Generalised Coordinates* with learned per-order weights λ.
%
% This version lifts residuals and Jacobians into generalised coordinates of motion:
%   r^(0:q) = [r, Dr, D^2 r, ...], with D approximated via finite-difference
%   (or supplied by the model). The cost is weighted across orders via Γ(λ),
%   and λ is updated by evidence maximisation (Type-II ML or MAP with Gamma prior).
%
% Inputs:
%   y      : observed vector (length T) OR matrix (T×d stacked columnwise*). We treat it as vec.
%   f      : model handle. Must support:
%            yhat = f(m)
%            [yhat, dY] = f(m)  % OPTIONAL: dY is [T×(q)] or cell{q} of derivatives along sample axis
%   m0, S0 : prior mean and covariance
%   maxIter, tol : as before
%   opts.gc : struct with fields:
%       .order   (default 3)    – highest generalised order q
%       .dt      (default 1)    – sample spacing along the axis to differentiate
%       .lambda  (default [])   – init per-order weights Γ_k; if empty, uses dt-scaled [1, 1/dt^2, 1/dt^4, ...]
%       .op      ('fd2')        – derivative operator: 'fd1' (forward), 'fd2' (central)
%       .boundary('rep')        – boundary handling: 'rep','mirror','zero'
%       .learn_lambda (true)    – learn λ each iter by EM/MAP
%       .lambda_prior  ([])     – struct with .a (shape), .b (rate); scalar or (q+1)-vector
%       .lambda_bounds ([1e-6,1e6]) – clamp λ
%       .lambda_step (1.0)      – damping in log-space for λ update (<=1)
%   opts.varpercthresh : relative eigen threshold for low-rank precision (default 0.01)
%   opts.plots         : 0/1 quick progress plots (default 1)
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

if ~isfield(gc,'order'),        gc.order = 3;        end
if ~isfield(gc,'dt'),           gc.dt    = 1;        end
if ~isfield(gc,'lambda'),       gc.lambda = [];      end
if ~isfield(gc,'op'),           gc.op    = 'fd2';    end
if ~isfield(gc,'boundary'),     gc.boundary = 'rep'; end
if ~isfield(gc,'learn_lambda'), gc.learn_lambda = true; end
if ~isfield(gc,'lambda_prior'), gc.lambda_prior = [];  end
if ~isfield(gc,'lambda_bounds'),gc.lambda_bounds = [1e-6, 1e6]; end
if ~isfield(gc,'lambda_step'),  gc.lambda_step   = 1.0; end

q   = max(0, round(gc.order));
dt  = gc.dt;

% per-order weights Γ_k: [1, 1/dt^2, 1/dt^4, ...] as initialisation
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
T_k = T*ones(q+1,1);        % each order has T samples

% Build GC derivative operators (E{1}=I, E{2}=D, ..., E{q+1}=D^q)
E = build_gc_operators(T, q, dt, gc.op, gc.boundary);

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
K = computeSmoothCovariance(P, 2);          % K is P×P
A = (S0 + K);                               % keep symmetric
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
            yhat_k = dY_model(:,kOrd);     % model predicted derivative order k
            y_k    = E{kOrd+1} * y;        % numeric derivative of data
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

    % ----- Per-order sufficient statistics for λ updates -----
    S_k = zeros(q+1,1);
    off = 0;
    for kOrd = 0:q
        idx = (off+1):(off+T_k(kOrd+1));
        rk  = R(idx);
        sk  = sigma2(idx);
        S_k(kOrd+1) = sum((rk.^2) ./ max(1e-12, sk));
        off = off + T_k(kOrd+1);
    end

    % ----- Update λ (Type-II ML or MAP with Gamma prior) -----
    if gc.learn_lambda
        if ~isempty(gc.lambda_prior)
            a = gc.lambda_prior.a; b = gc.lambda_prior.b;
            if isscalar(a), a = a*ones(q+1,1); end
            if isscalar(b), b = b*ones(q+1,1); end
            lam_new = ((a(:)-1) + 0.5*T_k(:)) ./ (b(:) + 0.5*S_k(:));
        else
            lam_new = T_k(:) ./ max(1e-12, S_k(:));  % EB
        end
        % Damped update in log-space
        if gc.lambda_step < 1
            lam = exp( (1-gc.lambda_step)*log(lam) + gc.lambda_step*log(max(lam_new,1e-12)) );
        else
            lam = lam_new;
        end
        lam = min(max(lam, gc.lambda_bounds(1)), gc.lambda_bounds(2));
    end

    % ----- Build weighting matrices with updated λ -----
    sqrtGamma = spdiags(sqrt(repmat(lam,T,1)), 0, (q+1)*T, (q+1)*T);

    % Weighted residuals & log-likelihood (now λ-aware)
    Wsqrt  = spdiags(1./sqrt(max(epsilon,sigma2)), 0, n, n);
    RW     = Wsqrt * R;
    RWg    = sqrtGamma * RW;

    ll_orders = 0.5*sum(T_k .* log(max(1e-12,lam))) - 0.5*sum(lam .* S_k);
    const_sig = -0.5 * sum(log(2*pi*max(epsilon, sigma2)));
    logL_lik  = ll_orders + const_sig;

    % Prior over λ if provided: sum_k [(a-1)logλ - bλ] (const omitted)
    logL_lam_prior = 0;
    if ~isempty(gc.lambda_prior)
        a = gc.lambda_prior.a; b = gc.lambda_prior.b;
        if isscalar(a), a = a*ones(q+1,1); end
        if isscalar(b), b = b*ones(q+1,1); end
        logL_lam_prior = sum((a(:)-1).*log(max(1e-12,lam)) - b(:).*lam);
    end

    % Jacobian in GC: J0 = dyhat/dm (T×P). Then stack [J0; E2*J0; ...; E{q+1}*J0]
    J0 = computeJacobian(@(mm) f(mm), m, T);
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
    JW  = Wsqrt * J_gc;
    JWG = sqrtGamma * JW;

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
    try
        Lchol = chol(H_elbo,'lower');
        logdetH = 2*sum(log(diag(Lchol)));
    catch
        Lchol = chol(H_elbo + 1e-6*eye(size(H_elbo)),'lower');
        logdetH = 2*sum(log(diag(Lchol)));
    end
    logL_entropy = -0.5 * logdetH;
    logL = logL_lik + logL_prior + logL_entropy + logL_lam_prior;
    all_elbo = [all_elbo, logL]; %#ok<AGROW>

    % Backtracking if ELBO drops
    if numel(all_elbo) > 1 && logL < all_elbo(end-1)
        step = 0.5; improved = false; tries = 0;
        while ~improved && tries < 8
            tries = tries+1;
            m_try = m_prev + (step^tries)*dm;
            % quick ELBO proxy with current λ
            [logL_try] = quick_elbo_gc(y, f, m_try, T, E, lam, q);
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

        % λ display
        sgtitle(sprintf('it %d | ELBO %.4f | \\lambda: %s', ...
            iter_i, logL, strjoin(compose('%.3g', lam.'), ' ')));

        drawnow;
    end

    % Convergence
    if norm(dm) < tol
        fprintf('Converged @ %d\n', iter_i);
        break;
    end

    fprintf('it %3d | ELBO %.4f | ||dm||=%.3e | lam: ', iter_i, logL, norm(dm));
    fprintf('%6.3g ', lam); fprintf('\n');
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
% robust PD solve with fallback
try
    L = chol(H,'lower');
    dm = L'\(L\g);
catch
    [dm,flag] = pcg(H + 1e-6*eye(size(H)), g, 1e-8, 200);
    if flag~=0, dm = (H + 1e-6*eye(size(H)))\g; end
end
end

function [logL] = quick_elbo_gc(y, f, m, T, E, lam, q)
% Fast ELBO proxy (for line-search/backtracking). Omits entropy and m-prior.
yhat = f(m); yhat = yhat(:);
r0   = y - yhat;
R    = r0;
for kOrd=1:q, R=[R; E{kOrd+1}*r0]; end

sigma2 = (R.^2 + 1e-3) ./ (3 + R.^2/2);
n = (q+1)*T;

% Per-order stats
S_k = zeros(q+1,1); off=0;
for kOrd = 0:q
    idx = (off+1):(off+T);
    rk  = R(idx);
    sk  = sigma2(idx);
    S_k(kOrd+1) = sum((rk.^2) ./ max(1e-12, sk));
    off = off + T;
end

T_k = T*ones(q+1,1);
ll_orders = 0.5*sum(T_k .* log(max(1e-12,lam))) - 0.5*sum(lam .* S_k);
const_sig = -0.5 * sum(log(2*pi*max(1e-6, sigma2)));

logL = ll_orders + const_sig;
end

function K = computeSmoothCovariance(n, lengthScale)
% Size-aware GP-like smooth covariance over parameters (P×P).
idx = (1:n)';                     
D2  = pdist2(idx, idx).^2;        
K   = exp(-D2/(2*lengthScale^2));
K   = K + 1e-6*eye(n);            
end
