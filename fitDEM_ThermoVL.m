function [POST, TRACE] = fitDEM_ThermoVL(y, f, g, x0, theta0, opts)
% fitDEM_ThermoVL  —  DEM-style D/E/M using thermoVL
%
% DEM over nonlinear state-space:
%   x' = f(x, v, theta) + w
%   y  = g(x, v, theta) + z
%
% Generalised coordinates (order p):
%   \tilde{x} = [x, x', x'', ...], \tilde{y} likewise (estimated from data)
%   Motion error:   eps_x = D*\tilde{x} - \tilde{f}(\tilde{x},theta)
%   Output error:   eps_y = \tilde{y} - \tilde{g}(\tilde{x},theta)
%
% D-step: update \tilde{x}_t (mode path) by Gauss–Newton on free-action
% E-step: update theta using accumulated curvature over t
% M-step: update precisions (lambda_y, lambda_x) with Gamma-posteriors
%
% Inputs
%   y        [T x ny]   observations over time
%   f        @(x,theta) -> nx x 1   (state flow; v handled inside if needed)
%   g        @(x,theta) -> ny x 1   (observation map)
%   x0       [nx x 1]   initial state mean
%   theta0   [np x 1]   initial parameter mean
%   opts     struct with fields (all optional, sensible defaults provided):
%       .dt              sample period (default 1)
%       .p               generalised order (default 2 => 0,1,2)
%       .Dx_iters        D-step inner iters per time (default 2)
%       .outer_iters     DEM cycles (D over all t, then E, then M) (default 8)
%       .gamma_y         temporal weights for output GC (default 1.0)
%       .gamma_x         temporal weights for state GC  (default 1.0)
%       .lambda_y0       obs precision init (default 1/var(y))
%       .lambda_x0       state motion precision init (default 1)
%       .pE_theta        prior mean for theta (default = theta0)
%       .pC_theta        prior cov for theta (default = eye*1e1)
%       .pE_x0           prior mean for initial GC state (default zeros)
%       .pC_x0           prior cov for initial GC state (default eye*1e1)
%       .varpercthresh   low-rank threshold for parameter Hessian (default .01)
%       .max_rank_theta  cap for low-rank (default = numel(theta0))
%       .plot            0/1 quick progress plots (default 0)
%
% Outputs
%   POST struct:
%       .x_path      [T x nx]  posterior mode of state path (0th order)
%       .xgc_path    [T x (p+1)*nx] posterior mode in generalised coords
%       .theta_mean  [np x 1]
%       .theta_V, .theta_D   low-rank + diag precision factors for theta Hessian
%       .lambda_y, .lambda_x  precisions (obs & motion)
%   TRACE struct:
%       .Fa          free-action trace per outer iter
%       .sumsq_y     sum of output error squares per iter
%       .sumsq_x     sum of motion error squares per iter
%       .theta_hist  parameter means over iterations
%       .lambda_hist precisions over iterations
%
% AS2025

% -------------------- defaults --------------------
if nargin < 6, opts = struct; end
dt      = getd(opts,'dt',1.0);
p       = getd(opts,'p',2);
py   = getd(opts,'py',0);

Dx_it   = getd(opts,'Dx_iters',2);
outerIt = getd(opts,'outer_iters',8);
gamY    = getd(opts,'gamma_y',1.0);
gamX    = getd(opts,'gamma_x',1.0);
[Tx,ny] = size(y);

nx = numel(x0);
theta = theta0(:);
np = numel(theta);

pE_theta = getd(opts,'pE_theta',theta0(:));
pC_theta = getd(opts,'pC_theta',eye(np)*1e1);

% prior on initial generalised state
xgc0_mu = getd(opts,'pE_x0',zeros((p+1)*nx,1));
xgc0_C  = getd(opts,'pC_x0',eye((p+1)*nx)*1e1);

% precisions
%ly = getd(opts,'lambda_y0', 1/max(var(y,0,1),1e-6) ); % scalar or 1xny (we use scalar)
v_all = var(y(:));                          % scalar variance over all entries
ly    = getd(opts,'lambda_y0', 1/max(v_all, 1e-6));

if numel(ly) > 1, ly = mean(ly); end
lx = getd(opts,'lambda_x0', 1.0);

% low-rank threshold for param Hessian
varthr = getd(opts,'varpercthresh',0.01);
kmax   = getd(opts,'max_rank_theta',np);

doplot = getd(opts,'plot',0);

% temporal weights for each GC order: w_k = (gamma^k)/(dt^k)
wy = (gamY.^((0:py)))./(dt.^(0:py)); wy = wy / max(wy);
wx = (gamX.^((0:p )))./(dt.^(0:p )); wx = wx / max(wx);

% derivative operator D on generalised coordinates (truncated shift)
Dop = kron( eye(p+1), zeros(nx) );
for k=1:p
    block = eye(nx);
    row   = (k-1)*nx + (1:nx);
    col   = k*nx     + (1:nx);
    Dop(row, col) = block;
end

% Derivative operator for outputs (same truncated shift, but ny-sized)
DopY = kron(eye(p+1), zeros(ny));
for k = 1:p
    row = (k-1)*ny + (1:ny);
    col =  k   *ny + (1:ny);
    DopY(row,col) = eye(ny);
end

% last rows stay zero; Dop: [(x' <- xdot), (x'' <- xddot), ...]
% size: (p+1)nx x (p+1)nx

% precompute generalised observations from data
Ygc = generalise_observation(y, py, dt); % [T x (p+1)*ny]

% storage
xgc_path = zeros(Tx,(p+1)*nx);
x_path   = zeros(Tx,nx);

theta_hist  = zeros(np,outerIt);
lambda_hist = zeros(2,outerIt);
Fa_trace    = zeros(1,outerIt);
sumsq_y     = zeros(1,outerIt);
sumsq_x     = zeros(1,outerIt);

% initial GC state (just tile dynamics from x0)
xgc = zeros((p+1)*nx,1);
xgc(1:nx) = x0(:);
% Fill higher orders by chaining f and its Jacobian (rough init)
[xgc] = seed_generalised_state(xgc, theta, f, p, nx);

% -------------------- DEM outer loop --------------------
for it = 1:outerIt
    % --- D-step: update x(t) path, given theta, precisions ---
    ss_y = 0; ss_x = 0;  % accumulate squares for M-step
    for t = 1:Tx
        ygc_t = Ygc(t,:).';      % (p+1)*ny x 1
        % carry previous xgc as init (causal filter)
        if t > 1
            xgc = xgc_path(t-1,:).';
        end

        % inner recognition updates for this time point
        for k = 1:Dx_it

            % --- D-step ------------------------

            % residual vector r = [sqrt(ly)*W_y*eps_y ; sqrt(lx)*W_x*eps_x]
            [r, epsy, epsx] = residual_gc(xgc, ygc_t, theta, f, g, Dop, wy, wx, nx, ny, p, py, ly, lx);
            Jx = jacobian_fd(@(xz) residual_gc(xz, ygc_t, theta, f, g, Dop, wy, wx, nx, ny, p, py, ly, lx), xgc);


            % optional weak prior tying to initial GC (only at t=1)
            if t == 1
                Hprior_x = (xgc0_C \ eye(size(xgc0_C)));
                gprior_x = -Hprior_x*(xgc - xgc0_mu);
            else
                Hprior_x = 0;
                gprior_x = 0;
            end

            Hx = (Jx.'*Jx) + blkdiag(Hprior_x);  % Gauss–Newton Hessian
            gx = (Jx.'*r)  + gprior_x;           % gradient

            ridge = 1e-3 * trace(Hx)/max(1,numel(Hx));
            Hx = Hx + ridge*eye(size(Hx));

            % damped solve
            dm = solve_posdef(Hx, gx);

            % ---- Armijo-style line search ----
            alpha = 1.0;
            xgc_try = xgc + alpha*dm;
            [r_try, ~, ~] = residual_gc(xgc_try, ygc_t, theta, f, g, Dop, wy, wx, nx, ny, p,py, ly, lx);

            if norm(r_try) > norm(r)
                alpha = 0.5;
                xgc_try = xgc + alpha*dm;
                [r_try, ~, ~] = residual_gc(xgc_try, ygc_t, theta, f, g, Dop, wy, wx, nx, ny, p,py, ly, lx);
                if norm(r_try) > norm(r)
                    alpha = 0.25;
                    xgc_try = xgc + alpha*dm;
                end
            end

            xgc = xgc_try;

            % trust region
            %alpha = min(1, 1/(1+norm(dm)));
            %xgc = xgc + alpha*dm;
        end

        xgc_path(t,:) = xgc.';
        x_path(t,:)   = xgc(1:nx).';

        % Accumulate squared errors for M-step
        ss_y = ss_y + sum(epsy.^2);
        ss_x = ss_x + sum(epsx.^2);
    end

    % --- E-step: update theta (one GN step with low-rank precision) ---
    % accumulate over time: r_t(theta), J_theta_t
    gth = zeros(np,1);
    Hth = zeros(np,np);
    for t = 1:Tx
        xgc_t = xgc_path(t,:).';
        ygc_t = Ygc(t,:).';
        % residual function of theta
        %rfun = @(th) residual_gc(xgc_t, ygc_t, th, f, g, Dop, wy, wx);
        rfun = @(th) residual_gc(xgc_t, ygc_t, th, f, g, Dop, wy, wx, nx, ny, p, py, ly, lx);
        rt   = rfun(theta);
        Jt   = jacobian_fd(rfun, theta);

        Hth  = Hth + (Jt.'*Jt);
        gth  = gth + (Jt.'*rt);
    end
    % add parameter prior
    Hth_prior = (pC_theta \ eye(size(pC_theta)));
    gth_prior = -Hth_prior*(theta - pE_theta);

    Hth_elbo = Hth + Hth_prior;
    gth_elbo = gth + gth_prior;

    ridge_th = 1e-3 * trace(Hth_elbo)/max(1,numel(Hth_elbo));
    Hth_elbo = Hth_elbo + ridge_th*eye(size(Hth_elbo));

    % low-rank+diag precision factorisation for theta Hessian
    [U,Sv,~] = svd(Hth_elbo, 'econ');
    lam = diag(Sv);
    k   = max(1, sum(lam >= varthr*lam(1)));
    k   = min(k,kmax);
    Vth = U(:,1:k)*sqrt(Sv(1:k,1:k));
    Dth = diag(diag(Hth_elbo) - sum(Vth.^2,2));
    % robust solve
    %dth = solve_posdef(Hth_elbo, gth_elbo);
    %theta = theta + dth;
    dth = solve_posdef(Hth_elbo, gth_elbo);
    alpha_theta = 1e-2;   % damp parameter step
    theta = theta + alpha_theta * dth;


    % --- M-step: update precisions (Gamma posterior updates) ---
    % Use ML/Gamma-regularised updates; weak prior (nu,beta)
    nu_y = 1e-2;  beta_y = 1e-2;
    nu_x = 1e-2;  beta_x = 1e-2;

    % Recompute sum of squares using updated theta (optional small pass)
    ss_y = 0; ss_x = 0;
    for t = 1:Tx
        xgc_t = xgc_path(t,:).';  ygc_t = Ygc(t,:).';
        [~, epsy, epsx] = residual_gc(xgc_t, ygc_t, theta, f, g, Dop, wy, wx, nx, ny, p, py, ly, lx);
        % use ONLY 0th-order obs residual for ly (first ny entries)
        ss_y = ss_y + sum(epsy(1:ny).^2);
        ss_x = ss_x + sum(epsx.^2);
    end

    dim_y = Tx * (p+1)*ny;
    dim_x = Tx * (p+1)*nx;

    %ly = (nu_y + dim_y) / (beta_y + ss_y + 1e-12);
    %lx = (nu_x + dim_x) / (beta_x + ss_x + 1e-12);

    ly_min = getd(opts,'ly_min',1e-6);
    nu_y   = 1e-2;  beta_y = 1e-2;
    dim_y0 = Tx * ny;

    ly = (nu_y + dim_y0) / (beta_y + ss_y + 1e-12);
    ly = max(ly, ly_min);

    % (lx stays as before; its "dim" is Tx*(p+1)*nx because all x-orders are used)
    dim_x = Tx * (p+1)*nx;
    lx = (nu_x + dim_x) / (beta_x + ss_x + 1e-12);

    ly = max(1e-6, min(1e6, (nu_y + dim_y0) / (beta_y + ss_y + 1e-12)));
    lx = max(1e-6, min(1e6, (nu_x + dim_x) / (beta_x + ss_x + 1e-12)));

    % --- bookkeeping & free-action (approximate) ---
    % Fa ~ -0.5*( ly*ss_y + lx*ss_x ) + log det terms/prior terms (omitted consts)
    Fa = -0.5*( ly*ss_y + lx*ss_x ) ...
         -0.5*((theta - pE_theta).'*Hth_prior*(theta - pE_theta)) ...
         -0.5*((xgc_path(1,:).' - xgc0_mu).'*((xgc0_C \ eye(size(xgc0_C)))*(xgc_path(1,:).' - xgc0_mu)));

    theta_hist(:,it)  = theta;
    lambda_hist(:,it) = [ly; lx];
    Fa_trace(it)      = Fa;
    sumsq_y(it)       = ss_y;
    sumsq_x(it)       = ss_x;

    if doplot
        fprintf('Iter %d | Fa %.3f | ly %.3g | lx %.3g | ||dtheta|| %.3g\n', ...
            it, Fa, ly, lx, norm(dth));
    end

    if doplot
        % Set up figure on first iteration
        if it == 1
            fh = figure('Name','DEM-ThermoVL progress','Color','w');
            tiledlayout(fh,2,2,'TileSpacing','compact','Padding','compact');
        else
            figure(fh);
        end

        % --- (a) observed vs predicted ---
        nexttile(1); cla;
        tspan = (0:Tx-1)*dt;
        % predicted observation from current x_path
        yhat = zeros(Tx,ny);
        for tt = 1:Tx
            yhat(tt,:) = g(xgc_path(tt,1:nx).',theta);
        end
        plot(tspan,y,'k','DisplayName','Observed'); hold on;
        plot(tspan,yhat,'r','DisplayName','Predicted'); hold off;
        title('Fit to data'); xlabel('time'); ylabel('y');
        legend show;

        % --- (b) Free-action trace ---
        nexttile(2); cla;
        plot(1:it,Fa_trace(1:it),'-o');
        xlabel('Outer iter'); ylabel('Free-action');
        title('Free-action');

        % --- (c) Precisions ---
        nexttile(3); cla;
        plot(1:it,lambda_hist(1,1:it),'-o','DisplayName','\lambda_y'); hold on;
        plot(1:it,lambda_hist(2,1:it),'-s','DisplayName','\lambda_x'); hold off;
        legend show; xlabel('Outer iter'); ylabel('Precision');
        title('Precisions');

        % --- (d) Parameters ---
        nexttile(4); cla;
        plot(1:it,theta_hist(:,1:it)','-o');
        xlabel('Outer iter'); ylabel('value');
        title('Parameters');
        drawnow;
    end



end

% -------------------- outputs --------------------
POST.x_path      = x_path;
POST.xgc_path    = xgc_path;
POST.theta_mean  = theta;
POST.theta_V     = Vth;
POST.theta_D     = Dth;
POST.lambda_y    = ly;
POST.lambda_x    = lx;

TRACE.Fa         = Fa_trace;
TRACE.sumsq_y    = sumsq_y;
TRACE.sumsq_x    = sumsq_x;
TRACE.theta_hist = theta_hist;
TRACE.lambda_hist= lambda_hist;

end % main

% ==================== helpers ====================

function [r, epsy, epsx] = residual_gc(xgc, ygc, theta, f, g, DopX, wy, wx, nx, ny, p, py, ly, lx)
    [~,  epsx] = eps_x_gc(xgc, theta, f, DopX, wx, nx, p);
    [~,  epsy] = eps_y_gc(xgc, ygc, theta, g, wy, nx, ny, p, py);
    r = [sqrt(ly)*epsy; sqrt(lx)*epsx];
end

function [ygc_hat, epsy] = eps_y_gc(xgc, ygc, theta, g, wy, nx, ny, p, py)
    % Build ĝ̃ up to order py and form eps_y = W * ( ỹ - ĝ̃ )
    X   = reshape(xgc, nx, p+1);   % now we know p
    x0  = X(:,1);
    Yhat = zeros(ny, py+1);

    % 0th order
    Yhat(:,1) = g(x0,theta);

    if py >= 1
        gx0 = jacobian_fd(@(x) g(x,theta), x0);
        Yhat(:,2) = gx0 * X(:,2);
    end
    if py >= 2
        gx0 = jacobian_fd(@(x) g(x,theta), x0);
        dir   = X(:,2); nd = max(1e-6, 1e-6*norm(dir));
        gx_f  = jacobian_fd(@(x) g(x,theta), x0 + nd*dir);
        dgx_d = (gx_f - gx0)/nd;
        Yhat(:,3) = dgx_d*X(:,2) + gx0*X(:,3);
    end

    ygc_hat = reshape(Yhat, (py+1)*ny, 1);
    W  = kron(diag(wy), eye(ny));
    epsy = W * (ygc - ygc_hat);
end

function [xflow_hat, epsx] = eps_x_gc(xgc, theta, f, DopX, wx, nx, p)
    % eps_x = W * ( D*x̃ – f̃ )
    X   = reshape(xgc, nx, p+1);
    x0  = X(:,1);
    fx0 = jacobian_fd(@(x) f(x,theta), x0);
    f0  = f(x0,theta);

    Fgc = zeros(nx, p+1);
    Fgc(:,1) = f0;
    if p >= 1
        Fgc(:,2) = fx0 * X(:,2);
    end
    if p >= 2
        dir   = X(:,2); nd = max(1e-6, 1e-6*norm(dir));
        fx_f  = jacobian_fd(@(x) f(x,theta), x0 + nd*dir);
        dfx_d = (fx_f - fx0)/nd;
        Fgc(:,3) = dfx_d*X(:,2) + fx0*X(:,3);
    end

    Dxgc    = DopX * xgc;
    fgc_vec = Fgc(:);

    W = kron(diag(wx), eye(nx));
    epsx = W * (Dxgc - fgc_vec);
    xflow_hat = fgc_vec;
end

function pp = p_from_len(L,nx)
    % infer p from length of xgc and nx: L = (p+1)*nx
    pp = (L / nx) - 1;
end


function J = jacobian_fd(fun, x)
    % central finite-difference Jacobian
    y0 = fun(x);
    m  = numel(y0);
    n  = numel(x);
    J  = zeros(m,n);
    eps0 = 1e-6;
    for i = 1:n
        dx = zeros(n,1);  dx(i) = eps0*max(1,abs(x(i)));
        fp = fun(x + dx);
        fm = fun(x - dx);
        J(:,i) = (fp - fm) / (2*dx(i));
    end
end

function [sol, info] = solve_posdef(H, g)
% Levenberg–Marquardt + diag-jitter + SVD fallback
    info = struct('lam',[],'method','chol');
    lam = 1e-3 * max(1, norm(H, 'fro')/max(1,numel(H)));  % scale-aware
    D   = diag(max(1e-12, diag(H)));                     % keep structure

    for k = 1:8
        try
            L = chol(H + lam*D + 1e-10*eye(size(H)), 'lower');
            sol = L'\(L\g);
            info.lam = lam;
            return
        catch
            lam = lam * 10;         % increase damping
        end
    end
    % Fallback: truncated SVD (floor small singular values)
    [U,S,V] = svd(H, 'econ');
    s = diag(S);
    s_floor = 1e-8 * max(s);
    s(s < s_floor) = s_floor;
    sol = V * ((U' * g) ./ s);
    info.method = 'svd';
    info.lam = lam;
end

function nx = infer_nx_from_xgc(xgc)
    % Guess nx from the assumption xgc is stacked (p+1)*nx
    % We'll infer (p+1) by assuming it's at least 1 and nx<=length(xgc)
    L = numel(xgc);
    % choose smallest nx such that (L/nx) is integer and >=1
    nx = L; 
    for cand = 1:L
        if mod(L,cand)==0
            nx = cand; break;
        end
    end
end

function Xgc = generalise_observation(y, p, dt)
    % Stack y, Dy, D2y, ... via simple finite differences (causal-ish)
    [T,ny] = size(y);
    Ygc = zeros(T,(p+1)*ny);
    % 0th
    Ygc(:,1:ny) = y;
    if p >= 1
        Dy = [diff(y)/dt; zeros(1,ny)];
        Ygc(:,ny+(1:ny)) = Dy;
    end
    if p >= 2
        D2y = [diff(Dy)/dt; zeros(1,ny)];
        Ygc(:,2*ny+(1:ny)) = D2y;
    end
    % (can add Savitzky-Golay smoothing here if desired)
    Xgc = Ygc;
end

function xgc = seed_generalised_state(xgc, theta, f, p, nx)
    X = reshape(xgc, nx, p+1);
    x0 = X(:,1);
    if p >= 1
        X(:,2) = f(x0,theta);
    end
    if p >= 2
        fx0 = jacobian_fd(@(x) f(x,theta), x0);
        X(:,3) = fx0*X(:,2);
    end
    xgc = X(:);
end
function val = getd(s,field,default)
    if isfield(s,field) && ~isempty(s.(field)), val = s.(field); else, val = default; end
end
