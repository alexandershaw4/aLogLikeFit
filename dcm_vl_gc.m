function DCM = dcm_vl_gc(DCM, opts)
% Variational Laplace (Gauss–Newton) in "generalised coordinates" for spectral DCM (parameters-only).
% Now with ONLINE PLOTS.
%
% Options (all optional):
%   opts.gc_order    : integer >=0 (default 1)
%   opts.mask_fun    : @(f) f>0.25
%   opts.logpsd      : true (fit log-PSD)
%   opts.Wfreq       : vector length F (default ones)
%   opts.beta_sched  : [0.1 0.3 0.6 1]
%   opts.maxiter     : 128
%   opts.step        : 1e-1
%   opts.verbose     : true
%   --- plotting ---
%   opts.plot        : true/false (default true)
%   opts.plot_every  : integer (default 5)   % iterations per refresh
%   opts.show_params : true/false (default true)  % parameter trajectories

% --- defaults
if nargin<2, opts = struct; end
dflt = struct('gc_order',1,'mask_fun',@(f) f>0.25,'logpsd',false,...
              'Wfreq',[],'beta_sched',[0.1 0.3 0.6 1],...
              'maxiter',128,'step',1e-1,'verbose',true,...
              'plot',true,'plot_every',5,'show_params',true);
fn = fieldnames(dflt);
for i=1:numel(fn)
    if ~isfield(opts,fn{i})||isempty(opts.(fn{i})), opts.(fn{i})=dflt.(fn{i}); end
end

% --- data
Hz_full = DCM.xY.Hz(:);
Sy_full = DCM.xY.y{1};
if any(Sy_full(:)<=0), Sy_full = max(Sy_full, eps); end

mask = opts.mask_fun(Hz_full);
Hz   = Hz_full(mask);
Sy   = Sy_full(mask,:);                % [F x C]
F    = numel(Hz);
%if isempty(opts.Wfreq), opts.Wfreq = ones(F,1); end
%Wf   = opts.Wfreq(:);

if isempty(opts.Wfreq)
    % per-frequency variance normalisation (robust scaling)
    base = opts.logpsd * log(Sy) + (~opts.logpsd) * Sy;
    v = var(base, 0, 2);                   % variance across channels
    v(v<=eps) = 1;
    Wf = 1 ./ sqrt(v);                     % whiten-ish
else
    Wf = opts.Wfreq(:);
end


% --- priors / active set
pE = DCM.M.pE;  pC = DCM.M.pC;
Evec = spm_vec(pE);
Cvec = spm_vec(pC);
active = find(isfinite(Cvec) & (Cvec > 1e-12));
na = numel(active);

mu = Evec(active);
Pi_theta = diag(1 ./ max(Cvec(active), 1e-12));   % diagonal precision (typical for DCM)

% --- generalised multipliers Φ_k(f) = (i*2πf)^k
K   = opts.gc_order;
iw  = 1i*2*pi*Hz(:);
Phi = cell(K+1,1);
for k=0:K, Phi{k+1} = iw.^k; end

use_log = opts.logpsd;

% --- model wrapper using DCM.M.IS
Mloc = DCM.M;
predict = @(theta_vec) local_S(theta_vec, DCM, Mloc, Hz, size(Sy));

% --- pack observations (GC stack)
[y_gc, ~] = pack_gc_obs(Sy, Phi, Wf, use_log);

% --- Jacobian via FD (parfor-friendly)
%h    = 1e-4;
h = 1e-4 * max(1, abs(mu));
Jfun = @(th) fd_J_parfor(@(t) pack_gc_mod( predict(embed(Evec,active,t)) , Phi, Wf, use_log), th, h);

% --- initial θ (active)
theta  = mu;
alpha  = opts.step;                 % simple trust-region scalar

lambda = 1e-1; lambda_up = 5; lambda_down = 0.3;
max_step_norm = 3.0;   % cap in prior metric units

Ftrace = [];
theta_hist = [];

% --- prepare plotting
if opts.plot
    ph = init_plot_fig();
end

% --- annealed VL (Gauss–Newton on GC-augmented objective)
iter_global = 0;
for beta = opts.beta_sched(:)'
    if opts.verbose, fprintf('β=%.2f\n',beta); end
    for it = 1:opts.maxiter
        iter_global = iter_global + 1;

        % forward, residuals
        Smask   = predict(embed(Evec,active,theta));    % [F x C]
        yhat_gc = pack_gc_mod(Smask, Phi, Wf, use_log);
        e       = y_gc - yhat_gc;

        % --- GN pieces at current theta
        J = Jfun(theta);
        G = beta*(J.'*J) + Pi_theta;                % GN metric for F (H_F ≈ -G)
        g = -beta*(J.'*e) + Pi_theta*(theta - mu);  % grad F

        % Levenberg–Marquardt damping
        Gd = G + lambda * eye(size(G));
        dtheta = -(Gd \ g);                         % proposed step (maximising F)

        % Optional: cap step in prior metric
        sn = sqrt(dtheta.' * (Pi_theta * dtheta));
        if sn > max_step_norm
            dtheta = dtheta * (max_step_norm / max(sn, eps));
        end

        % --- Evaluate at proposed theta_new
        theta_new   = theta + dtheta;
        Smask_new   = predict(embed(Evec,active,theta_new));
        yhat_gc_new = pack_gc_mod(Smask_new, Phi, Wf, use_log);
        e_new       = y_gc - yhat_gc_new;

        % Free energy (up to consts)
        Fold = free_energy_quad(e,     beta) - 0.5*(theta     -mu).'*(Pi_theta*(theta     -mu));
        Fnew = free_energy_quad(e_new, beta) - 0.5*(theta_new -mu).'*(Pi_theta*(theta_new -mu));

        % --- Trust-region ratio ρ = actual / predicted increase
        % For F, Hessian ≈ -G  => quadratic model: ΔF_pred = g'Δ - 0.5 Δ' G Δ
        dF_pred = (g.'*dtheta) - 0.5*(dtheta.' * (G * dtheta));
        dF_act  = (Fnew - Fold);
        rho     = dF_act / max(dF_pred, eps);

        % --- Accept / reject & update lambda
        if rho > 0      % some actual increase
            theta  = theta_new;
            Smask  = Smask_new;
            e      = e_new;
            Ftrace(end+1,1) = Fnew;
            theta_hist(:,end+1) = theta;
        end

        if     rho > 0.75, lambda = max(lambda * lambda_down, 1e-9);
        elseif rho < 0.25, lambda = min(lambda * lambda_up,   1e9);
        end

        % (optional) early stopping
        if norm(g) < 1e-4 || norm(dtheta) < 1e-6, break; end



        if opts.verbose && mod(it,16)==0
            fprintf('  it %3d | ||g||=%.3e | ||dθ||=%.3e | α=%.2g | F≈%.3f\n', ...
                    it, norm(g), norm(dtheta), alpha, Fnew);
        end

        % --- online plots
        if opts.plot && mod(iter_global, opts.plot_every)==0
            do_plot(ph, Hz, Sy, Smask, Ftrace, theta_hist, use_log, beta, iter_global, opts.show_params);
        end

        if norm(dtheta) < 1e-6, break; end
    end
end

% --- final outputs
Jfinal = Jfun(theta);
Gfinal = (Jfinal.'*Jfinal) + (Pi_theta);   % ≈β=1
Cp = pinv(Gfinal);
Ep = spm_unvec(embed(Evec,active,theta), pE);

Sfit = predict(embed(Evec,active,theta));

DCM.Ep     = Ep;
DCM.Cp     = Cp;
DCM.F      = Ftrace;
DCM.fit.S  = Sfit;
DCM.fit.Hz = Hz;

% one last plot with final fit
if opts.plot
    do_plot(ph, Hz, Sy, Sfit, Ftrace, theta_hist, use_log, 1, iter_global, opts.show_params);
end

% ------------------ helpers ------------------
function S = local_S(theta_full_vec, DCM_, M_, Hz_, data_shape)
    P   = spm_unvec(theta_full_vec, DCM_.M.pE);
    M2  = M_;
    M2.Hz = Hz_(:).';
    if ~isfield(M2,'s_jitter'), M2.s_jitter = 2*pi*1e-3; end
    if ~isfield(M2,'ridge'),    M2.ridge    = 1e-6;      end
    yhat = feval(M2.IS, P, M2, []);
    if isstruct(yhat) && isfield(yhat,'y') && ~isempty(yhat.y)
        S = yhat.y{1};
    elseif iscell(yhat)
        S = yhat{1};
    else
        S = yhat;
    end
    S = real(S);
    S = max(S, eps);
    S = reshape(S, data_shape);
end

function vec = embed(E0, active_ix, theta_a)
    vec = E0; vec(active_ix) = theta_a;
end

    function [ygc, pk] = pack_gc_obs(Sy_, Phi_, Wf_, use_log_)
    F_  = size(Sy_,1);
    base  = use_log_ * log(Sy_) + (~use_log_) * real(Sy_);
    y0    = base(:);
    W0    = repmat(Wf_, numel(Sy_(:))/F_, 1);
    ygc   = W0 .* y0;
    if numel(Phi_)>1
        for kk = 2:numel(Phi_)
            phi = Phi_{kk};                           % F x 1
            yk  = base .* real(repmat(phi,1,size(Sy_,2)));
            ygc = [ygc; W0 .* yk(:)]; %#ok<AGROW>
        end
    end
    pk = struct();
end

% --- replace inside pack_gc_mod ---
function ygc = pack_gc_mod(Sm_, Phi_, Wf_, use_log_)
    F_  = size(Sm_,1);
    base  = use_log_ * log(Sm_) + (~use_log_) * real(Sm_);
    y0    = base(:);
    W0    = repmat(Wf_, numel(Sm_(:))/F_, 1);
    ygc   = W0 .* y0;
    if numel(Phi_)>1
        for kk = 2:numel(Phi_)
            phi = Phi_{kk};
            yk  = base .* real(repmat(phi,1,size(Sm_,2)));
            ygc = [ygc; W0 .* yk(:)];
        end
    end
end
function Fq = free_energy_quad(e, beta)
    Fq = -0.5 * beta * (e.'*e);
end

function J = fd_J_parfor(fun, th, h_)
    th = th(:);
    n  = numel(th);
    f0 = fun(th);
    m  = numel(f0);
    J  = zeros(m,n);
    if isscalar(h_), h_ = repmat(h_, n, 1); else, h_ = h_(:); end
    parfor j = 1:n
        t1 = th; t2 = th; hj = h_(j);
        t1(j) = t1(j) + hj;  t2(j) = t2(j) - hj;
        J(:,j) = (fun(t1) - fun(t2)) / (2*hj);
    end
end

function ph = init_plot_fig()
    ph.fig = figure('Name','GC-VL Online','Color','w','NumberTitle','off');
    tlo = tiledlayout(ph.fig, 2, 2, 'Padding','compact','TileSpacing','compact');
    ph.tlo = tlo;
    ph.ax1 = nexttile(tlo,1); % data vs model
    ph.ax2 = nexttile(tlo,2); % residuals
    ph.ax3 = nexttile(tlo,3); % free energy
    ph.ax4 = nexttile(tlo,4); % params
end

function do_plot(ph, Hz_, Sy_, Sm_, Ftr_, th_hist_, use_log_, beta_, iter_, show_params_)
    if ~isvalid(ph.fig), return; end
    % 1) data vs model
    axes(ph.ax1); cla(ph.ax1);
    if use_log_
        plot(ph.ax1, Hz_, log(Sy_), '.', 'DisplayName','log data'); hold(ph.ax1,'on');
        plot(ph.ax1, Hz_, log(Sm_), '-', 'DisplayName','log model');
        ylabel(ph.ax1,'log PSD'); 
    else
        plot(ph.ax1, Hz_, Sy_, '.', 'DisplayName','data'); hold(ph.ax1,'on');
        plot(ph.ax1, Hz_, Sm_, '-', 'DisplayName','model');
        ylabel(ph.ax1,'PSD');
    end
    xlabel(ph.ax1,'Hz'); title(ph.ax1, sprintf('Fit (\\beta=%.2f, iter=%d)', beta_, iter_));
    legend(ph.ax1,'Location','best'); grid(ph.ax1,'on');

    % 2) residuals
    axes(ph.ax2); cla(ph.ax2);
    if use_log_
        res = log(Sy_) - log(Sm_);
    else
        res = Sy_ - Sm_;
    end
    plot(ph.ax2, Hz_, res, '-');
    xlabel(ph.ax2,'Hz'); ylabel(ph.ax2,'residual'); title(ph.ax2,'Residuals'); grid(ph.ax2,'on');

    % 3) free-energy trace
    axes(ph.ax3); cla(ph.ax3);
    plot(ph.ax3, Ftr_, '-o'); xlabel(ph.ax3,'accepted step'); ylabel(ph.ax3,'F (surrogate)');
    title(ph.ax3,'Free-energy (↑ better)'); grid(ph.ax3,'on');

    % 4) parameters
    axes(ph.ax4); cla(ph.ax4);
    if show_params_ && ~isempty(th_hist_)
        plot(ph.ax4, th_hist_.'); xlabel(ph.ax4,'step'); ylabel(ph.ax4,'\theta (active)');
        title(ph.ax4,'Parameter trajectories'); grid(ph.ax4,'on');
    else
        text(ph.ax4,0.5,0.5,'(param plotting disabled)','HorizontalAlignment','center');
        axis(ph.ax4,'off');
    end
    set(findall(ph.fig,'-property','FontSize'),'FontSize',12);
    drawnow;
end

end
