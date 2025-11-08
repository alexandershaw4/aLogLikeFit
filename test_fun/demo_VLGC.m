function R = demo_VLGC()
% DEMO_VLGC  Showcase strengths of fitVariationalLaplaceThermo_GC on a
% nonstationary oscillatory signal with heteroscedastic noise and outliers.
%
% Produces figures and returns a results struct R with metrics.

rng(7);

% ---------- Ground-truth generative process ----------
% y(t) = A * exp(-d*t) * sin(2*pi*(f0 + c*t).*t + phi)  +  B * sin(2*pi*f2*t)
% params θ = [A, d, f0, c, phi, B, f2]

Tsec   = 5.0;                 % seconds
fs     = 600;                 % Hz (fine grid)
dt     = 1/fs;
t      = (0:dt:Tsec-dt)';     % T×1
T      = numel(t);

theta_true = [...
    1.20;     % A: amplitude of damped chirp
    0.35;     % d: damping
    9.00;     % f0: base freq (Hz)
    1.50;     % c: chirp rate (Hz/sec)
    0.40;     % phi: phase (rad)
    0.40;     % B: tonic oscillator amplitude
    14.0];    % f2: tonic frequency (Hz)

[y_clean, dY_clean] = forward_model(theta_true, t);

% Heteroscedastic noise + sparse outliers
sigma0  = 0.15;
hetero  = 1 + 0.5*sin(2*pi*0.6*t + 0.3);         % slow variance modulation
eps_t   = sigma0*hetero .* randn(T,1);
y_noisy = y_clean + eps_t;

% Inject outliers (rare spikes)
out_idx = randsample(T, round(0.01*T));          % 1% outliers
y_noisy(out_idx) = y_noisy(out_idx) + 3*sigma0.*randn(numel(out_idx),1);

%---------- Priors ----------
P = numel(theta_true);
m0 = theta_true .* (1 + 0.20*randn(P,1));        % biased initialisation
S0 = diag([0.5  0.5  2.0  1.0  1.0  0.5  1.5].^2); % fairly loose prior SDs

maxIter = 256;
tol     = 1e-4;

% Common opts
opts          = struct;
opts.varpercthresh = 0.02;
opts.plots    = 0;
opts.gc       = struct('order',2,'dt',dt,'op','fd2','boundary','rep','lambda',[]);

% Compare q = 0,1,2
q_list = [0 1 2];
for qi = 1:numel(q_list)
    q     = q_list(qi);
    opts_i = opts;
    opts_i.gc.order = q;

    f = @(theta) forward_wrapper(theta, t); % returns [yhat, dY] when asked

    fprintf('\n=== Fitting with GC order q = %d ===\n', q);
    [m{qi}, V{qi}, D{qi}, F{qi}, it{qi}, s2{qi}, allm{qi}, Fellbo{qi}] = ...
        fitVariationalLaplaceThermo_GC(y_noisy, f, m0, S0, maxIter, tol, opts_i);

    % Compute per-order fits using final posterior mean
    [yhat{qi}, dY{qi}] = forward_model(m{qi}, t);

    % Forecast on held-out future segment
    Tsec_fore = 1.0;
    t_fore    = (Tsec:dt:Tsec+Tsec_fore-dt)';            % 1s ahead
    [y_true_fore, ~] = forward_model(theta_true, t_fore);
    [yhat_fore{qi}, ~] = forward_model(m{qi}, t_fore);

    % Metrics
    MET(qi) = compute_metrics(y_clean, dY_clean, y_noisy, yhat{qi}, dY{qi}, ...
                              t, Fellbo{qi}, y_true_fore, yhat_fore{qi}, q);
end

% ---------- Visualisation ----------
figure('Name','VL-GC demo: Fit comparisons','Position',[100 80 1500 900]);

% Panel 1: raw fit
subplot(3,3,1);
plot(t, y_noisy, 'Color',[0.6 0.6 0.6]); hold on;
plot(t, y_clean, 'k','LineWidth',1.2);
plot(t, yhat{1}, 'r','LineWidth',1.0);
plot(t, yhat{2}, 'g','LineWidth',1.0);
plot(t, yhat{3}, 'b','LineWidth',1.0);
title('Signal fit (order 0)'); grid on; box on;
legend({'Observed (noisy)','True','q=0','q=1','q=2'},'Location','best');
xlabel('t (s)'); ylabel('y');

% Panel 2-3: derivative fits
subplot(3,3,2);
plot(t, dY_clean(:,1),'k','LineWidth',1.2); hold on;
if ~isempty(dY{1}), plot(t,dY{1}(:,1),'r'); end
if ~isempty(dY{2}), plot(t,dY{2}(:,1),'g'); end
if ~isempty(dY{3}), plot(t,dY{3}(:,1),'b'); end
title('First derivative fit (order 1)'); grid on; box on;
legend({'True \dot y','q=0','q=1','q=2'},'Location','best');

subplot(3,3,3);
plot(t, dY_clean(:,2),'k','LineWidth',1.2); hold on;
if ~isempty(dY{1}), plot(t,dY{1}(:,2),'r'); end
if ~isempty(dY{2}), plot(t,dY{2}(:,2),'g'); end
if ~isempty(dY{3}), plot(t,dY{3}(:,2),'b'); end
title('Second derivative fit (order 2)'); grid on; box on;
legend({'True \ddot y','q=0','q=1','q=2'},'Location','best');

% Panel 4: ELBO trajectories
subplot(3,3,4);
hold on;
for qi=1:numel(q_list)
    plot(1:numel(Fellbo{qi}), Fellbo{qi}, 'LineWidth',1.0);
end
title('ELBO trajectories'); grid on; box on;
xlabel('Iteration'); ylabel('ELBO');
legend({'q=0','q=1','q=2'},'Location','southeast');

% Panel 5: Residual PSD (Welch) (show reduction of dynamic mismatch)
subplot(3,3,5);
for qi=1:numel(q_list)
    r = y_clean - yhat{qi};
    [Pxx,Fw] = pwelch(r, hamming(512), 256, 512, fs);
    plot(Fw,Pxx,'LineWidth',1.0); hold on;
end
title('Residual PSD'); grid on; box on;
xlabel('Hz'); ylabel('PSD');
legend({'q=0','q=1','q=2'},'Location','northeast');

% Panel 6: Forecast accuracy
subplot(3,3,6);
bar([MET.RMSE_fore]); xticklabels({'q=0','q=1','q=2'});
ylabel('RMSE (forecast)'); title('Out-of-sample forecast');
grid on; box on;

% Panel 7-9: Text metrics
subplot(3,3,7); axis off;
text(0,1, sprintf('Signal RMSE (train):\n q=0: %.3f\n q=1: %.3f\n q=2: %.3f', ...
    MET(1).RMSE_sig, MET(2).RMSE_sig, MET(3).RMSE_sig), 'FontSize',11);
text(0,0.4, sprintf('Derivative RMSE (train):\n q=0: [%.3f, %.3f]\n q=1: [%.3f, %.3f]\n q=2: [%.3f, %.3f]', ...
    MET(1).RMSE_dy(1), MET(1).RMSE_dy(2), MET(2).RMSE_dy(1), MET(2).RMSE_dy(2), MET(3).RMSE_dy(1), MET(3).RMSE_dy(2)), 'FontSize',11);

subplot(3,3,8); axis off;
text(0,1, sprintf('Neg. log-pred (hetero) ↓:\n q=0: %.1f\n q=1: %.1f\n q=2: %.1f', ...
    MET(1).NLPD, MET(2).NLPD, MET(3).NLPD), 'FontSize',11);

subplot(3,3,9); axis off;
text(0,1, sprintf('Best ELBO:\n q=0: %.1f @%d\n q=1: %.1f @%d\n q=2: %.1f @%d', ...
    MET(1).bestELBO, MET(1).bestIter, MET(2).bestELBO, MET(2).bestIter, MET(3).bestELBO, MET(3).bestIter), 'FontSize',11);

sgtitle('VL in Generalised Coordinates — Complex Demo');

% ---------- Coarse sampling ablation (optional, illustrates GC robustness) ----------
fs2  = 150;                      % 4× downsample
y_ds  = decimate(y_noisy, fs/fs2);
t_ds  = (0:1/fs2:Tsec-1/fs2)';

opts2          = opts;
opts2.gc.dt    = 1/fs2;
opts2.gc.order = 0;  f0 = @(th) forward_wrapper(th, t_ds);
[~,~,~,~,~,~,~,F0] = fitVariationalLaplaceThermo_GC(y_ds, f0, m0, S0, maxIter, tol, setfield(opts2,'plots',0)); %#ok<SFLD>

opts2.gc.order = 2;  f2 = @(th) forward_wrapper(th, t_ds);
[~,~,~,~,~,~,~,F2] = fitVariationalLaplaceThermo_GC(y_ds, f2, m0, S0, maxIter, tol, setfield(opts2,'plots',0));

fprintf('\nAblation (downsampled %d Hz): best ELBO q=0: %.1f, q=2: %.1f (GC typically wins)\n', fs2, max(F0), max(F2));

% ---------- Return struct ----------
R = struct('t',t,'y_true',y_clean,'y_noisy',y_noisy, ...
           'theta_true',theta_true, 'm',{m}, 'V',{V}, 'D',{D}, ...
           'yhat',{yhat}, 'dY',{dY}, 'ELBO',{Fellbo}, 'metrics',MET);

end % demo_VLGC


% ======= Helpers =======

function [y, dY] = forward_model(theta, t)
% Nonstationary oscillator (damped chirp) + tonic oscillator.
% Returns y and analytic time-derivatives dY(:,1)=dy/dt, dY(:,2)=d2y/dt2.
A   = theta(1); d = theta(2); f0= theta(3); c = theta(4);
phi = theta(5); B = theta(6); f2 = theta(7);

% phase for chirp: phi_c(t) = 2*pi*(f0 t + 0.5 c t^2) + phi
phi_c  = 2*pi*(f0*t + 0.5*c*t.^2) + phi;
e_dt   = exp(-d*t);
y1     = A * e_dt .* sin(phi_c);
y2     = B * sin(2*pi*f2*t);
y      = y1 + y2;

% analytic derivatives
% d/dt [A e^{-d t} sin(φ_c)] = A e^{-d t}[ cos(φ_c)*(2π(f0 + c t)) - d sin(φ_c) ]
omega_c  = 2*pi*(f0 + c*t);
dy1      = A*e_dt .* (cos(phi_c).*omega_c - d*sin(phi_c));
d2y1     = A*e_dt .* ( ...
              -sin(phi_c).*omega_c.^2 ...
              - 2*d*cos(phi_c).*omega_c ...
              + (2*pi*c)*cos(phi_c) ...
              + d^2 * sin(phi_c) );

dy2      = B * 2*pi*f2 * cos(2*pi*f2*t);
d2y2     = -B * (2*pi*f2)^2 * sin(2*pi*f2*t);

dy       = dy1 + dy2;
d2y      = d2y1 + d2y2;

dY = [dy, d2y];
end

function varargout = forward_wrapper(theta, t)
% Wrapper that supports either 1 or 2 outputs for fitVLtherm_GC.
[y, dY] = forward_model(theta, t);
switch nargout
    case 1
        varargout = {y};
    otherwise
        varargout = {y, dY};
end
end

function M = compute_metrics(y_true, dY_true, y_obs, yhat, dYhat, t, ELBO, y_fore_true, y_fore_hat, q)
% Training RMSEs
RMSE_sig = sqrt(mean((y_true - yhat).^2));
if isempty(dYhat)
    RMSE_dy = [NaN NaN];
else
    RMSE_dy = [ ...
        sqrt(mean((dY_true(:,1) - dYhat(:,1)).^2)), ...
        sqrt(mean((dY_true(:,2) - dYhat(:,2)).^2)) ];
end

% Heteroscedastic negative log predictive density (simple proxy)
% Estimate local variance from residuals with a moving window
r   = y_obs - yhat;
w   = max(11, round(numel(r)/80));
r2  = movmean(r.^2, w, 'Endpoints','shrink');
r2  = max(r2, 1e-6);
NLPD = 0.5*sum(log(2*pi*r2) + (r.^2)./r2);

% Forecast RMSE
RMSE_fore = sqrt(mean((y_fore_true - y_fore_hat).^2));

M = struct('q',q,'RMSE_sig',RMSE_sig,'RMSE_dy',RMSE_dy, ...
           'NLPD',NLPD,'RMSE_fore',RMSE_fore, ...
           'bestELBO',max(ELBO),'bestIter',find(ELBO==max(ELBO),1));
end
