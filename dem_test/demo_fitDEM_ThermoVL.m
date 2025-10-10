function demo_fitDEM_ThermoVL()
% DEMO for fitDEM_ThermoVL on a 2-state nonlinear damped oscillator
%   x1' = x2
%   x2' = -k*x1 - c*x2 + a*tanh(x1)
%   y   = x1 + e
%
% True theta = [k; c; a]; we try to recover them from y(t).

rng(42);

% ----- ground truth system -----
dt     = 1/200;             % 200 Hz
Tsec   = 5;                 % 5 seconds
T      = round(Tsec/dt);    % samples
nx     = 2; ny = 1;

theta_true = [ (2*pi*2)^2;   0.8*(2*pi*2);  0.6 ]; % k, c, a
x0_true    = [0.8; 0.0];

% simulate noiseless states
x = zeros(nx,T);
x(:,1) = x0_true;
for t = 2:T
    x(:,t) = x(:,t-1) + dt * f_osc_nl(x(:,t-1), theta_true);
end

% generate observations
sigma_y = 0.05;                  % observation noise SD
y = g_lin(x) + sigma_y*randn(1,T);
y = y.';                         % T x ny

% ----- estimation setup -----
% initial guesses
x0_guess  = [0; 0];
theta0    = [ (2*pi*1.5)^2;  0.5*(2*pi*2);  0.0 ];

opts = struct;
opts.dt            = dt;
opts.p             = 2;       % GC order for states
opts.py            = 0;       % GC order for outputs (0th order only)
opts.Dx_iters      = 2;
opts.outer_iters   = 10;
opts.gamma_y       = 1.0;
opts.gamma_x       = 1.0;
%opts.lambda_y0     = 1/var(y,0,'all');
opts.lambda_y0     = 1/max(var(y(:)), 1e-6);
opts.lambda_x0     = 1.0;
opts.pE_theta      = theta0;
opts.pC_theta      = eye(numel(theta0))*1e2;
opts.pE_x0         = zeros((opts.p+1)*nx,1);
opts.pC_x0         = eye((opts.p+1)*nx)*1e2;
opts.varpercthresh = 0.01;
opts.max_rank_theta= numel(theta0);
opts.plot          = 1;

% function handles matching the required signatures:
%   f(x,theta)->nx×1, g(x,theta)->ny×1
f = @(x,theta) f_osc_nl(x,theta);
g = @(x,theta) g_lin(x);

% ----- fit -----
[POST, TRACE] = fitDEM_ThermoVL(y, f, g, x0_guess, theta0, opts);

% ----- quick report -----
fprintf('\n=== Parameter recovery ===\n');
disp(table(theta_true(:), POST.theta_mean(:), ...
    'VariableNames',{'theta_true','theta_est'}));

% overlay fit
tspan = (0:T-1)*dt;
yhat = zeros(T,ny);
for tt = 1:T
    yhat(tt,:) = g(POST.xgc_path(tt,1:nx).', POST.theta_mean);
end

figure('Color','w','Name','Fit overlay');
plot(tspan, y, 'k-', 'DisplayName','Observed'); hold on;
plot(tspan, yhat, 'r-', 'DisplayName','Predicted');
xlabel('Time (s)'); ylabel('y'); legend; title('Observed vs Predicted');

% parameter trajectory
figure('Color','w','Name','Parameter trajectory');
plot(TRACE.theta_hist','-o'); hold on;
yline(theta_true(1),'--'); yline(theta_true(2),'--'); yline(theta_true(3),'--');
xlabel('Outer iteration'); ylabel('\theta'); title('Parameter means over iterations');
legend({'\theta_1','\theta_2','\theta_3','true'},'Location','best');

end
