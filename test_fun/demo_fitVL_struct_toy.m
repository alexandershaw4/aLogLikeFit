function demo_fitVL_struct_toy
% DEMO — Structure-learning Thermo-VL on a sparse linear model
% Model: y = X * theta  + N(0, sigma^2), with only K<<p non-zero entries in theta.
% We give the inference a *supermodel* with p candidates and ask it to
% "only update what pays rent".

rng(7);

% -------------------- generate toy data --------------------
N  = 400;                       % samples
p  = 60;                        % total candidate parameters (supermodel)
K  = 6;                         % number of truly active parameters
sigma_true = 0.08;              % obs noise

% Design matrix with correlated columns (harder structure learning)
x  = linspace(-2, 2, N)';
Phi = [ones(N,1), x, x.^2, sin(2*x), cos(3*x)];  % base dictionary (5 cols)
% inflate to p columns via random mixes of smooth bases
X = zeros(N,p);
for j=1:p
    w = randn(size(Phi,2),1);
    w = w / norm(w);
    X(:,j) = Phi*w + 0.15*randn(N,1);  % add mild collinearity/noise
end
% pick K true indices and amplitudes
true_idx = sort(randsample(p, K));
theta_true = zeros(p,1);
theta_true(true_idx) = [1.2, -0.9, 0.65, -0.5, 0.9, 0.35]';  % any values

y_clean = X*theta_true;
y = y_clean + sigma_true*randn(N,1);

% -------------------- wrap as f(theta) with analytic J --------------------
% This matches your function signature:
%   yhat = f(theta)
%   [yhat, J] = f(theta)
f = @(theta, varargin) f_lin(theta, X);

% -------------------- priors --------------------
m0 = zeros(p,1);
S0 = (0.5^2) * eye(p);          % moderately tight zero-mean prior

% -------------------- options for structure learning --------------------
opts = struct();
opts.tau_sched      = [linspace(0.4,1,6), ones(1,24)];  % explore → settle
opts.useARD         = true;          % ARD helps prune false positives
opts.lambda0        = 1e-3;
opts.lambda_max     = 1e8;
opts.z_thresh       = 2.2;           % z-score fallback trigger
opts.kappa_prune    = 0.0;           % prune if ΔF_keep ≤ 0
opts.kappa_add      = 2.0;           % add only if ΔF_add > 2 (~BF>~7)
opts.max_add_per_it = 3;

% Start *small* (empty or tiny seed); structure will grow if warranted
opts.I0             = [];            % start with nothing
% Proposer: offer all currently inactive columns; you can make this smarter
opts.propose_fun    = @(I,p_) setdiff((1:p_)', I);

% -------------------- run the fit --------------------
maxIter = 40; tol = 1e-3; plots = 0;
[m, V, D, logF, iter, s2, trace] = ...
  fitVariationalLaplaceThermoStruct(y, f, m0, S0, maxIter, tol, plots, [], opts);

% -------------------- post-hoc diagnostics & visuals --------------------
fprintf('Finished in %d iterations. Final ELBO ~ %.3f\n', iter, logF(end));
fprintf('True nonzeros:   %s\n', mat2str(true_idx(:)'));
active_final = sort(D.active(:)');
fprintf('Recovered active: %s\n', mat2str(active_final));
prec = length(intersect(true_idx, active_final)) / max(1,length(active_final));
rec  = length(intersect(true_idx, active_final)) / length(true_idx);
fprintf('Precision=%.2f, Recall=%.2f\n', prec, rec);

figure('Color','w','Position',[80 80 1400 720]);

% (1) True vs Estimated parameters (with CI)
subplot(2,2,1);
ci = 1.96*sqrt(max(real(diag(V)),0));
stem(theta_true, 'filled'); hold on;
errorbar(m, ci, 'LineStyle','none');
stem(m, ':'); hold off;
xlabel('\theta index'); ylabel('value');
title('Parameters: true (solid) vs posterior mean m (dotted) ±95% CI');
legend({'\theta_{true}','95% CI','m'}, 'Location','best'); grid on;

% (2) Active-set evolution (heat map)
subplot(2,2,2);
I_mat = active_matrix(trace, p);
imagesc(I_mat); colormap(gray);
xlabel('\theta index'); ylabel('iteration');
title('Active-set evolution (white=active)');
set(gca,'YDir','normal');

% (3) ELBO trajectory
subplot(2,2,3);
plot(logF,'-o'); grid on;
xlabel('outer iteration'); ylabel('Free energy (ELBO)');
title('Free-energy trajectory');

% (4) Residuals & fit quality
subplot(2,2,4);
yhat = X*m;
plot(y,'-'); hold on; plot(yhat,'--'); hold off; grid on;
xlabel('sample'); ylabel('signal');
title(sprintf('Data vs fit (R^2 = %.3f)', rsq(y,yhat)));

% Helpful printed summary of top candidates by |m|/sd
z = abs(m)./sqrt(max(real(diag(V)), eps));
[zs, ord] = sort(z,'descend');
fprintf('\nTop z-scores:\n');
disp(table(ord(1:10), zs(1:10), m(ord(1:10)), 'VariableNames', {'idx','z','m'}));

end % demo

% ---------- local utilities ----------
function [yhat, J] = f_lin(theta, X)
    yhat = X*theta;
    if nargout>1, J = X; end
end

function R2 = rsq(y, yhat)
    y = y(:); yhat = yhat(:);
    R2 = 1 - sum((y - yhat).^2)/sum((y - mean(y)).^2 + eps);
end

function I_mat = active_matrix(trace, p)
    % Build an iters×p binary matrix marking active parameters per iteration
    T = numel(trace);
    I_mat = zeros(T, p);
    for t=1:T
        if isfield(trace(t),'I') && ~isempty(trace(t).I)
            I_mat(t, trace(t).I) = 1;
        end
    end
    % display as white=active (flip colormap later)
    I_mat = 1 - I_mat;
end
