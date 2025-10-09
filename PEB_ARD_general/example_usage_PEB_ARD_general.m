% Synthetic regression example (works everywhere)
rng(1)
N = 200; p = 5;
X = randn(N, p);
true_beta = [2; -1; 0; 0.5; 0];
y = X*true_beta + 0.2*randn(N,1);
X = [ones(N,1), X];   % intercept

% Fit
M = peb_ard_novar(y, X);

% Predict on same data
[yhat, ~] = peb_ard_predict(X, M);

fprintf('R² = %.3f\n', 1 - sum((y - yhat).^2)/sum((y - mean(y)).^2));
disp('Estimated β:'); disp(M.beta);
disp('True β (approx, first five features):');
disp([0; true_beta]);

% plots
peb_plot_betas(M)
peb_plot_beta_densities(M)
peb_plot_lambda(M)

% Example density ribbon for one coefficient (say X2):
Bsamp = peb_sample_beta(M, 5000);
figure; histogram(Bsamp(:,3), 60, 'Normalization','pdf'); grid on
title('Posterior samples for \beta_{X2}')
xlabel('\beta'); ylabel('density')
