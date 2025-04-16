% --- Test Function for Variational Laplace: Bi-Exponential Decay ---

% Ground truth parameters
a1_true = 2.0;
b1_true = 1.2;
a2_true = 1.0;
b2_true = 0.2;

% Generate input data
x = linspace(0, 10, 100)';

% Generate synthetic data with noise
rng(1);
y_clean = a1_true * exp(-b1_true * x) + a2_true * exp(-b2_true * x);
noise = 0.05 + 0.05 * randn(size(x));  % slight heteroscedasticity
y = y_clean + noise .* randn(size(x));

% Plot raw data
figure;
scatter(x, y, 20, 'k', 'filled');
xlabel('x'); ylabel('y');
title('Synthetic Bi-Exponential Decay Data');
grid on;

% Define model function: f(m) = a1 * exp(-b1*x) + a2 * exp(-b2*x)
f = @(m) m(1) * exp(-m(2) * x) + m(3) * exp(-m(4) * x);  % [a1, b1, a2, b2]

% Initial guess
m0 = [1.5; 0.5; 0.8; 0.1];

% Prior covariance (diagonal)
S0 = diag([1, 0.1, 1, 0.05]);

% Fit using your VL routine
[m_est, V_est, D_est, logL, iter, sigma2_est, allm] = fitVariationalLaplaceThermo(y, f, m0, S0, 50, 1e-6);

% Predictions from estimated parameters
y_pred = f(m_est);

% Plot results
figure; hold on;
scatter(x, y, 20, 'k', 'filled', 'DisplayName', 'Observed');
plot(x, y_clean, 'b--', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
plot(x, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'VL Fit');
xlabel('x'); ylabel('y');
legend('Location', 'best');
title('Bi-Exponential Fit using Variational Laplace');
grid on;

% Plot prior vs posterior marginals
S_post_diag = sum(V_est.^2, 2) + D_est;
S_prior_diag = diag(S0);
xRangeFactor = 3;
numParams = length(m_est);

figure;
for i = 1:numParams
    x_vals = linspace(m0(i) - xRangeFactor * sqrt(S_prior_diag(i)), ...
                      m0(i) + xRangeFactor * sqrt(S_prior_diag(i)), 100);
    y_prior = normpdf(x_vals, m0(i), sqrt(S_prior_diag(i)));

    x_vals_post = linspace(m_est(i) - xRangeFactor * sqrt(S_post_diag(i)), ...
                           m_est(i) + xRangeFactor * sqrt(S_post_diag(i)), 100);
    y_post = normpdf(x_vals_post, m_est(i), sqrt(S_post_diag(i)));

    subplot(2,2,i);
    plot(x_vals, y_prior, 'b--', 'LineWidth', 2, 'DisplayName', 'Prior');
    hold on;
    plot(x_vals_post, y_post, 'r-', 'LineWidth', 2, 'DisplayName', 'Posterior');
    xlabel(['Parameter ', num2str(i)]);
    ylabel('Density');
    title(['Prior vs. Posterior for Param ', num2str(i)]);
    legend;
    grid on;
end
sgtitle('Bi-Exponential Model: Prior vs Posterior Marginals');
