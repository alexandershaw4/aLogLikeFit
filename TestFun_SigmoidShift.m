% --- Test Function for Variational Laplace: Sigmoid + Linear Drift ---

% Ground truth parameters
a_true = 6;       % Sigmoid amplitude
b_true = 1.5;     % Sigmoid slope
c_true = 12;      % Sigmoid centre
d_true = 0.5;     % Linear drift

% Generate input data
x = linspace(0, 20, 100)';

% Generate synthetic data with heteroscedastic noise
rng(2);
y_clean = a_true ./ (1 + exp(-b_true * (x - c_true))) + d_true * x;
sigma = 0.2 + 0.1 * randn(size(x));  % small random heteroscedasticity
y = y_clean + sigma .* randn(size(x));

% Plot raw data
figure;
scatter(x, y, 20, 'k', 'filled');
xlabel('x'); ylabel('y');
title('Synthetic Data: Sigmoid + Linear Drift');
grid on;

% Define model function: f(m) = a / (1 + exp(-b*(x - c))) + d*x
f = @(m) m(1) ./ (1 + exp(-m(2)*(x - m(3)))) + m(4)*x;  % [a, b, c, d]

% Initial guess
m0 = [5; 1; 10; 0.3];

% Prior covariance (diagonal)
S0 = diag([2, 0.5, 1, 0.1]);

% Run VL routine
[m_est, V_est, D_est, logL, iter, sigma2_est, allm] = ...
    fitVariationalLaplaceThermo(y, f, m0, S0, 50, 1e-6);

% Predictions from estimated parameters
y_pred = f(m_est);

% Plot results
figure; hold on;
scatter(x, y, 20, 'k', 'filled', 'DisplayName', 'Observed');
plot(x, y_clean, 'b--', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
plot(x, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'VL Fit');
xlabel('x'); ylabel('y');
legend('Location', 'best');
title('Sigmoid + Drift Fit using Variational Laplace');
grid on;

% Plot prior vs posterior marginals
S_post_diag = sum(V_est*V_est') + D_est;
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
sgtitle('Sigmoid + Drift Model: Prior vs Posterior Marginals');
