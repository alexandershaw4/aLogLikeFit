% Example test function for the Vairational Laplace routine

% Ground truth parameters
a_true = 5; 
b_true = 1.5;
c_true = 10;

% Generate input data
x = linspace(0, 20, 50)';

% True function (logistic growth) with noise
sigma_true = 0.2 + 0.1 * randn(size(x)); % Heteroscedastic noise
y = a_true ./ (1 + exp(-b_true * (x - c_true))) + sigma_true .* randn(size(x));

% Plot data
figure;
scatter(x, y, 'k', 'filled');
xlabel('x'); ylabel('y');
title('Synthetic Data from Logistic Growth Model');
grid on;


% Model function for VL routine
f = @(m) m(1) ./ (1 + exp(-m(2) * (x - m(3)))); % Logistic function


% Initial guess for parameters [a, b, c]
m0 = [4; 1; 8];

% Initial covariance (diagonal for simplicity)
S0 = diag([1, 1, 1]);

% Run VL routine
[m_est, V_est, D_est, logL, iter, sigma2_est, allm] = fitVariationalLaplaceThermo(y, f, m0, S0, 50, 1e-6);


% Predictions from estimated parameters
y_pred = f(m_est);

% Plot results
figure; hold on;
scatter(x, y, 'k', 'filled', 'DisplayName', 'Observed Data');
plot(x, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'VL Fit');
xlabel('x'); ylabel('y');
title('Variational Laplace Fit to Logistic Model');
legend;
grid on;

% parameters
% Compute posterior variances (diagonal of S_post)
S_post_diag = sum(V_est.^2, 2) + D_est; % Variance from (VV' + D)

% Compute prior variances (diagonal of S0)
S_prior_diag = diag(S0); % Since S0 is initially diagonal

% Define range for plotting (mean ± 3 standard deviations)
xRangeFactor = 3;
numParams = length(m_est);

figure;
for i = 1:numParams
    % Define x range for prior and posterior distributions
    x_vals = linspace(m0(i) - xRangeFactor * sqrt(S_prior_diag(i)), ...
                      m0(i) + xRangeFactor * sqrt(S_prior_diag(i)), 100);

    % Prior Gaussian density
    y_prior = normpdf(x_vals, m0(i), sqrt(S_prior_diag(i)));

    % Define x range for posterior
    x_vals_post = linspace(m_est(i) - xRangeFactor * sqrt(S_post_diag(i)), ...
                           m_est(i) + xRangeFactor * sqrt(S_post_diag(i)), 100);

    % Posterior Gaussian density
    y_post = normpdf(x_vals_post, m_est(i), sqrt(S_post_diag(i)));

    % Plot prior and posterior
    subplot(ceil(numParams/2), 2, i);
    plot(x_vals, y_prior, 'b--', 'LineWidth', 2, 'DisplayName', 'Prior');
    hold on;
    plot(x_vals_post, y_post, 'r-', 'LineWidth', 2, 'DisplayName', 'Posterior');
    hold off;
    xlabel(['Parameter ', num2str(i)]);
    ylabel('Density');
    title(['Prior vs. Posterior for Parameter ', num2str(i)]);
    legend;
    grid on;
end

sgtitle('Comparison of Prior and Posterior Gaussians');

