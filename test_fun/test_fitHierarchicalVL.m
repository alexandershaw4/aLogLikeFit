function test_fitHierarchicalVL()
% Test the fitHierarchicalVL routine on synthetic multimodal data

%rng(42);  % For reproducibility

% -------------------------
% Synthetic model definition
% -------------------------
T = 50;             % Time points
f = @(m) sin(linspace(0, 2*pi, T)' * m(1)) + m(2);
% Parameters
d = 2;              % 2 parameters: frequency + bias
N_per_group = 25;
N = 2 * N_per_group;

% True group means
true_params_group1 = [0.2; 2.0];
true_params_group2 = [0.5; -3.0];

% Covariance for subject variability
true_cov = 0.05 * eye(d);

% Generate data
data = cell(1, N);
true_params = zeros(d, N);
for i = 1:N
    if i <= N_per_group
        m_true = mvnrnd(true_params_group1, true_cov)';
    else
        m_true = mvnrnd(true_params_group2, true_cov)';
    end
    y_i = f(m_true) + 0.1 * randn(T, 1);  % Add Gaussian noise
    data{i} = y_i;
    true_params(:, i) = m_true;
end

% -------------------------
% Initial priors
% -------------------------
m0 = zeros(d, N);                     % Start from flat prior per subject
S0 = eye(d) * 1;                      % Loose prior covariance
maxIter = 30;
tol = 1e-5;
nIter = 12;

% -------------------------
% Design matrix
% -------------------------
group_labels = [zeros(N_per_group, 1); ones(N_per_group, 1)];
design = [ones(N, 1), group_labels];  % Intercept + group

% -------------------------
% Run hierarchical fit
% -------------------------
results = fitHierarchicalVL(data, f, m0', S0, maxIter, tol, nIter, design);

% -------------------------
% Visualisation
% -------------------------
figure;
for j = 1:d
    subplot(1, d, j);
    g1 = results.all_subject_means(j, group_labels == 0);
    g2 = results.all_subject_means(j, group_labels == 1);
    boxplot([g1; g2], [zeros(size(g1))'; ones(size(g2))']);
    title(sprintf('Param %d — p=%.4f', j, results.group_stats.p_vals(j)));
    ylabel(sprintf('Param %d value', j));
    xlabel('Group');
end
sgtitle('Group Comparison After Hierarchical Fit');

end
