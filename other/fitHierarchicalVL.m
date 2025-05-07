% fitHierarchicalVL.m
% ====================
% Iterative Hierarchical Variational Laplace using fitVL_LowRankNoise for subject-level fits.
% Updates group-level priors iteratively using empirical Bayes.
% Adds basic group statistics (t-values and p-values) for each parameter.

function results = fitHierarchicalVL(data, f, m0, S0, maxIter, tol, nIter, design)
% Inputs:
% -------
% data     : Cell array of observed data per subject, {y1, y2, ..., yN}
% f        : Function handle for model prediction: y = f(m)
% m0       : Initial mean of subject-level prior
% S0       : Initial covariance of subject-level prior
% maxIter  : Max iterations for subject-level inference
% tol      : Tolerance for convergence in subject fits
% nIter    : Number of top-level (hierarchical) iterations
% design   : Optional design matrix for group-level effects (N x p)

if nargin < 8
    design = [];
end

N = numel(data);
d = length(m0);

% Initialize group-level prior
mu_g = m0;
Sigma_g = S0;

% Store results
subject_posteriors = cell(N,1);
all_group_means = zeros(d, nIter);
all_group_covs = cell(nIter, 1);

for iter = 1:nIter
    fprintf('\nHierarchical iteration %d/%d\n', iter, nIter);
    m_all = zeros(d,N);
    S_all = zeros(d,d,N);

    % ====== Subject-level inference ======
    parfor i = 1:N
        y_i = data{i};
        [m_i, V_i, D_i, ~, ~, ~, ~] = fitVL_LowRankNoise(y_i, f, mu_g, Sigma_g, maxIter, tol, 0);
        S_i = V_i * V_i' + diag(D_i);

        m_i = denan(real(m_i));
        S_i = denan(real(S_i));

        m_all(:,i) = m_i;
        S_all(:,:,i) = S_i;
        subject_posteriors{i}.m = m_i;
        subject_posteriors{i}.S = S_i;
    end

    % ====== Group-level update ======
    mu_g = mean(m_all, 2);
    Sigma_g = zeros(d);
    for i = 1:N
        m_i = m_all(:,i);
        S_i = S_all(:,:,i);
        Sigma_g = Sigma_g + S_i + (m_i - mu_g) * (m_i - mu_g)';
    end
    Sigma_g = Sigma_g / N;

    % Optional shrinkage or regularisation (to keep positive-definite)
    Sigma_g = Sigma_g + 1e-6 * eye(d);

    % Store
    all_group_means(:,iter) = mu_g;
    all_group_covs{iter} = Sigma_g;
end

% Optional: project onto group-level covariates (design matrix)
if ~isempty(design)
    % Use ridge regression for robustness
    X = design;
    beta = (X' * X + 1e-6 * eye(size(X,2))) \ (X' * m_all');
    group_model.X = X;
    group_model.beta = beta;
    group_model.fitted = X * beta;

    % If design is binary group contrast, compute t- and p-values
    if size(X,2) == 2 && all(ismember(unique(X(:,2)), [0, 1]))
        group1_idx = X(:,2) == 0;
        group2_idx = X(:,2) == 1;
        m1 = m_all(:, group1_idx);
        m2 = m_all(:, group2_idx);

        n1 = sum(group1_idx);
        n2 = sum(group2_idx);

        mean_diff = mean(m2,2) - mean(m1,2);
        pooled_std = sqrt(((std(m1,0,2)).^2 + (std(m2,0,2)).^2) / 2);
        t_vals_contrast = mean_diff ./ (pooled_std .* sqrt(2 / min(n1, n2)));
        df = n1 + n2 - 2;
        p_vals_contrast = 2 * (1 - tcdf(abs(t_vals_contrast), df));

        group_model.contrast.group1_mean = mean(m1,2);
        group_model.contrast.group2_mean = mean(m2,2);
        group_model.contrast.t_vals = t_vals_contrast;
        group_model.contrast.p_vals = p_vals_contrast;
    end
else
    group_model = [];
end

% ====== Compute group statistics ======
% Mean and std for each parameter
param_means = mean(m_all, 2);
param_stds = std(m_all, 0, 2);
t_vals = param_means ./ (param_stds / sqrt(N));
p_vals = 2 * (1 - tcdf(abs(t_vals), N - 1));

% Output structure
results.subject_posteriors = subject_posteriors;
results.group_means = all_group_means;
results.group_covariances = all_group_covs;
results.final_mu_g = mu_g;
results.final_Sigma_g = Sigma_g;
results.group_model = group_model;
results.all_subject_means = m_all;
results.group_stats.t_vals = t_vals;
results.group_stats.p_vals = p_vals;
results.group_stats.param_means = param_means;
results.group_stats.param_stds = param_stds;
end
