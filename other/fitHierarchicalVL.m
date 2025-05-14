function results = fitHierarchicalVL(data, f, m0, S0, maxIter, tol, nIter, design)
% fitHierarchicalVL - Iterative hierarchical variational Laplace inference
%
% Iteratively fits a hierarchical Bayesian model using low-rank variational Laplace
% (via fitVariationalLaplaceThermo) for subject-level inference and empirical Bayes updates
% for group-level prior estimation.
%
% This function refines subject-level parameter estimates and updates the group-level
% prior (mean and covariance) over multiple iterations, allowing for optional modeling
% of group-level covariates via a design matrix.
%
% Inputs:
% -------
% data     : Cell array of observed data per subject, {y1, y2, ..., yN}
% f        : Function handle for the generative model: y = f(m)
% m0       : Initial group-level mean vector (prior mean)
% S0       : Initial group-level covariance matrix (prior covariance)
% maxIter  : Maximum iterations for subject-level inference using fitVL_LowRankNoise
% tol      : Convergence tolerance for subject-level evidence (F) changes
% nIter    : Number of top-level iterations (group-level prior updates)
% design   : [Optional] Design matrix (N x p) for between-subject effects; if provided,
%            regression is performed on inferred subject-level means to yield group-level
%            parameter estimates (e.g. beta weights, t-stats, p-values)
%
% Outputs:
% --------
% results - Struct with fields:
%   .Ep         : Cell array of posterior means for each subject
%   .Cp         : Cell array of posterior covariances for each subject
%   .F          : Free energy (evidence lower bound) per subject per iteration
%   .m          : Final estimated group-level prior mean
%   .S          : Final estimated group-level prior covariance
%   .B          : [Optional] Group-level regression coefficients (if design provided)
%   .T          : [Optional] t-statistics for each parameter (design-based)
%   .P          : [Optional] p-values associated with t-statistics
%
% Notes:
% ------
% - This routine assumes independent subjects and approximates posteriors with Gaussians.
% - Group-level covariance updates assume a diagonal or low-rank structure for scalability.
% - Intended for models where a subject-level generative model can be efficiently fit via
%   variational methods with heteroscedastic (low-rank) noise modeling.
%
% Example:
%   results = fitHierarchicalVL(y, f, m0, S0, 8, 1e-6, 5, design);
%
% Dependencies:
%   Requires fitVariationalLaplaceThermo for subject-level fitting.
%
% AS2025

if nargin < 8
    design = [];
else
    design = design - mean(design);
end

N = numel(data);
d = size(m0,2);

mu_g = mean(m0,1);
Sigma_g = S0;

subject_posteriors = cell(N,1);
all_group_means = zeros(d, nIter);
mu_subjectwise = m0';
F_all_iters = zeros(N, nIter);

for iter = 1:nIter
    fprintf('\nHierarchical iteration %d/%d\n', iter, nIter);
    m_all = mu_subjectwise;
    S_all = zeros(d,d,N);

    % ====== Subject-level inference ======
    if iter > 1
        fprintf('Individual level fits...\n');
        parfor i = 1:N
            y_i = data{i};
            group_mu_i = mu_subjectwise(:, i);
            [m_i, V_i, D_i, logL, ~, ~, ~] = fitVariationalLaplaceThermo(y_i, f, group_mu_i, Sigma_g, maxIter, tol, 0);
            S_i = V_i * V_i' + diag(D_i);
            F_all_iters(i, iter) = logL;
            m_all(:,i) = denan(m_i);
            S_all(:,:,i) = denan(S_i);
            subject_posteriors{i}.m = m_all(:,i);
            subject_posteriors{i}.S = S_all(:,:,i);
        end
    end

    % ====== PCA-derived latent group assignment ======
    if ~isempty(design)
        m_for_pca = real(m_all)';
        [~, score, latent] = pca(m_for_pca);
        explained_var = cumsum(latent) / sum(latent);
        k_est = find(explained_var >= 0.90, 1, 'first');
        k_est = max(2, min(k_est, 8));
        try
            latent_group = kmeans(score(:, 1:k_est), k_est, 'Replicates', 5);
        catch
            latent_group = double(score(:,1) > median(score(:,1))) + 1;
            k_est = 2;
        end
        design(:,3) = latent_group;
        fprintf('Current estimate: %d subgroups\n', k_est);
        results.latent_pca_scores = score(:,1);
        results.estimated_k = k_est;
        results.explained_variance = explained_var;
    end

    % ====== Group-level prior updates with precision weighting ======
    if iter < nIter
        fprintf('Group level fits...\n');
        group_mu_suggestions = zeros(d, N, 2);

        for j = 2:3
            predictor = design(:, j);
            [groupIDs, ~, groupIndex] = unique(predictor);
            G = numel(groupIDs);
            group_means = zeros(d, G);
            for g = 1:G
                idx = (groupIndex == g);
                m_g = m_all(:, idx);
                S_g = S_all(:, :, idx);
                Lambda_sum = zeros(d);
                weighted_m_sum = zeros(d,1);
                for s = 1:sum(idx)
                    Si = S_g(:,:,s) + 1e-6 * eye(d);
                    Lambda_i = diag(1 ./ max(1e-6, diag(Si)));
                    Lambda_sum = Lambda_sum + Lambda_i;
                    weighted_m_sum = weighted_m_sum + Lambda_i * m_g(:, s);
                end
                group_means(:, g) = Lambda_sum \ weighted_m_sum;
            end

            for i = 1:N
                g = groupIndex(i);
                group_mu_suggestions(:, i, j - 1) = group_means(:, g);
            end
        end

        % ====== Combine priors with weighted shrinkage ======
        %alpha_diag   = 0.02;
        %alpha_latent = 0.01;

        for i = 1:N

            if iter > 1
                elbo_current = F_all_iters(i, iter);
                elbo_prev = F_all_iters(i, iter - 1);
    
                deltaF = elbo_current - elbo_prev;
                alpha_diag   = 0.02 / (1 + exp(deltaF));
                alpha_latent = 0.01 / (1 + exp(deltaF));
            else
                alpha_diag   = 0.02;
                alpha_latent = 0.01;
            end
        
            prior_diag   = group_mu_suggestions(:, i, 1);
            prior_latent = group_mu_suggestions(:, i, 2);
            m_i = m_all(:, i);
            m_new = m_i;
            m_new = (1 - alpha_diag)   * m_new + alpha_diag   * prior_diag;
            m_new = (1 - alpha_latent) * m_new + alpha_latent * prior_latent;
            mu_subjectwise(:, i) = m_new;
        end

        mu_g = mean(mu_subjectwise, 2);
        all_group_means(:, iter) = mu_g;
    end
end

% ====== Refit poorly performing subjects using original priors ======
fprintf('\nRefitting low-performing subjects from original priors...\n');

refit_thresh = quantile(F_all_iters(:, end), 0.05);  % bottom 5% ELBOs
refit_count = 0;

for i = 1:N
    if F_all_iters(i, end) < refit_thresh
        fprintf('  Refit subject %d (ELBO = %.2f)...\n', i, F_all_iters(i, end));
        y_i = data{i};
        m0_i = m0(i, :)';           % original prior mean
        S0_i = S0;                  % original prior covariance

        [m_refit, V_i, D_i, logL, ~, ~, ~] = fitVariationalLaplaceThermo(...
            y_i, f, m0_i, S0_i, maxIter * 2, tol, 0);

        S_refit = V_i * V_i' + diag(D_i);

        m_all(:, i) = m_refit;
        S_all(:, :, i) = S_refit;
        subject_posteriors{i}.m = m_refit;
        subject_posteriors{i}.S = S_refit;
        F_all_iters(i, end) = logL;

        refit_count = refit_count + 1;
    end
end

fprintf('Refit %d subject(s) using original priors.\n', refit_count);

if ~isempty(design)
    group_col = design(:,2);
    group_ids = unique(group_col);

    if numel(group_ids) == 2
        idx1 = group_col == group_ids(1);
        idx2 = group_col == group_ids(2);

        m1 = m_all(:, idx1);
        m2 = m_all(:, idx2);

        n1 = sum(idx1);
        n2 = sum(idx2);

        mean1 = mean(m1, 2);
        mean2 = mean(m2, 2);

        mean_diff = mean2 - mean1;
        pooled_std = sqrt(((std(m1, 0, 2)).^2 + (std(m2, 0, 2)).^2) / 2);
        t_vals = mean_diff ./ (pooled_std .* sqrt(2 / min(n1, n2)));
        df = n1 + n2 - 2;
        p_vals = 2 * (1 - tcdf(abs(t_vals), df));

        group_model.group_labels = group_ids;
        group_model.group1_idx = idx1;
        group_model.group2_idx = idx2;
        group_model.group1_mean = mean1;
        group_model.group2_mean = mean2;
        group_model.mean_diff = mean_diff;
        group_model.t_vals = t_vals;
        group_model.p_vals = p_vals;

        results.group_model = group_model;
    else
        warning('design(:,2) has %d groups — skipping contrast stats.', numel(group_ids));
        results.group_model = [];
    end
else
    results.group_model = [];
end

% ====== Final group-level stats ======
param_means = mean(m_all, 2);
param_stds = std(m_all, 0, 2);
t_vals = param_means ./ (param_stds / sqrt(N));
p_vals = 2 * (1 - tcdf(abs(t_vals), N - 1));

results.subject_posteriors = subject_posteriors;
results.group_means = all_group_means;
results.final_mu_g = mu_g;
results.group_stats.t_vals = t_vals;
results.group_stats.p_vals = p_vals;
results.group_stats.param_means = param_means;
results.group_stats.param_stds = param_stds;
results.F_all_iters = F_all_iters;
results.design = design;
results.data = data;
results.f = f;
results.all_subject_means = m_all;
end