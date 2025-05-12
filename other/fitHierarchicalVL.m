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

% Initialize group-level prior
mu_g = mean(m0,1);
Sigma_g = S0;

% Store results
subject_posteriors = cell(N,1);
all_group_means = zeros(d, nIter);

mu_subjectwise = m0';
F_all_iters = zeros(N, nIter);


for iter = 1:nIter
    fprintf('\nHierarchical iteration %d/%d\n', iter, nIter);
    m_all = zeros(d,N);
    S_all = zeros(d,d,N);

    % ====== Subject-level inference ======
    parfor i = 1:N
        y_i = data{i};

        group_mu_i = mu_subjectwise(:, i);
        [m_i, V_i, D_i, logL, ~, ~, ~] = fitVariationalLaplaceThermo(y_i, f, group_mu_i, Sigma_g, maxIter, tol, 0);
        %[m_i, V_i, D_i, logL, ~, ~, ~] = fitVariationalLaplaceThermo(y_i, f, mu_g, Sigma_g, maxIter, tol, 0);
        S_i = V_i * V_i' + diag(D_i);

        % Store ELBO/log evidence for plotting and group-wise KDE
        F_all_iters(i, iter) = logL;

        m_i = denan(real(m_i));
        S_i = denan(real(S_i));

        m_all(:,i) = m_i;
        S_all(:,:,i) = S_i;
        subject_posteriors{i}.m = m_i;
        subject_posteriors{i}.S = S_i;
    end

    % ====== Group-level update (GMM shrinkage only) ======
    if iter < nIter
        
        [groupIDs, ~, groupIndex] = unique(design(:,2));  % Assume column 2 codes group
        G = numel(groupIDs);
        group_means = zeros(d, G);

        % new GMM code
        % For each group
        for g = 1:G
            idx = (groupIndex == g);
            Xg = m_all(:, idx)';  % [n_g x d]

            % Fit GMM to group
            alpha = .1;
            try
                gmm = fitgmdist(Xg, 2, 'RegularizationValue', 1e-4);
            catch
                warning('GMM fit failed for group %d — falling back to mean', g);
                group_mean = mean(Xg, 1);
                for i = find(idx)
                    mu_subjectwise(:, i) = (1 - alpha) * m_all(:, i) + alpha * group_mean';
                end
                continue;
            end

            % Assign subjects to closest mode
            P = pdf(gmm, Xg);
            [~, comps] = max(P, [], 2);

            % Update priors toward assigned component
            for j = 1:sum(idx)
                i = find(idx);
                comp = comps(j);
                mu_target = gmm.mu(comp, :)';
                mu_subjectwise(:, i(j)) = (1 - alpha) * m_all(:, i(j)) + alpha * mu_target;
            end
        end
    
        % old non-GMM code
        % % Compute mean parameter vector for each group
        % for g = 1:G
        %     idx = (groupIndex == g);
        %     group_means(:, g) = mean(m_all(:, idx), 2);
        % end
        % 
        % % Shrink each subject’s prior mean toward their group mean
        % mu_subjectwise = zeros(d, N);
        % alpha = .1;
        % for i = 1:N
        %     g = groupIndex(i);
        %     %mu_subjectwise(:, i) = (1 - alpha) * mu_subjectwise(:, i) + alpha * group_means(:, g);
        %     mu_subjectwise(:, i) = (1 - alpha) * m_all(:,i) + alpha * group_means(:, g);
        % end
    
        % Update global prior mean just for tracking/debug (not needed for model)
        mu_g = mean(mu_subjectwise, 2);
    
        % Store for diagnostics
        all_group_means(:, iter) = mu_g;
    end

    % if mod(iter, 1) == 0  % Plot every iteration (change to e.g. mod(iter,5)==0 if needed)
    %     group_col = design(:,2);
    %     group_ids = unique(group_col);
    %     colors = lines(numel(group_ids));
    % 
    %     figure(fig); clf;
    %     hold on;
    % 
    %     for g = 1:numel(group_ids)
    %         idx = group_col == group_ids(g);
    %         F_group = real(F_all_iters(idx, iter));
    %         [f, xi] = ksdensity(F_group);
    % 
    %         plot(xi, f, 'LineWidth', 2, 'DisplayName', sprintf('Group %d', group_ids(g)), ...
    %             'Color', colors(g,:));
    % 
    %         xline(mean(F_group), '--', sprintf('\\mu = %.2f', mean(F_group)), ...
    %             'Color', colors(g,:), 'LabelHorizontalAlignment', 'left');
    %     end
    % 
    %     xlabel('Free Energy (ELBO)');
    %     ylabel('Density');
    %     title(sprintf('ELBO Distributions by Group — Iteration %d', iter));
    %     legend('show');
    %     drawnow;
    %     pause(0.01);
    % end
end

if ~isempty(design)
    group_col = design(:,2);  % assume column 2 defines group membership
    group_ids = unique(group_col);

    if numel(group_ids) == 2
        idx1 = group_col == group_ids(1);
        idx2 = group_col == group_ids(2);

        m1 = m_all(:, idx1);
        m2 = m_all(:, idx2);

        n1 = sum(idx1);
        n2 = sum(idx2);

        % Means
        mean1 = mean(m1, 2);
        mean2 = mean(m2, 2);

        % Differences and pooled variance
        mean_diff = mean2 - mean1;
        pooled_std = sqrt(((std(m1, 0, 2)).^2 + (std(m2, 0, 2)).^2) / 2);
        t_vals = mean_diff ./ (pooled_std .* sqrt(2 / min(n1, n2)));
        df = n1 + n2 - 2;
        p_vals = 2 * (1 - tcdf(abs(t_vals), df));

        % Store results
        group_model.group_labels = group_ids;
        group_model.group1_idx = idx1;
        group_model.group2_idx = idx2;
        group_model.group1_mean = mean1;
        group_model.group2_mean = mean2;
        group_model.mean_diff = mean_diff;
        group_model.t_vals = t_vals;
        group_model.p_vals = p_vals;
    else
        warning('More than two unique group values detected in design(:,2); skipping contrast stats.');
        group_model = [];
    end
else
    group_model = [];
end

% ====== Final group stats ======
param_means = mean(m_all, 2);
param_stds = std(m_all, 0, 2);
t_vals = param_means ./ (param_stds / sqrt(N));
p_vals = 2 * (1 - tcdf(abs(t_vals), N - 1));

% Output
results.subject_posteriors = subject_posteriors;
results.group_means = all_group_means;
%results.group_covariances = all_group_covs;

results.group_covariances = [];  % Not used in simplified version
results.final_Sigma_g = [];      % Ditto

results.final_mu_g = mu_g;
%results.final_Sigma_g = Sigma_g;
results.group_model = group_model;
results.all_subject_means = m_all;
results.group_stats.t_vals = t_vals;
results.group_stats.p_vals = p_vals;
results.group_stats.param_means = param_means;
results.group_stats.param_stds = param_stds;
end
