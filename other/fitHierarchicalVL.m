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
d = length(m0);

% Initialize group-level prior
mu_g = m0;
Sigma_g = S0;

% Store results
subject_posteriors = cell(N,1);
all_group_means = zeros(d, nIter);
all_group_covs = cell(nIter, 1);

% Stability parameters
alpha = 0.1;                % Damping factor
jitter = 1e-6;              % Diagonal jitter
min_var = 1e-4;             % Minimum allowed variance (variance floor)
max_precision = 1e6;        % Max precision cap (optional)

mu_subjectwise = mu_g;

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

        m_i = denan(real(m_i));
        S_i = denan(real(S_i));

        m_all(:,i) = m_i;
        S_all(:,:,i) = S_i;
        subject_posteriors{i}.m = m_i;
        subject_posteriors{i}.S = S_i;
    end

    if iter < nIter % skip on last run
        % ====== Empirical Bayes Group-level update ======
        %mu_empirical = mean(m_all, 2);
        if ~isempty(design)
            X = design;
            X = X - mean(X, 1); % mean-centering
    
            % Ridge regression for each parameter (d parameters)
            beta = (X' * X + 1e-6 * eye(size(X,2))) \ (X' * m_all');
    
            % Reconstruct subject-specific fitted values
            mu_empirical = (X * beta)';  % [d x N]
            mu_empirical_mean = mean(mu_empirical, 2);  % for group-centering
        else
            mu_empirical = repmat(mean(m_all, 2), 1, N);
            mu_empirical_mean = mean(m_all, 2);
        end
        
        Sigma_empirical = zeros(d);
        for i = 1:N
           m_i = m_all(:,i);
           S_i = S_all(:,:,i);
           Sigma_empirical = Sigma_empirical + S_i + (m_i - mu_empirical) * (m_i - mu_empirical)';
        end
    
        %Sigma_empirical = zeros(d);
        %for i = 1:N
        %    m_i = m_all(:,i);
        %    mu_i = mu_empirical(:,i);
        %    S_i = S_all(:,:,i);
        %    Sigma_empirical = Sigma_empirical + S_i + (m_i - mu_i) * (m_i - mu_i)';
        %end
    
        Sigma_empirical = Sigma_empirical / N;
    
        % Damp only the mean
        %mu_g = (1 - alpha) * mu_g + alpha * mu_empirical;
        %mu_g = (1 - alpha) * mu_g + alpha * mu_empirical_mean;
    
        % new - Find unique group IDs from design matrix
        [groupIDs, ~, groupIndex] = unique(design(:,2));  % Assume column 2 codes group
        G = numel(groupIDs);
        group_means = zeros(d, G);
    
        % Compute mean parameter vector for each group
        for g = 1:G
            idx = (groupIndex == g);
            group_means(:, g) = mean(m_all(:, idx), 2);
        end
    
        % Assign group-specific priors to subjects (optional: use this in next iteration)
        mu_subjectwise = zeros(d, N);
        for i = 1:N
            g = groupIndex(i);
            mu_subjectwise(:, i) = group_means(:, g);
        end
    
        % Damped update: move global prior slightly toward these means
        mu_g = (1 - alpha) * mu_g + alpha * mean(mu_subjectwise, 2);
    
    
        % Regularize with shrinkage toward prior
        lambda = 0.1;  % 0 = full data-driven, 1 = stay near prior
        Sigma_g = (1 - lambda) * Sigma_empirical + lambda * S0;
    
        % Ensure numerical stability (non-cumulative)
        [~, p] = chol(Sigma_g);
        if p > 0
            % If not positive definite, add jitter once
            Sigma_g = Sigma_g + jitter * eye(d);
        end
    
        % print update
        fprintf('\nHierarchical iteration -- group update %d/%d\n', iter, nIter);
        fprintf('  Min var: %.2e | Max var: %.2e | cond(Sigma_g): %.2e\n', ...
            min(diag(Sigma_g)), max(diag(Sigma_g)), cond(Sigma_g));
    
        % Optional: cap precision values (commented out; can enable if needed)
        % P = inv(Sigma_g);
        % P = min(P, max_precision);
        % Sigma_g = inv(P);
    
        % Store
        all_group_means(:,iter) = mu_g;
        all_group_covs{iter} = Sigma_g;
    end
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
