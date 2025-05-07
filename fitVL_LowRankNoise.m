function [m, V_post, D_post, logL, iter, Sigma_struct, allm] = fitVL_LowRankNoise(y, f, m0, S0, maxIter, tol, plots)
% Extended Variational Laplace with Low-Rank Observation Covariance
% ================================================================
% 
% This routine extends the basic fitVariationalLaplaceThermo approach
% by incorporating a **low-rank model of heteroscedastic observation noise** 
% (i.e., structured and adaptive noise covariance across observations).
%
% The method fits dynamical systems models of the form:
%          y = f(m) + e
% where e ~ N(0, Sigma), with Sigma estimated adaptively during inference.
%
% It estimates a **Gaussian variational posterior** over the latent 
% variables, refining its mean and covariance structure by optimizing a 
% variational free-energy bound (the ELBO).
%
% The posterior covariance is approximated using a **low-rank plus diagonal structure**:
%          S ≈ V * Vᵀ + D
% where:
% - V captures **low-dimensional correlations** between parameters,
% - D captures **individual parameter variances** (diagonal noise).
%
% The **observation noise covariance** Sigma is also modeled as:
%          Sigma ≈ U * Uᵀ + diag(D_noise)
% and updated during inference based on model residuals.
%
%
% Key Features
% ------------
% - **Low-Rank Posterior Approximation**:
%   Efficient low-rank decomposition of the posterior precision matrix,
%   avoiding full covariance estimation in high dimensions.
%
% - **Smarter Observation Variance Updates**:
%   The observation noise is heteroscedastic and low-rank correlated, 
%   with both the diagonal variances and low-rank factors adapting to the model residuals.
%
% - **Thermodynamic Integration**:
%   Annealed free energy (ELBO) estimates support thermodynamic evidence computations.
%
% - **Smooth Structured Noise Modeling**:
%   Observation noise is regularized by smoothing residuals via radial basis functions
%   (radialPD), capturing local structure rather than treating noise as purely random.
%
%
% Usage
% -----
% [m, V, D, logL, iter, Sigma_struct, allm] = fitVL_LowRankNoise(y, f, m0, S0, maxIter, tol, plots)
%
%
% Inputs
% ------
%   y         : Observed data vector (n x 1)
%   f         : Model function handle (f(m) returns predicted data)
%   m0        : Initial mean of the variational distribution (d x 1)
%   S0        : Initial covariance of the variational distribution (d x d)
%   maxIter   : Maximum number of optimization iterations
%   tol       : Convergence tolerance on the mean update ||dm||
%   plots     : Whether to produce diagnostic plots (1 = yes, 0 = no)
%
%
% Outputs
% -------
%   m          : Optimized mean of the variational posterior
%   V, D       : Low-rank decomposition of the posterior precision matrix 
%                (approximate S ≈ V * Vᵀ + D)
%   logL       : Final value of the variational free energy (ELBO)
%   iter       : Number of iterations performed
%   Sigma_struct : Struct containing the final low-rank observation noise factors (U, D_noise)
%   allm       : Evolution of the mean estimate across iterations
%
%
% Notes
% -----
% - The final posterior covariance can be recovered approximately via the 
%   Woodbury formula if needed:
%       Sigma_q ≈ inv(D) - inv(D) * V * inv(I + Vᵀ * inv(D) * V) * Vᵀ * inv(D)
%
% - although you'll get pretty close with:
%       Sigma_q = inv( V*V' + D)
%
% - The evolving observation noise structure (Sigma) can capture 
%   time-varying or locally correlated observation noise, improving robustness
%   over simple homoscedastic Gaussian noise models.
%
% - The smoother (radialPD) regularizes the noise model during learning, ensuring
%   stability and meaningful low-rank decompositions even with noisy residuals.
%
%
% Developed by AS, 2025


if nargin < 7 || isempty(plots)
    plots = 1;
end

n = length(y);
m = m0(:);
k_post = min(length(m), 20);

% Prior decomposition (for q(z))
[U, Sval, ~] = svd(full(S0), 'econ');
eigvals = diag(Sval);
k_prior = sum(eigvals > 0.01 * max(eigvals));
k_prior = max(k_prior, 10);
V_post = U(:, 1:k_prior) * diag(sqrt(eigvals(1:k_prior)));
D_post = diag(diag(S0) - sum(V_post.^2, 2));

% Initial observation noise structure: Sigma = U_noise*U_noise^T + D_noise
k_noise = min(n, 10);
%U_noise = randn(n, k_noise) * 0.1;
R_init = y - f(m0);
%[U_init, ~, ~] = svd(R_init * ones(1, k_noise), 'econ');
k    = radialPD(R_init,2);
kern = k*diag(R_init)*k'; 
[U_init, ~, ~] = svd(kern);
U_noise = U_init(:, 1:k_noise) * 0.1;

D_noise = 0.1 * ones(n,1);
epsilon = 1e-6;
beta = 1e-3;

allm = m(:);
all_elbo = [];
allentropy = [];
allloglike = [];
alllogprior = [];
all_D_noise = D_noise(:)';

if plots
    fw = figure('position',[570,659,1740,849]);
end

for iter = 1:maxIter
    y_pred = f(m);
    residuals = y - y_pred;

    % % Construct observation noise covariance: Sigma = UU^T + D
    % Sigma = U_noise * U_noise' + diag(D_noise + epsilon);
    % [L_noise, p_noise] = chol(Sigma, 'lower');
    % if p_noise > 0
    %     fprintf('Sigma not PD at iter %d. Adding jitter...\n', iter);
    %     L_noise = chol(Sigma + 1e-4*eye(n), 'lower');
    % end
    % invSigma = L_noise' \ (L_noise \ eye(n));

    % Sometimes Sigma is ill conditioned so putting in a fallback for when
    % cholesky never works...
    
    % Construct observation noise covariance: Sigma = UU^T + D
    Sigma = U_noise * U_noise' + diag(D_noise + epsilon);

    % Robust Cholesky with jitter
    maxTries = 5;
    jitter = 1e-6;
    success = false;
    for t = 1:maxTries
        try
            L_noise = chol(Sigma + jitter * eye(n), 'lower');
            success = true;
            break;
        catch
            jitter = jitter * 10;
        end
    end

    if success
        invSigma = L_noise' \ (L_noise \ eye(n));
    else
        fprintf('Cholesky failed after %d attempts. Falling back to Woodbury inversion...\n', maxTries);

        % Woodbury-based inversion: Sigma = UU^T + D => inv(Sigma)
        Dinv = diag(1 ./ (D_noise + epsilon));
        A = U_noise' * Dinv * U_noise;
        B = (eye(k_noise) + A) \ (U_noise' * Dinv);
        invSigma = Dinv - Dinv * U_noise * B;
    end



    % Log-likelihood
    logL_likelihood = -0.5 * (residuals' * invSigma * residuals + log(det(Sigma)) + n*log(2*pi));

    % Jacobian
    J = computeJacobian(f, m, n);

    % Posterior precision
    H = J' * invSigma * J;
    H_prior = inv(S0 + computeSmoothCovariance(m, 2));
    H_post = H + H_prior;
    g_post = J' * invSigma * residuals - H_prior * (m - m0);

    % Posterior covariance (low-rank update)
    [U_h, S_h, ~] = svd(H_post, 'econ');
    V_post = U_h(:,1:k_post) * sqrt(S_h(1:k_post,1:k_post));
    D_post = diag(diag(H_post) - sum(V_post.^2, 2));

    % Mean update via preconditioned CG
    try
        L = chol(H_post, 'lower');
        dm = L' \ (L \ g_post);
    catch
        [dm, ~] = pcg(H_post + eye(size(H_post))*1e-6, g_post, 1e-6, 100);
    end
    m = m + dm;
    allm = [allm m(:)];

    % ELBO
    logL_entropy = 0.5 * sum(log(diag(D_post) + 1e-6));
    logL_prior = -0.5 * ((m - m0)' * H_prior * (m - m0));
    logL = logL_likelihood + logL_prior + logL_entropy;
    all_elbo(end+1) = logL;
    allentropy = [allentropy logL_entropy];
    allloglike = [allloglike logL_likelihood];
    alllogprior = [alllogprior logL_prior];
    all_D_noise = [all_D_noise; D_noise(:)'];

    % Update observation noise structure using factor analysis-style update
    E2 = residuals.^2;
    D_noise = 0.9 * D_noise + 0.1 * max(E2 - sum(U_noise.^2,2), epsilon);
    k = radialPD(residuals,2);
    kern = k*diag(residuals)*k';
    [U_new, ~, ~] = svd(kern);

    %[U_new, ~, ~] = svd(residuals * ones(1, k_noise), 'econ');
    U_noise = 0.9 * U_noise + 0.1 * U_new(:, 1:k_noise);
    %U_noise = 0.9 * U_noise + 0.1 * (residuals * randn(1, k_noise));

    if plots
        figure(fw); clf;
        t = tiledlayout(3,4, 'TileSpacing', 'compact', 'Padding', 'compact');

        nexttile([1 4]);
        hold on;
        errorbar(1:n, y, sqrt(D_noise), 'k.', 'DisplayName', 'Observed ±σ', 'CapSize', 0);
        plot(1:n, y, 'k', 'LineWidth', 1);
        plot(1:n, y_pred, '-', 'Color', [0.8 0 0], 'LineWidth', 2);
        plot(1:n, sqrt(D_noise), '-', 'Color', [0.1 0.6 0.1], 'LineWidth', 1.5);
        hold off;
        title('Model Fit with Low-Rank Covariance');
        legend('Location','best');
        grid on; box on;

        lineColor = [1 0.7 0.7];
        scatterColor = 'k';
        scatterSize = 30;
        lineWidth = 2;

        nexttile;
        plot(1:iter, allentropy, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
        scatter(1:iter, allentropy, scatterSize, scatterColor, 'filled'); hold off;
        title('Entropy'); grid on; box on;

        nexttile;
        plot(1:iter, allloglike, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
        scatter(1:iter, allloglike, scatterSize, scatterColor, 'filled'); hold off;
        title('Log-Likelihood'); grid on; box on;

        nexttile;
        plot(1:iter, alllogprior, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
        scatter(1:iter, alllogprior, scatterSize, scatterColor, 'filled'); hold off;
        title('Log-Prior'); grid on; box on;

        nexttile;
        plot(1:iter, all_elbo, 'Color', lineColor, 'LineWidth', lineWidth); hold on;
        scatter(1:iter, all_elbo, scatterSize, scatterColor, 'filled'); hold off;
        title('ELBO'); grid on; box on;

        nexttile([1 2]);
        imagesc(all_D_noise');
        title('Evolving Diagonal Noise (D)');
        xlabel('Iteration'); ylabel('Data Index');
        colorbar; axis tight; colormap(turbo);

        nexttile([1 2]);
        imagesc(U_noise);
        title('Current Low-Rank Noise Factors (U)');
        xlabel('Component'); ylabel('Data Index');
        colorbar; axis tight; colormap(parula);

        set(gcf, 'Color', 'w');
        set(findall(gcf,'-property','FontSize'),'FontSize',18);
        drawnow;
    end

    if norm(dm) < tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end

    fprintf('Iter %d | ELBO %.4f | ||dm|| %.4f\n', iter, logL, norm(dm));
end

Sigma_struct.U = U_noise;
Sigma_struct.D = D_noise;

end

function K = computeSmoothCovariance(x, lengthScale)
    n = length(x);
    K = exp(-pdist2(x(:), x(:)).^2 / (2 * lengthScale^2));
    K = K + 1e-6 * eye(n);
end

function J = computeJacobian(f, x, m)
    epsilon = 1e-6;
    n = length(x);
    J = zeros(m, n);
    parfor i = 1:n
        x_step = x;
        x_stepb = x;
        x_step(i) = x_step(i) + epsilon;
        x_stepb(i) = x_stepb(i) - epsilon;
        J(:, i) = (f(x_step) - f(x_stepb)) / (2 * epsilon);
    end
end