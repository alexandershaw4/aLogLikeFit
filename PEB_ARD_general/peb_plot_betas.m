function peb_plot_betas(M, names)
% PEB forest plot: posterior mean ± 95% CI (ORIGINAL INPUT ORDER).
% Assumes single-output (d=1). If your M includes *_ordered fields (beta_ordered,
% Vbeta_ordered, x_std_ordered), those will be used. Otherwise it falls back.

    % --- Choose beta in original order if available ---
    if isfield(M,'beta_ordered') && ~isempty(M.beta_ordered)
        beta = M.beta_ordered(:);
    else
        beta = M.beta(:);
    end
    p = numel(beta);

    % --- Build names (original order) ---
    if nargin < 2 || isempty(names)
        names = arrayfun(@(k)sprintf('x_%d',k-1), 1:p, 'uni', 0);
        if p >= 1, names{1} = 'intercept'; end
    else
        if numel(names) ~= p
            warning('peb_plot_betas: names length (%d) != #coeffs (%d). Ignoring provided names.', numel(names), p);
            names = arrayfun(@(k)sprintf('x_%d',k-1), 1:p, 'uni', 0);
            if p >= 1, names{1} = 'intercept'; end
        end
    end

    % --- Get covariance diagonal (std space) matching original order ---
    Vdiag = [];
    if isfield(M,'Vbeta_ordered') && ~isempty(M.Vbeta_ordered) ...
            && size(M.Vbeta_ordered,1) == p
        Vdiag = diag(M.Vbeta_ordered);
    elseif isfield(M,'Vbeta') && ~isempty(M.Vbeta) ...
            && size(M.Vbeta,1) == p
        Vdiag = diag(M.Vbeta);
    elseif isfield(M,'Vbeta')
        % We have a kept-order Vbeta but beta may be original (with NaNs). We'll just
        % produce NaN SEs to avoid mismatched indexing.
        Vdiag = nan(p,1);
        warning('peb_plot_betas: covariance not aligned to original order; SEs set to NaN for dropped/unknown columns.');
    else
        error('peb_plot_betas: No covariance matrix found (Vbeta or Vbeta_ordered).');
    end

    % --- x_std in original order (for scaling SEs) ---
    if isfield(M,'x_std_ordered') && ~isempty(M.x_std_ordered) ...
            && numel(M.x_std_ordered) == p
        x_std = M.x_std_ordered(:);
    elseif isfield(M,'x_std') && numel(M.x_std) == p
        x_std = M.x_std(:);
    else
        x_std = ones(p,1);  % fallback (no scaling)
        warning('peb_plot_betas: x_std not aligned; using ones as fallback.');
    end

    % --- y_std (assume single output) ---
    if isfield(M,'y_std') && ~isempty(M.y_std)
        y_std = M.y_std(1);
    else
        y_std = 1;
    end

    % --- SE on original scale ---
    se_std = sqrt(max(Vdiag, 0));            % p×1 in std space (may include NaNs)
    scale  = y_std ./ x_std;                  % p×1 elementwise
    se     = se_std .* scale;                 % p×1 in original units

    % Make sure dropped columns stay NaN consistently
    dropped = ~isfinite(beta);
    se(dropped) = NaN;

    % 95% CI
    ci95 = 1.96 * se;

    % --- Plot in ORIGINAL order (no sorting) ---
    figure; clf; hold on
    yy = 1:p;
    errorbar(beta, yy, ci95, 'horizontal', 'o', 'LineWidth', 1.5);
    xline(0,'k:');

    set(gca, 'YDir','reverse', 'YTick', yy, 'YTickLabel', names);
    xlabel('\beta (posterior mean \pm 95% CI)');
    ylabel('feature (original order)');
    title('Posterior coefficients');
    grid on; box on

    % make space around edges and use readable font
    ylim([0.5, p+0.5]);
    set(findall(gcf,'-property','FontSize'), 'FontSize', 16);
end
