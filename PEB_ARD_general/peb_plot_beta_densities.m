function peb_plot_beta_densities(M, names, out_idx)
% Plot marginal posterior densities for each coefficient (ORIGINAL ORDER).
% Uses Normal approximation N(mu_j, se_j^2) on the ORIGINAL scale.
%
% peb_plot_beta_densities(M)
% peb_plot_beta_densities(M, names)
% peb_plot_beta_densities(M, names, out_idx)   % for multi-output (column of beta)
%
% Notes:
% - Prefers *_ordered fields if present (beta_ordered, Vbeta_ordered, x_std_ordered).
% - Dropped/undefined coefficients appear as "dropped" panels.
% - No Statistics Toolbox required (manual Gaussian pdf).

    if nargin < 3 || isempty(out_idx), out_idx = 1; end

    % ---------- choose beta (original order if available) ----------
    if isfield(M,'beta_ordered') && ~isempty(M.beta_ordered)
        beta_all = M.beta_ordered;   % p x d (often d=1)
    else
        beta_all = M.beta;           % p x d
    end
    if isvector(beta_all), beta_all = beta_all(:); end
    p = size(beta_all,1);

    % pick output column
    if size(beta_all,2) < out_idx
        error('peb_plot_beta_densities: requested output %d but beta has only %d columns.', out_idx, size(beta_all,2));
    end
    beta = beta_all(:,out_idx);

    % ---------- names in original order ----------
    if nargin < 2 || isempty(names)
        names = arrayfun(@(k)sprintf('x_%d',k-1), 1:p, 'uni', 0);
        if p>=1, names{1} = 'intercept'; end
    else
        if numel(names) ~= p
            warning('peb_plot_beta_densities: names length (%d) != #coeffs (%d). Ignoring provided names.', numel(names), p);
            names = arrayfun(@(k)sprintf('x_%d',k-1), 1:p, 'uni', 0);
            if p>=1, names{1} = 'intercept'; end
        end
    end

    % ---------- covariance diagonal (std space), aligned to original order ----------
    if isfield(M,'Vbeta_ordered') && ~isempty(M.Vbeta_ordered) && size(M.Vbeta_ordered,1) == p
        Vdiag = diag(M.Vbeta_ordered);
    elseif isfield(M,'Vbeta') && ~isempty(M.Vbeta) && size(M.Vbeta,1) == p
        Vdiag = diag(M.Vbeta);
    else
        % fallback: no aligned covariance available
        Vdiag = nan(p,1);
        warning('peb_plot_beta_densities: covariance not aligned to original order; SEs set to NaN where unknown.');
    end

    % ---------- x_std (for scaling to original units) ----------
    if isfield(M,'x_std_ordered') && ~isempty(M.x_std_ordered) && numel(M.x_std_ordered) == p
        x_std = M.x_std_ordered(:);
    elseif isfield(M,'x_std') && numel(M.x_std) == p
        x_std = M.x_std(:);
    else
        x_std = ones(p,1);
        warning('peb_plot_beta_densities: x_std not aligned; using ones as fallback.');
    end

    % ---------- y_std (assume independent per output; default 1 if missing) ----------
    if isfield(M,'y_std') && numel(M.y_std) >= out_idx && ~isempty(M.y_std(out_idx))
        y_std = M.y_std(out_idx);
    else
        y_std = 1;
    end

    % ---------- SE on original scale ----------
    se_std = sqrt(max(Vdiag, 0));        % std-space
    scale  = y_std ./ x_std;             % to original units
    se     = se_std .* scale;            % original units

    % mask dropped
    dropped = ~isfinite(beta) | ~isfinite(se) | (se<=0);

    % ---------- layout ----------
    cols = ceil(sqrt(p));
    rows = ceil(p / cols);
    figure('Name','PEB-ARD posterior beta densities','Color','w');
    for j = 1:p
        subplot(rows, cols, j); hold on; box on; grid on

        if dropped(j)
            % show an empty panel with label
            title(sprintf('%s (dropped)', names{j}), 'Interpreter','none');
            axis off
            continue
        end

        mu = beta(j);
        sj = se(j);

        % x-range: mu ± 4*se, fallback if extremely small
        if ~isfinite(sj) || sj <= 0
            x = linspace(mu-1, mu+1, 200);
        else
            x = linspace(mu - 4*sj, mu + 4*sj, 240);
        end

        % manual Normal pdf (no toolbox)
        pdf = (1./(sqrt(2*pi)*sj)) .* exp(-0.5*((x-mu)/sj).^2);

        plot(x, pdf, 'LineWidth', 1.5);
        yline(0,'k:');
        title(names{j}, 'Interpreter','none');
        xlabel('\beta'); ylabel('density');

        % mark mean ± 1.96*se
        lo = mu - 1.96*sj; hi = mu + 1.96*sj;
        plot([mu mu], [0 max(pdf)], 'k--', 'HandleVisibility','off');
        plot([lo lo], [0 max(pdf)], ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility','off');
        plot([hi hi], [0 max(pdf)], ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility','off');
    end

    % consistent fonts
    set(findall(gcf,'-property','FontSize'), 'FontSize', 16);
end
