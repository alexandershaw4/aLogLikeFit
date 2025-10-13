function peb_plot_lambda(M, names, use_log10)
% Plot ARD precisions (lambda) in ORIGINAL INPUT ORDER.
%   peb_plot_lambda(M)
%   peb_plot_lambda(M, names)
%   peb_plot_lambda(M, names, use_log10)  % default: true (plot log10(lambda))
%
% Notes:
% - Prefers *_ordered fields (lambda_ordered, x_std_ordered) if present.
% - Dropped columns (from QR pruning) are shown as empty bars and labelled "(dropped)".
% - Larger lambda => stronger shrinkage (less relevant feature).

    if nargin < 3 || isempty(use_log10), use_log10 = true; end

    % ----- choose lambda in original order if available -----
    if isfield(M,'lambda_ordered') && ~isempty(M.lambda_ordered)
        lam = M.lambda_ordered(:);
    else
        lam = M.lambda(:);
    end
    p = numel(lam);

    % ----- build names (original order) -----
    if nargin < 2 || isempty(names)
        names = arrayfun(@(k)sprintf('x_%d',k-1), 1:p, 'uni', 0);
        if p >= 1, names{1} = 'intercept'; end
    else
        if numel(names) ~= p
            warning('peb_plot_lambda: names length (%d) != #coeffs (%d). Ignoring provided names.', numel(names), p);
            names = arrayfun(@(k)sprintf('x_%d',k-1), 1:p, 'uni', 0);
            if p >= 1, names{1} = 'intercept'; end
        end
    end

    % ----- decide dropped mask -----
    % Prefer explicit mask if provided by your reorder helper
    if isfield(M,'original_mask') && ~isempty(M.original_mask) && numel(M.original_mask)==p
        dropped = ~M.original_mask(:);
    else
        % fallback: treat NaN lambdas as dropped
        dropped = ~isfinite(lam);
    end

    % If some lambdas are finite but beta was NaN, consider them dropped for plotting
    if isfield(M,'beta_ordered') && numel(M.beta_ordered)==p
        dropped = dropped | ~isfinite(M.beta_ordered(:));
    end

    % ----- transform for display -----
    yvals = lam;
    ylab  = '\lambda (precision)';
    if use_log10
        yvals = log10(lam);
        ylab  = 'log_{10}(\lambda) (larger = more shrinkage)';
    end

    % For dropped → NaN so bar is omitted
    yvals(dropped) = NaN;

    % Add "(dropped)" suffix to labels where applicable
    names_shown = names;
    names_shown(dropped) = cellfun(@(s)[s ' (dropped)'], names_shown(dropped), 'uni', 0);

    % ----- plot -----
    figure('Name','PEB-ARD lambda (ARD precisions)','Color','w'); clf; hold on
    b = bar(yvals, 'FaceAlpha', 0.9);
    grid on; box on

    % A little styling for readability
    set(gca, 'XTick', 1:p, 'XTickLabel', names_shown, 'XTickLabelRotation', 45);
    xlim([0.5, p+0.5]);
    ylabel(ylab);
    title('ARD precisions by feature (original input order)');

    % Optional: zero line for log plot
    if use_log10
        yline(0,'k:','HandleVisibility','off');
    end

    % Make dropped bars light grey placeholders (optional)
    % If you prefer visible stubs instead of gaps:
    % hold on
    % stub = zeros(p,1); stub(~dropped) = NaN;        % only draw for dropped
    % bar(stub, 'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none');

    set(findall(gcf,'-property','FontSize'), 'FontSize', 12);
end
