function peb_plot_betas(M, names)
% Forest plot of β with 95% credible intervals (original scale).
    if nargin<2 || isempty(names)
        names = arrayfun(@(k)sprintf('x_%d',k-1), 1:size(M.beta,1), 'uni',0);
        names{1} = 'intercept';
    end
    mu  = M.beta(:);
    % SE in original scale: transform diag(Vbeta) from std space
    se_std = sqrt(max(diag(M.Vbeta),1e-16));              % p_kept×1
    % map std-SE to original scale per output (assume d=1 here)
    scale = (M.y_std(1) ./ M.x_std(:));                   % p_kept×1
    se  = se_std .* scale;

    ci95 = 1.96*se;
    [~,ord] = sort(abs(mu),'descend');
    mu  = mu(ord); ci95 = ci95(ord); names = names(ord);

    figure; clf; hold on
    y = 1:numel(mu);
    errorbar(mu, y, ci95, 'horizontal', 'o', 'LineWidth',1.5);
    xline(0,'k:'); set(gca,'YDir','reverse','YTick',y,'YTickLabel',names);
    xlabel('\beta (posterior mean ± 95% CI)'); ylabel('feature');
    title('Posterior coefficients'); grid on
end

% % Example:
% names = {'intercept','X1','X2','X3','X4','X5'};
% peb_plot_betas(M, names);
