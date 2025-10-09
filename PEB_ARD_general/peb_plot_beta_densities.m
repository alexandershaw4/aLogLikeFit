function peb_plot_beta_densities(M, names)
    if nargin<2 || isempty(names)
        names = arrayfun(@(k)sprintf('x_%d',k-1), 1:size(M.beta,1), 'uni',0);
        names{1}='intercept';
    end
    p = size(M.beta,1);
    se_std = sqrt(max(diag(M.Vbeta),1e-16));
    scale  = (M.y_std(1) ./ M.x_std(:));
    mu  = M.beta(:);
    se  = se_std .* scale;

    cols = ceil(sqrt(p)); rows = ceil(p/cols);
    figure; clf
    for j=1:p
        subplot(rows,cols,j)
        x = linspace(mu(j)-4*se(j), mu(j)+4*se(j), 200);
        plot(x, normpdf(x, mu(j), se(j)),'LineWidth',1.5); grid on
        title(names{j}); xlabel('\beta'); ylabel('density');
    end
    sgtitle('Marginal posterior (Normal approx)')
end

% Example:
% peb_plot_beta_densities(M, names);
