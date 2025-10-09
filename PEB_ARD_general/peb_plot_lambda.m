function peb_plot_lambda(M, names)
    if nargin<2 || isempty(names)
        names = arrayfun(@(k)sprintf('x_%d',k-1), 1:size(M.beta,1), 'uni',0);
        names{1}='intercept';
    end
    figure; clf
    bar(log10(M.lambda)); grid on
    set(gca,'XTick',1:numel(M.lambda),'XTickLabel',names,'XTickLabelRotation',45)
    ylabel('log_{10}(\lambda)'); title('ARD precision (larger = more shrinkage)')
end

%peb_plot_lambda(M, names);
