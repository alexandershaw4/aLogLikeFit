function plotHistAndDensity(BICvals)

num_models = size(BICvals, 2);
% Generate model labels (optional)
model_labels = arrayfun(@(x) sprintf('Model %d', x), 1:num_models, 'UniformOutput', false);

% Dynamically determine subplot layout
nCols = ceil(sqrt(num_models));
nRows = ceil(num_models / nCols);

% Plotting
figure;
for m = 1:num_models
    subplot(nRows, nCols, m);  % Dynamic layout
    histogram(BICvals(:, m), 'Normalization', 'pdf', 'FaceAlpha', 0.4, 'EdgeColor', 'none');
    hold on;

    % Overlay KDE (Kernel Density Estimate)
    [f, xi] = ksdensity(BICvals(:, m));
    plot(xi, f, 'LineWidth', 1.5);
    
    title(model_labels{m}, 'FontSize', 10);
    %xlabel('BIC approx'); 
    ylabel('Density');
    xlim([min(BICvals(:)) max(BICvals(:))]);  % Uniform x-axis across plots
end

%sgtitle('BIC Distributions per Model');

end
