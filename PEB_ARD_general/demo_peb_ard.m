function demo_peb_ard(dataset, target_idx)
% DEMO_PEB_ARD  End-to-end demo for PEB-ARD on open datasets.
% Usage:
%   demo_peb_ard            % defaults to 'boston'
%   demo_peb_ard('energy')  % Energy Efficiency (multi-output; choose target_idx)
%   demo_peb_ard('auto')    % Auto MPG
%   demo_peb_ard('wine')    % Wine Quality (red)
%
% Notes:
% - Requires peb_ard_novar.m and peb_ard_predict.m on path.
% - If you added reordering inside peb_ard_novar, plots will reflect original order.
% - For 'energy', set target_idx=1 (Heating Load) or 2 (Cooling Load).

if nargin < 1 || isempty(dataset), dataset = 'boston'; end
if nargin < 2, target_idx = 1; end

fprintf('--- PEB-ARD demo on dataset: %s ---\n', dataset);

switch lower(dataset)
    case 'boston'
        % Boston Housing (506x13 features, y=medv)
        url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv';
        T = readtable(url);
        y = T.medv;
        X = T{:, setdiff(T.Properties.VariableNames, {'medv'})};
        feat_names = T.Properties.VariableNames;
        feat_names = feat_names(~strcmp(feat_names,'medv'));
        names = ['intercept', feat_names];

    case 'energy'
        % UCI Energy Efficiency (y: Heating load, Cooling load)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx';
        T = readtable(url);
        X = T{:,1:8};
        Y = T{:,9:10};               % two outputs
        if target_idx < 1 || target_idx > 2
            warning('target_idx must be 1 or 2; defaulting to 1 (Heating Load).');
            target_idx = 1;
        end
        y = Y(:, target_idx);
        base_names = T.Properties.VariableNames(1:8);
        names = ['intercept', base_names];

    case 'auto'
        % Auto MPG (clean a bit)
        url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv';
        T = readtable(url);
        T = rmmissing(T);
        y = T.mpg;
        X = table2array(T(:, {'cylinders','displacement','horsepower','weight','acceleration'}));
        names = ['intercept', {'cyl','disp','hp','weight','accel'}];

    case 'wine'
        % UCI Wine Quality (red)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv';
        T = readtable(url, 'Delimiter',';');
        y = T.quality;
        X = T{:,1:end-1};
        names = ['intercept', T.Properties.VariableNames(1:end-1)];

    otherwise
        error('Unknown dataset "%s". Choose: boston | energy | auto | wine.', dataset);
end

% Ensure numeric, remove missing
X = double(X);
y = double(y(:));
good = all(isfinite(X),2) & isfinite(y);
X = X(good,:); y = y(good);

% Add intercept
X = [ones(size(X,1),1), X];

% Fit PEB-ARD
M = peb_ard_novar(y, X);

% Predict + uncertainty
[yhat, ysd] = peb_ard_predict(X, M);

% Metrics
R2   = 1 - sum((y - yhat).^2)/sum((y - mean(y)).^2);
RMSE = sqrt(mean((y - yhat).^2));
fprintf('R^2 = %.3f | RMSE = %.3f | n = %d | p = %d\n', R2, RMSE, size(X,1), size(X,2));

% Try to pass names in original order (if sizes match)
if numel(names) ~= size(X,2)
    names = arrayfun(@(k)sprintf('x_%d',k-1), 1:size(X,2), 'uni',0);
    names{1} = 'intercept';
end

% ---- Plots ----
% 1) Coefficient forest (original order)
if exist('peb_plot_betas','file')
    peb_plot_betas(M, names);
else
    warning('peb_plot_betas.m not found on path.');
end

% 2) Marginal densities (original order)
if exist('peb_plot_beta_densities','file')
    peb_plot_beta_densities(M, names, 1); % out_idx=1 by default
else
    warning('peb_plot_beta_densities.m not found on path.');
end

% 3) Lambda bars (original order)
if exist('peb_plot_lambda','file')
    peb_plot_lambda(M, names, true);
else
    warning('peb_plot_lambda.m not found on path.');
end

% 4) Predictive intervals (sorted by yhat for clarity) + parity plot
alpha = 1.96;
lo = yhat - alpha*ysd;
hi = yhat + alpha*ysd;
[~, idx] = sort(yhat);

figure('Name','PEB-ARD predictions','Color','w');
subplot(1,2,1); hold on; box on; grid on
xx = (1:numel(y))';
fill([xx; flipud(xx)], [lo(idx); flipud(hi(idx))], [0.87 0.87 0.87], ...
    'EdgeColor','none', 'FaceAlpha', 1);
plot(xx, yhat(idx), '-', 'LineWidth', 1.5);
plot(xx, y(idx), '.', 'MarkerSize', 10);
xlabel('samples (sorted by \hat{y})'); ylabel('y');
legend({'95% PI','\hat{y}','observed y'}, 'Location','best');
title('Predictions with 95% intervals');

subplot(1,2,2); hold on; box on; grid on
errorbar(yhat, yhat, alpha*ysd, 'LineStyle','none'); % vertical uncertainty around prediction
scatter(yhat, y, 16, 'filled');
mn = min([y; yhat]); mx = max([y; yhat]);
plot([mn mx], [mn mx], 'k--');
axis square; axis tight
xlabel('Predicted \hat{y}'); ylabel('Observed y');
title(sprintf('Observed vs Predicted (R^2=%.3f)', R2));

set(findall(gcf,'-property','FontSize'),'FontSize',12);

end
