% --- Fetch UCI Concrete dataset ---
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls';
fname = websave('concrete.xls', url);
T = readtable(fname);

% Prepare X, y
X = table2array(T(:,1:end-1));
y = T{:,end};
X = [ones(size(X,1),1) X];  % add intercept

% --- Fit PEB-ARD and predict with uncertainty ---
M = peb_ard_novar(y, X);
[yhat, ysd] = peb_ard_predict(X, M);  % ysd is posterior predictive std

% Metrics
R2   = 1 - sum((y - yhat).^2)/sum((y - mean(y)).^2);
RMSE = sqrt(mean((y - yhat).^2));
fprintf('R² = %.3f | RMSE = %.3f\n', R2, RMSE);

% 95% predictive intervals
alpha = 1.96;
lo = yhat - alpha*ysd;
hi = yhat + alpha*ysd;

% Sort by predicted mean for a clean ribbon plot
[~, idx] = sort(yhat);
ys  = y(idx);
yh  = yhat(idx);
los = lo(idx);
his = hi(idx);

names = ['intercept' T.Properties.VariableNames]

% Normal plots
peb_plot_betas(M)
peb_plot_beta_densities(M)
peb_plot_lambda(M,names(M.kept_columns))


% --- Plot 1: Sorted predictions with 95% PI ribbon + observed data ---
figure('Name','PEB-ARD Prediction with Uncertainty','Color','w'); 
subplot(1,2,1); hold on;

% Shaded predictive interval
x = (1:numel(y))';
fill([x; flipud(x)], [los; flipud(his)], [0.85 0.85 0.85], ...
    'EdgeColor','none', 'FaceAlpha', 1.0);

% Predicted mean
plot(x, yh, '-', 'LineWidth', 1.5);

% Observed points
plot(x, ys, '.', 'MarkerSize', 10);

grid on; box on;
xlabel('Samples (sorted by \hat{y})');
ylabel('Concrete compressive strength');
title('Predictions with 95% predictive intervals');
legend({'95% PI','\hat{y}','Observed y'}, 'Location','best');

% --- Plot 2: Parity (Obs vs Pred) with predictive error bars ---
subplot(1,2,2); hold on; box on; grid on;

% Vertical uncertainty bars centered at \hat{y}
errorbar(yhat, yhat, alpha*ysd, 'LineStyle','none');

% Scatter of observed vs predicted
scatter(yhat, y, 16, 'filled');

% Identity line
mn = min([y; yhat]); mx = max([y; yhat]);
plot([mn mx], [mn mx], 'k--');

xlabel('Predicted \hat{y}');
ylabel('Observed y');
title('Observed vs Predicted (with predictive uncertainty)');
axis tight; axis square;

% Optional: residual histogram
figure('Name','Residuals','Color','w');
res = y - yhat;
subplot(1,2,1); histogram(res, 30); grid on; box on;
xlabel('Residual'); ylabel('Count'); title('Residual distribution');

subplot(1,2,2); plot(res, '.-'); yline(0,'k:'); grid on; box on;
xlabel('Sample'); ylabel('Residual'); title('Residuals over samples');
