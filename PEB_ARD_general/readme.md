# PEB_ARD_general

This directory contains a **general-purpose implementation** of Parametric Empirical Bayes with Automatic Relevance Determination (PEB-ARD) for linear regression (single- or multi-output). The routines are intended for use in modular pipelines, diagnostics, and predictive modeling where uncertainty quantification is desirable.


---

## 🚀 Quick Start

```matlab
% Download or place this folder on your MATLAB path
addpath('path/to/PEB_ARD_general');

% Example with a real dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls';
fname = websave('concrete.xls', url);
T = readtable(fname);
X = table2array(T(:,1:end-1));
y = T{:, end};
X = [ones(size(X,1),1), X];  % add intercept

% Fit PEB-ARD
M = peb_ard_novar(y, X);

% Predict and get uncertainty
[yhat, ysd] = peb_ard_predict(X, M);

% Visualise
peb_plot_betas(M)
peb_plot_beta_densities(M)
peb_plot_lambda(M)

% Example density ribbon for one coefficient (say X2):
Bsamp = peb_sample_beta(M, 5000);
figure; 
for i = 1:size(Bsamp,2)-1
    subplot(1,size(Bsamp,2)-1,i)
    histogram(Bsamp(:,i+1), 60, 'Normalization','pdf'); grid on
    title(sprintf('Posterior samples for beta_{X%d}',i))
    xlabel('\beta'); ylabel('density')
end
```
