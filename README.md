# aLogLikeFit

A collection of flexible routines for **first-level and second-level modeling** in computational psychiatry, especially for fitting **spectral DCMs** to neurophysiology data.

This toolkit includes:
- **Variational Laplace algorithms** with heteroscedastic noise and thermodynamic integration
- **Low-rank observation noise modeling**
- **Parametric Empirical Bayes (PEB)** routines for group-level (second-level) Bayesian inference

---

## \ud83d\udce6 Main Components

### 1. Variational Laplace Fitters
- **`fitVariationalLaplaceThermo.m`**  
  Classical Variational Laplace with:
  - Smarter heteroscedastic variance updates
  - Thermodynamic integration for log-evidence estimation
  - Optimizes the ELBO (evidence lower bound)

- **`fitVL_LowRankNoise.m`**  
  Extended version with:
  - Low-rank plus diagonal modeling of **observation noise covariance**  
  - Dynamic adaptation of noise structure based on residuals

### 2. Parametric Empirical Bayes (PEB)
- **`peb_ard_with_stats_var.m`**, **`peb_ard_with_stats.m`**, **`peb_ard_with_stats_LM.m`**
  - Implements PEB with:
    - Ridge-like Bayesian regularization
    - **Automatic Relevance Determination (ARD)** for feature selection
    - (Optional) **Levenberg-Marquardt (LM)** enhanced optimization for stability
  - Returns:
    - Group-level parameter estimates
    - ARD hyperparameters
    - t-statistics and p-values
    - Individual posteriors

---

## \ud83d\ude80 Quickstart Guide

### Step 1: Fit a First-Level (Individual) Model

Fit a spectral DCM-like model, with fields:  
`DCM.M.pE`, `DCM.M.pC`, `DCM.M.f`, `DCM.M.IS`, `DCM.xY`

```matlab
M = aFitDCM(DCM);        % Construct fitting object
M.aloglikVLtherm;        % Run Variational Laplace with thermodynamic integration
```

Optionally refine:
```matlab
M.update_parameters(M.Ep); 
M.aloglikVLtherm;
```

Inspect posteriors:
```matlab
M.Ep   % Posterior means
M.CP   % Posterior covariances
M.F    % Free energy (model evidence)
```

<div align="center">
  <img src="https://github.com/alexandershaw4/aLogLikeFit/assets/your-image1.png" width="600">
</div>

---

### Step 2: Group-Level Inference via PEB

Extract the individual posteriors, and apply PEB:

```matlab
% Example usage:
[beta, lambda_vals, t_stats, p_values, posterior_means, posterior_covs] = peb_ard_with_stats_var(PosteriorMeans, PosteriorCovariances,X,num_iter);
```

- `G`: Group mean parameter estimates
- `H`: ARD hyperparameters
- `stats`: t-stats, p-values for group parameters

<div align="center">
  <img src="https://github.com/alexandershaw4/aLogLikeFit/assets/your-image2.png" width="600">
</div>

---

## \ud83d\udcda Documentation

Each function is documented internally.  
See:
- `fitVariationalLaplaceThermo.m` for classical VL
- `fitVL_LowRankNoise.m` for noise-structured VL
- `peb_ard_with_stats_var.m` for second-level modeling.

---

## \ud83e\uddd0 Why Use aLogLikeFit?

- Designed for **neurophysiology DCMs** but adaptable to general dynamical system models.
- Supports **structured noise** \u2014 not just homoscedastic Gaussian assumptions.
- **Variational + Empirical Bayes** in one lightweight toolbox.
- Minimal dependencies (pure MATLAB).
- Well-suited for **small samples** and **hierarchical modeling**.

---

## \ud83d\udd25 Repository Structure
| File                            | Description |
|----------------------------------|-------------|
| `fitVariationalLaplace.m`        | Basic VL optimizer |
| `fitVariationalLaplaceThermo.m`  | VL + Thermodynamic integration |
| `fitVL_LowRankNoise.m`           | VL + Low-rank noise covariance |
| `aFitDCM.m`                      | Simple wrapper for DCM fitting |
| `peb_ard_with_stats_var.m`       | PEB + ARD (variable noise) |
| `peb_ard_with_stats_LM.m`        | PEB + ARD + Levenberg-Marquardt |

---

## \ud83d\udce2 Notes
- MATLAB required (tested with R2020b+).
- All code \u00a9 Alexander Shaw 2025.
