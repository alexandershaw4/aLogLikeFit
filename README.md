This repository contains a collection of routines for fitting first-level (individual) models in computational psychiatry, specifically spectral DCMs to neurophysiology data. 
The repository includes several function but key elements include:

(1) - fitVariationalLaplaceThermo.m. This is an implementation of Variational Laplace with heterscedastic variance and thermodynamic integration. Like DCM, it optimises the ELBO or free energy and returns posterior means and variances.

(2) - peb_ard_with_stats_var.m. This function performs Parametric Empirical Bayes (PEB) - essentially a second level model posed as an optimisation problem and including second order terms.

Brief guide -->

Fit a spectral neural mass model, specified using the DCM conventional structure (DCM with DCM.M.pE, DCM.M.pC,DCM.M.f, DCM.M.IS and DCM.xY specified).

Usage:

Given a fully specified DCM, do:

M = aFitDCM(DCM); % constructor
M.aloglikVLtherm; % run routine

% to re-run / add more iterations:

M.update_parameters(M.Ep)
M.aloglikVLtherm;

% and to access posteriors:

M.Ep
M.CP
M.F

![image](/other/error_map_.png)


Next, extract the individual posterior parameter means and variances and employ Parametric Empirical Bayes using peb_ard_with_stats_var.m

These functions implement a Parametric Empirical Bayes (PEB) method for estimating group-level parameters while incorporating individual-level priors. The 
method combines ridge regression with Bayesian regularisation, using prior covariance information about individual parameters to shrink the group-level 
estimates. It also includes Automatic Relevance Determination (ARD) to determine the importance of each predictor. The _LM version of the code incorporates the 
Levenberg-Marquardt (LM) algorithm to optimise the parameter estimation, adjusting the update step to improve convergence and stability.

The function returns the group-level parameter estimates, ARD hyperparameters, t-statistics, p-values, and the individual-level posterior means and covariances.
