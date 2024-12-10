This repository contains a collection of routines for fitting first-level (individual) models in computational psychiatry, specifically spectral DCMs to neurophysiology data. 
The repository includes:

fitLogLikelihoodLM.m: A routine for log likelihood estimation, i.e. for model fitting in the context of hierarchical Bayesian models.

fitLogLikelihoodLMFE.m: A variant of the log likelihood estimation routine that incorporates free energy minimisation for better model optimisation.

peb_ard_with_stats.m: A Parametric Empirical Bayes (PEB) routine for fitting models with automatic relevance determination (ARD), including statistical output for model evaluation.

peb_ard_with_stats_LM.m: An adaptation of the PEB routine incorporating log likelihood and free energy estimation for more robust parameter estimation and statistical analysis.


Brief guide -->

Fit a spectral neural mass model, specified using the DCM conventional structure (DCM with DCM.M.pE, DCM.M.pC,DCM.M.f, DCM.M.IS and DCM.xY specified).

Parameter estimation using log-likelihood (and/or Free Energy) optimisation with Levenberg-Marquardt.

The functions estimate model parameters by maximising the log-likelihood (or free energy) through iterative optimisation. The algorithms employ the Levenberg-Marquardt 
method with dynamic updates of hyperparameters (observation variance and damping factor).

Usage:

Given a fully specified DCM, do:


M = aFitDCM(DCM);

M.aloglik(num_iter) ; <-- log likelihood estimation or

M.aloglikFE(num_iter) ; <-- free enery estimation 

% to re-run / add more iterations:

M.update_parameters(M.Ep)

M.aloglikFE(num_iter)

% and to access posteriors:

M.Ep

M.CP

M.F

Next, extract the individual posterior parameter means and variances and employ Parametric Empirical Bayes using peb_ard_with_stats.m or peb_ard_with_stats_LM.m. 

These functions implement a Parametric Empirical Bayes (PEB) method for estimating group-level parameters while incorporating individual-level priors. The 
method combines ridge regression with Bayesian regularisation, using prior covariance information about individual parameters to shrink the group-level 
estimates. It also includes Automatic Relevance Determination (ARD) to determine the importance of each predictor. The _LM version of the code incorporates the 
Levenberg-Marquardt (LM) algorithm to optimise the parameter estimation, adjusting the update step to improve convergence and stability.

The function returns the group-level parameter estimates, ARD hyperparameters, t-statistics, p-values, and the individual-level posterior means and covariances.
