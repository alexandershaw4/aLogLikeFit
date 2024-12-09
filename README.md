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


