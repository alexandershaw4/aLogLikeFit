# ThermoVL & Variational Laplace Toolkit

**Author:** Alexander D. Shaw
**Lab:** Computational Psychiatry & Neuropharmacological Systems (CPNS), University of Exeter
**Website:** [https://cpnslab.com](https://cpnslab.com)

This repository contains a research-grade MATLAB toolkit for **Variational Laplace (VL)** inference in nonlinear dynamical systems, with a particular focus on **thermodynamic integration, low-rank and heteroscedastic noise models, hierarchical (PEB-style) inference, generalised coordinates, and active / polyphonic extensions**.

The codebase reflects an experimental and methodological platform rather than a single “library-style” package. Many files represent alternative formulations, ablation studies, and exploratory variants used in ongoing research on **Dynamic Causal Modelling (DCM), thalamo-cortical neural mass models, computational psychiatry, and neuro-inspired AI / active inference**.

---

## Conceptual Overview

At its core, this repository implements and extends **Variational Laplace** as a practical inference scheme for models of the form:

[ y = f(m, u, M) + e, \quad e \sim \mathcal{N}(0, \Sigma(\theta)) ]

where:

* ( m ) are latent parameters or states
* ( f(\cdot) ) is a nonlinear forward / generative model
* ( \Sigma ) may be **structured, low-rank, heteroscedastic, or learned online**

Key research directions represented here:

* **Thermodynamic Variational Laplace (TherMO-VL)**
  Annealed free energy optimisation using temperature schedules to improve convergence and reduce local minima.

* **Low-rank & structured noise models**
  Efficient covariance representations for high-dimensional observations and spectral data.

* **Generalised Coordinates (VL-GC)**
  State-space inference in generalised coordinates of motion for dynamic systems and DCM-style models.

* **Hierarchical & Empirical Bayes (PEB / ARD)**
  Group-level shrinkage, automatic relevance determination, and parameter field inference.

* **Active / Expected Free Energy variants**
  Extensions toward control, policy selection, and closed-loop inference.

* **Polyphonic Inference**
  Multi-voice, non-dominating posterior representations for multimodal and non-Gaussian posteriors.

* **Riemannian & geometric updates**
  Metric-aware natural gradient style updates for improved optimisation stability.

---

## Repository Structure

### Core Algorithms

* `fitVariationalLaplace.m`
  Baseline Variational Laplace implementation.

* `fitVariationalLaplaceThermo.m`
  Thermodynamic / annealed VL with temperature scheduling.

* `fitVariationalLaplaceThermoFE.m`
  Free-energy–driven variant with explicit FE tracking.

* `fitVariationalLaplaceThermoStable.m`
  Stability-enhanced version with safeguarded updates.

* `fitVariationalLaplaceThermoStruct.m`
  Structured covariance and model-aware noise updates.

---

### Generalised Coordinates & Dynamic Systems

* `fitVariationalLaplaceThermo_GC.m`
* `fitVariationalLaplaceThermo_GClam.m`
* `dcm_vl_gc.m`, `dcm_vl_gc_time.m`

VL in generalised coordinates for state-space and DCM-style inference.

See:

* `VL_in_GeneralisedCoordinates.pdf`
* `VL_in_GC_new.pdf`

---

### Hierarchical Inference & Empirical Bayes

* `fitHierarchicalVL.m`
* `Wrapper_AlogLikeDCM_fitHierarchicalVL.m`

#### PEB / ARD Framework (`PEB_ARD_general/`)

Group-level parameter field inference with automatic relevance determination:

* `demo_peb_ard.m`
* `peb_ard_novar.m`
* `peb_ard_predict.m`
* `peb_plot_betas.m`
* `peb_plot_beta_densities.m`
* `peb_plot_lambda.m`

Includes examples, cross-validation, and shrinkage visualisation tools.

---

### Noise Models & Likelihood Variants

* `fitVL_LowRankNoise.m`, `fitVL_LowRankNoise2.m`
  Low-rank and structured observation noise models.

* `fitVariationalLaplaceThermoRadialPrecision.m`
  Radial / precision-field updates.

* `fitLogLikelihoodLM.m`, `fitLogLikelihoodLMFE.m`
  Likelihood-based fitting and free-energy variants.

---

### Polyphonic Inference (Multimodal Posteriors)

Located in `polyphonic/`

* `fitVariationalLaplaceThermoPolyphonic.m`
  Multi-voice VL where several coupled Gaussian posteriors coexist and are softly aligned by predictive agreement rather than collapsed into a single mode.

* `plotPolyphonicPosterior.m`
  Visualisation of multimodal and non-unimodal posterior structure.

This is experimental and intended for research into **pluralistic inference and non-dominating integration**.

---

### Active / Control-Oriented Variants

* `fitVariationalLaplaceThermo_active.m`
  Extension toward expected free energy minimisation and action selection.

* `fitDEM_ThermoVL.m`
  DEM-style state and parameter inference under thermodynamic VL.

---

### Riemannian & Geometric Optimisation

Located in `riemannian/`

* `fitVariationalLaplaceThermo_Riemannian.m`
* `defaultMetricDiagonalH.m`

Metric-aware update rules inspired by natural gradients and information geometry.

---

### DCM, Neurodynamics & Model Fitting

* `aFitDCM.m`
  Wrapper for fitting DCM / neural mass models using ThermoVL backends.

* `condFIM.m`
  Conditional Fisher Information Matrix utilities.

* `propose_deltaF_ranked.m`
  Model comparison and ranked free-energy perturbations.

* `dip/`
  Hybrid optimisation and MOGA-based parameter search for dynamical inversion problems.

---

### Demonstrations & Test Functions

Located in:

* `dem_test/`
* `test_fun/`

Includes toy systems, nonlinear oscillators, bi-exponential delays, sigmoid shifts, and hierarchical test cases for validating inference behaviour.

---

### Theory & Technical Notes

* `thermoVL_equations.pdf`
  Formal derivation of thermodynamic VL and annealed free energy updates.

* `Extending_Variational_Laplace_for_Hierarchical_Model_Fitting__Innovations_in_fitVLLowRankNoise_3.pdf`
  Low-rank noise, hierarchical inference, and empirical Bayes extensions.

---

## Typical Workflow

### 1. Define a Forward Model

Write a MATLAB function:

```matlab
function yhat = f(m, M, U)
    % m : parameter vector
    % M : model structure
    % U : inputs
    % yhat : predicted observations
end
```

### 2. Set Priors

```matlab
m0 = prior_mean;
S0 = prior_covariance;
```

### 3. Run ThermoVL

```matlab
OPT.Tschedule = linspace(2.0, 1.0, 16);
OUT = fitVariationalLaplaceThermo(y, @f, m0, S0, OPT);
```

### 4. Inspect

* Posterior means / covariances: `OUT.m`, `OUT.S`
* Free energy trajectory: `OUT.F`
* Predictions: `OUT.yhat`

---

## Research Context

This toolkit underpins ongoing work in:

* Computational psychiatry (M/EEG, pharmacological modelling, synaptic inference)
* Thalamo-cortical and neural mass modelling
* Predictive coding and active inference
* Free energy methods for adaptive and neuro-inspired AI
* Hierarchical Bayesian modelling in clinical and translational neuroscience

Several components directly support work described in:

> Shaw, A.D. *Polyphonic Intelligence: Constraint-Based Emergence, Pluralistic Inference, and Non-Dominating Integration*

and related manuscripts on **thermodynamic variational inference, predictive coding, and mechanistic AI**.

---

## Status & Philosophy

This is a **living research repository**. Code quality, interfaces, and naming conventions reflect an evolving experimental platform rather than a polished software release.

Expect:

* Multiple overlapping implementations
* Partially documented experimental variants
* Research-grade performance rather than production-grade APIs

If you are looking for a clean entry point, start with:

* `fitVariationalLaplaceThermo.m`
* `aFitDCM.m`
* `demo_peb_ard.m`

---

## Citation

If you use this code in academic work, please cite:

> Shaw, A.D. Variational Laplace, Thermodynamic Inference, and Polyphonic Models for Nonlinear Dynamical Systems. Computational Psychiatry & Neuropharmacological Systems Lab, University of Exeter.

(Preprint links forthcoming.)

---

## Contact

**Alexander D. Shaw**
Senior Lecturer in Neuroscience & Computational Psychiatry
University of Exeter
[https://cpnslab.com](https://cpnslab.com)

---

## License

This repository is released for **academic and research use**. Please see the project’s license file or contact the author for commercial usage and collaboration.
