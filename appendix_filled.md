# Appendix

## Implementation details for ABROCA surrogate, operating points, and optimization

This section documents the exact implementation choices used to compute and optimize the ABROCA-based objective so that the training and evaluation pipeline can be reproduced. We separate the numerical test-set ABROCA computation from the differentiable surrogate used during training because they serve different purposes: reporting versus optimization. We also define the operating-point metrics (SPD and EOD) under the top-q% outreach policy used in the main experiments. Finally, we list all optimization controls that were held fixed across paired runs.

### Numerical computation of test-set ABROCA

To compute ABROCA in (2), we construct empirical ROC curves separately for each group on the test set. We evaluate both ROC curves on a shared monotone FPR grid that includes 0 and 1, and apply trapezoidal integration to the absolute difference between group TPRs.

**Description.**  
This procedure gives a threshold-agnostic summary of group disparity in ranking performance. Using a shared FPR grid avoids misalignment artifacts that can appear when each group is evaluated on different grids. Including endpoints 0 and 1 ensures full ROC support and stable integration boundaries. The trapezoidal rule is used for numerical consistency with the rest of the ROC/AUC pipeline.

### Differentiable ABROCA surrogate: grids, interpolation, and integration

**Threshold grid.** We use N_bins=50 equally spaced thresholds on [0,1] (in code: `linspace(0.01, 0.99, 50)`).

**Temperature.** We set tau=0.1 (or anneal from tau_start=0.05 to a lower value over epochs using exponential decay tau * 0.5^(floor(epoch/(epochs/3)))).

**Interpolation.** For each group, we obtain paired sequences (FPR_tilde_g(t_k), TPR_tilde_g(t_k)). We then construct a shared FPR_tilde grid and linearly interpolate TPR_tilde_g onto that grid before trapezoidal integration of the absolute difference.

**Absolute value.** We use the standard subgradient for |.| (and, if needed, specify the tie convention).

**Description.**  
The surrogate replaces hard thresholding with smooth approximations so gradients can propagate through fairness disparity terms during training. The N_bins=50 setting balances numerical resolution and compute cost, while the tau schedule controls smoothness versus sharpness over training. Shared-grid interpolation keeps group comparisons pointwise comparable before integration. In practice, this design stabilizes optimization while remaining close to the reporting definition of ABROCA.

### Definitions and operating points for SPD and EOD

SPD and EOD are computed under an outreach policy that flags the top q% of students on the test set (equivalently, top K students for the test-set cohort size). For a chosen cutoff q% (equivalently K for a fixed cohort size), let Y_hat=1 denote "flagged". Then:

SPD = P(Y_hat=1 | A=0) - P(Y_hat=1 | A=1)  
Delta TPR = P(Y_hat=1 | Y=1,A=0) - P(Y_hat=1 | Y=1,A=1)  
Delta FPR = P(Y_hat=1 | Y=0,A=0) - P(Y_hat=1 | Y=0,A=1)  
EOD = max(|Delta TPR|, |Delta FPR|)

We report the value(s) of q used in each figure/table.

**Description.**  
These metrics are reported at explicit operating points to match the decision workflow where only a limited outreach quota is available. SPD captures differences in overall flagging rates across groups, while EOD captures the larger of true-positive and false-positive rate gaps. Reporting both helps distinguish allocation imbalance from error-rate imbalance under the same cutoff policy. This operating-point view complements ABROCA, which summarizes disparity across all thresholds.

### Optimization settings (held constant across paired runs)

**Model.**  
Logistic regression implemented in PyTorch.

**Optimizer and learning rate.**  
Adam; learning rate 0.01; weight decay 0.001.

**Batching.**  
Full-batch (no minibatching).

**Early stopping.**  
Patience 50; maximum epochs 1000; checkpoint criterion matches the training objective (CE vs. total loss).

**Stabilization.**  
Gradient clipping with max norm 1.0 (no EMA or other smoothing in the reported runs).

**Random seeds.**  
Ten fixed seeds: 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021.

**Description.**  
All optimization settings are fixed across paired comparisons so observed differences are attributable to objective design rather than tuning drift. Full-batch training and fixed seeds reduce run-to-run variability from stochastic minibatch order effects. Objective-matched checkpointing prevents bias toward one loss family during model selection. Gradient clipping is included only for stability control and is applied uniformly in all compared conditions.

---

## Digital Appendix Table A1. Full Hyperparameter Sensitivity (Validation)

**Scope.** Full version of the sensitivity table with all metrics (AUC, ACC, ABROCA, SPD, EOD) across Kaggle, OULAD, and UCI Student Performance.  
**Protocol.** Values are mean ± standard deviation on the validation split (5 seeds for Kaggle and UCI; 3 seeds for OULAD).  
For each sweep block, only one hyperparameter varies and the other two are fixed.

**Description.**  
Table A1 provides the complete operational sensitivity view, including both threshold-free and threshold-dependent metrics. The table is organized into three one-factor sweeps so the contribution of each hyperparameter can be interpreted without cross-factor confounding. Across datasets, readers should compare metric stability trends first, then utility-fairness trade-offs at larger regularization values. The main paper keeps a compact subset (AUC + ABROCA), while this table preserves full deployment-facing diagnostics.

### A1.1 Lambda sweep (fixed: temperature = 0.1, threshold bins = 50)

| Hyperparameter | Value | Kaggle AUC | Kaggle ACC | Kaggle ABROCA | Kaggle SPD | Kaggle EOD | OULAD AUC | OULAD ACC | OULAD ABROCA | OULAD SPD | OULAD EOD | UCI AUC | UCI ACC | UCI ABROCA | UCI SPD | UCI EOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| λ | 0.0 | 0.948±0.001 | 0.908±0.002 | 0.036±0.001 | -0.233±0.011 | 0.009±0.001 | 0.791±0.000 | 0.718±0.001 | 0.058±0.000 | 0.079±0.001 | 0.041±0.002 | 0.955±0.002 | 0.901±0.005 | 0.027±0.003 | -0.048±0.026 | -0.033±0.016 |
| λ | 0.1 | 0.948±0.001 | 0.905±0.003 | 0.034±0.001 | -0.221±0.008 | 0.017±0.002 | 0.791±0.000 | 0.720±0.001 | 0.057±0.000 | 0.074±0.001 | 0.036±0.003 | 0.953±0.002 | 0.896±0.009 | 0.027±0.002 | -0.043±0.014 | -0.020±0.013 |
| λ | 0.3 | 0.945±0.001 | 0.900±0.001 | 0.031±0.001 | -0.209±0.007 | 0.009±0.005 | 0.791±0.000 | 0.720±0.000 | 0.054±0.000 | 0.062±0.002 | 0.027±0.003 | 0.945±0.001 | 0.876±0.015 | 0.026±0.004 | -0.045±0.020 | -0.009±0.028 |
| λ | 0.5 | 0.942±0.001 | 0.899±0.001 | 0.028±0.001 | -0.214±0.005 | -0.009±0.000 | 0.790±0.000 | 0.716±0.000 | 0.052±0.000 | 0.055±0.001 | 0.020±0.001 | 0.939±0.002 | 0.881±0.013 | 0.019±0.002 | -0.016±0.022 | 0.035±0.031 |
| λ | 1.0 | 0.935±0.001 | 0.889±0.002 | 0.021±0.001 | -0.196±0.004 | -0.003±0.000 | 0.786±0.001 | 0.711±0.002 | 0.045±0.001 | 0.036±0.003 | 0.006±0.003 | 0.918±0.005 | 0.848±0.000 | 0.027±0.006 | 0.087±0.000 | 0.151±0.000 |

**Description.**  
In the lambda sweep, larger λ generally lowers ABROCA, indicating stronger pressure toward reduced across-threshold disparity. This fairness gain is accompanied by utility decline in AUC/ACC, with the steepest degradation appearing at λ=1.0, especially on UCI. The middle range (λ≈0.3-0.5) appears to offer a better trade-off than extreme regularization. This pattern supports using moderate fairness weight as the default operating regime.

### A1.2 Temperature sweep (fixed: λ = 0.3, threshold bins = 50)

| Hyperparameter | Value | Kaggle AUC | Kaggle ACC | Kaggle ABROCA | Kaggle SPD | Kaggle EOD | OULAD AUC | OULAD ACC | OULAD ABROCA | OULAD SPD | OULAD EOD | UCI AUC | UCI ACC | UCI ABROCA | UCI SPD | UCI EOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| τ | 0.01 | 0.945±0.001 | 0.899±0.001 | 0.029±0.001 | -0.213±0.006 | 0.000±0.005 | 0.790±0.000 | 0.718±0.001 | 0.055±0.000 | 0.068±0.003 | 0.032±0.001 | 0.944±0.002 | 0.871±0.012 | 0.021±0.004 | -0.054±0.024 | -0.022±0.034 |
| τ | 0.05 | 0.945±0.001 | 0.900±0.001 | 0.030±0.001 | -0.211±0.005 | 0.005±0.002 | 0.791±0.000 | 0.719±0.000 | 0.055±0.000 | 0.066±0.002 | 0.029±0.001 | 0.945±0.001 | 0.873±0.014 | 0.022±0.005 | -0.049±0.023 | -0.016±0.032 |
| τ | 0.1 | 0.945±0.001 | 0.900±0.001 | 0.031±0.001 | -0.209±0.007 | 0.009±0.005 | 0.791±0.000 | 0.720±0.000 | 0.054±0.000 | 0.062±0.002 | 0.027±0.003 | 0.945±0.001 | 0.876±0.015 | 0.026±0.004 | -0.045±0.020 | -0.009±0.028 |
| τ | 0.5 | 0.947±0.001 | 0.904±0.003 | 0.034±0.001 | -0.217±0.007 | 0.016±0.004 | 0.791±0.000 | 0.721±0.000 | 0.056±0.000 | 0.071±0.002 | 0.033±0.004 | 0.952±0.002 | 0.891±0.006 | 0.025±0.002 | -0.033±0.014 | -0.011±0.004 |
| τ | 1.0 | 0.948±0.001 | 0.905±0.003 | 0.035±0.001 | -0.223±0.007 | 0.016±0.005 | 0.791±0.000 | 0.719±0.000 | 0.057±0.000 | 0.075±0.001 | 0.037±0.002 | 0.953±0.002 | 0.899±0.011 | 0.028±0.003 | -0.047±0.008 | -0.020±0.013 |

**Description.**  
Temperature affects the smoothness of the surrogate fairness signal and therefore shifts the utility-fairness balance. Lower τ tends to produce slightly lower ABROCA in some settings but can also reduce utility on UCI, suggesting sharper optimization behavior. Higher τ improves AUC/ACC in several cases but often weakens disparity reduction. The selected default (τ=0.1) sits in a stable middle region with balanced performance across datasets.

### A1.3 Threshold-bin sweep (fixed: λ = 0.3, temperature = 0.1)

| Hyperparameter | Value | Kaggle AUC | Kaggle ACC | Kaggle ABROCA | Kaggle SPD | Kaggle EOD | OULAD AUC | OULAD ACC | OULAD ABROCA | OULAD SPD | OULAD EOD | UCI AUC | UCI ACC | UCI ABROCA | UCI SPD | UCI EOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N_bins | 10  | 0.945±0.001 | 0.900±0.001 | 0.031±0.001 | -0.209±0.007 | 0.009±0.005 | 0.791±0.000 | 0.720±0.000 | 0.055±0.000 | 0.062±0.001 | 0.027±0.003 | 0.946±0.000 | 0.876±0.015 | 0.026±0.004 | -0.045±0.020 | -0.009±0.028 |
| N_bins | 25  | 0.945±0.001 | 0.900±0.001 | 0.031±0.001 | -0.209±0.007 | 0.009±0.005 | 0.791±0.000 | 0.720±0.000 | 0.054±0.000 | 0.063±0.001 | 0.027±0.002 | 0.945±0.001 | 0.876±0.015 | 0.026±0.003 | -0.045±0.020 | -0.009±0.028 |
| N_bins | 50  | 0.945±0.001 | 0.900±0.001 | 0.031±0.001 | -0.209±0.007 | 0.009±0.005 | 0.791±0.000 | 0.720±0.000 | 0.054±0.000 | 0.062±0.002 | 0.027±0.003 | 0.945±0.001 | 0.876±0.015 | 0.026±0.004 | -0.045±0.020 | -0.009±0.028 |
| N_bins | 100 | 0.945±0.001 | 0.900±0.001 | 0.031±0.001 | -0.209±0.007 | 0.009±0.005 | 0.791±0.000 | 0.719±0.000 | 0.054±0.000 | 0.062±0.002 | 0.027±0.003 | 0.945±0.001 | 0.876±0.015 | 0.026±0.004 | -0.045±0.020 | -0.009±0.028 |
| N_bins | 200 | 0.945±0.001 | 0.900±0.001 | 0.031±0.001 | -0.209±0.007 | 0.009±0.005 | 0.791±0.000 | 0.719±0.000 | 0.054±0.000 | 0.062±0.002 | 0.027±0.002 | 0.945±0.001 | 0.876±0.015 | 0.026±0.004 | -0.045±0.020 | -0.009±0.028 |

**Description.**  
Performance is nearly invariant to N_bins from 10 to 200 in this setup, indicating that surrogate integration resolution is not a primary driver of outcomes. This suggests that the fairness gradient signal is already sufficiently represented at moderate grid sizes. The stability across datasets supports using N_bins=50 as a computationally efficient default. In other words, tuning λ and τ matters more than further refining bin granularity.

## Notes

- Fixed settings per sweep block:
  - **λ sweep:** temperature = 0.1, threshold bins = 50
  - **temperature sweep:** λ = 0.3, threshold bins = 50
  - **threshold-bin sweep:** λ = 0.3, temperature = 0.1
- Within each dataset, all runs use the same split rule, preprocessing, optimizer, and early-stopping criterion.
- This appendix table is the **full operational view** (includes threshold-dependent metrics), while the main paper keeps the compact view (AUC + ABROCA) for across-threshold sensitivity interpretation.

---

## Appendix Figures

### RQ1 ABROCA model (validation ABROCA)
![rq1_abroca_val_abroca](figure/rq1_abroca_val_abroca.png)

**Description.**  
This figure shows how validation ABROCA changes for the ABROCA-objective model across the hyperparameter sweep. Lower values indicate smaller across-threshold disparity between groups. The plot is used to identify stable regions where fairness improves without abrupt optimization behavior. It should be interpreted together with the corresponding AUC figure to evaluate trade-offs.

### RQ1 ABROCA model (validation AUC)
![rq1_abroca_val_auc](figure/rq1_abroca_val_auc.png)

**Description.**  
This figure reports the utility side (AUC) for the ABROCA-objective model under the same sweep conditions. It complements the prior fairness plot by showing how discrimination performance moves as fairness pressure changes. Joint reading of the two panels supports selection of moderate settings that avoid utility collapse. The key use is to locate balanced operating points rather than maximizing a single metric.

### RQ1 CE model (validation ABROCA)
![rq1_ce_val_abroca](figure/rq1_ce_val_abroca.png)

**Description.**  
This figure provides a cross-entropy baseline view of validation ABROCA under matched sweep settings. It serves as a reference for the fairness level obtained without explicit ABROCA regularization. Comparing this baseline against ABROCA and dual-objective panels isolates the incremental fairness effect of objective design. The figure therefore anchors interpretation of fairness gains in RQ1.

### RQ1 CE model (validation AUC)
![rq1_ce_val_auc](figure/rq1_ce_val_auc.png)

**Description.**  
This figure shows baseline utility (AUC) for the CE model across the same hyperparameter axis definitions. It provides the utility anchor used to quantify any utility cost from fairness-oriented objectives. Because protocol and preprocessing are held fixed, differences are attributable to objective formulation rather than setup changes. This panel is the primary baseline for utility comparison in RQ1.

### RQ1 Dual-objective model (validation ABROCA)
![rq1_dual_val_abroca](figure/rq1_dual_val_abroca.png)

**Description.**  
This figure presents validation ABROCA for the dual-objective model that combines CE and ABROCA terms. It is used to test whether joint optimization can reduce disparity more than CE while staying more stable than pure fairness optimization. The shape of the curve indicates how sensitive fairness is to regularization in the combined objective. Interpretation should be paired with the dual-objective AUC plot to assess practical trade-offs.

### RQ1 Dual-objective model (validation AUC)
![rq1_dual_val_auc](figure/rq1_dual_val_auc.png)

**Description.**  
This figure reports validation AUC for the dual-objective model and completes the fairness-utility comparison for RQ1. It helps determine whether combined training preserves enough ranking quality while improving disparity metrics. When read with the dual ABROCA panel, it supports selecting moderate hyperparameters with better balance than extremes. This panel is central to model selection decisions reported in the main text.
