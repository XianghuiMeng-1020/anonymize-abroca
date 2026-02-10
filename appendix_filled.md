# Appendix

## Implementation details for ABROCA surrogate, operating points, and optimization

### Numerical computation of test-set ABROCA

To compute ABROCA in (2), we construct empirical ROC curves separately for each group on the test set. We evaluate both ROC curves on a shared monotone FPR grid that includes 0 and 1, and apply trapezoidal integration to the absolute difference between group TPRs.

### Differentiable ABROCA surrogate: grids, interpolation, and integration

**Threshold grid.** We use $N_{\text{bins}}=50$ equally spaced thresholds on $[0,1]$ (in code: `linspace(0.01, 0.99, 50)`).

**Temperature.** We set $\tau=0.1$ (or anneal from $\tau_{\text{start}}=0.05$ to a lower value over epochs using exponential decay $\tau \times 0.5^{\lfloor \text{epoch}/(\text{epochs}/3)\rfloor}$).

**Interpolation.** For each group, we obtain paired sequences $(\widetilde{\mathrm{FPR}}_g(t_k),\widetilde{\mathrm{TPR}}_g(t_k))$. We then construct a shared $\widetilde{\mathrm{FPR}}$ grid and linearly interpolate $\widetilde{\mathrm{TPR}}_g$ onto that grid before trapezoidal integration of the absolute difference.

**Absolute value.** We use the standard subgradient for $|\cdot|$ (and, if needed, specify the tie convention).

### Definitions and operating points for SPD and EOD

SPD and EOD are computed under an outreach policy that flags the top $q\%$ of students on the test set (equivalently, top $K$ students for the test-set cohort size). For a chosen cutoff $q\%$ (equivalently $K$ for a fixed cohort size), let $\hat{Y}=1$ denote "flagged". Then:

$$
\begin{aligned}
\mathrm{SPD} &= \Pr(\hat{Y}=1\mid A=0)-\Pr(\hat{Y}=1\mid A=1),\\
\Delta \mathrm{TPR} &= \Pr(\hat{Y}=1\mid Y=1,A=0)-\Pr(\hat{Y}=1\mid Y=1,A=1),\\
\Delta \mathrm{FPR} &= \Pr(\hat{Y}=1\mid Y=0,A=0)-\Pr(\hat{Y}=1\mid Y=0,A=1),\\
\mathrm{EOD} &= \max\left(|\Delta \mathrm{TPR}|,\ |\Delta \mathrm{FPR}|\right).
\end{aligned}
$$

We report the value(s) of $q$ used in each figure/table.

### Optimization settings (held constant across paired runs)

**Model.**  
Logistic regression implemented in PyTorch.

**Optimizer and learning rate.**  
Adam; learning rate $0.01$; weight decay $0.001$.

**Batching.**  
Full-batch (no minibatching).

**Early stopping.**  
Patience $50$; maximum epochs $1000$; checkpoint criterion matches the training objective (CE vs. total loss).

**Stabilization.**  
Gradient clipping with max norm $1.0$ (no EMA or other smoothing in the reported runs).

**Random seeds.**  
Ten fixed seeds: $42$, $123$, $456$, $789$, $1011$, $1213$, $1415$, $1617$, $1819$, $2021$.


% =========================
% (C) Digital Appendix full table (rotated 90 degrees)
% Add to preamble if needed: \usepackage{rotating}
% =========================
\begin{sidewaystable}[p]
\centering
\caption{Digital Appendix: Full hyperparameter sensitivity table (AUC, ACC, ABROCA, SPD, EOD) across Kaggle, OULAD, and UCI Student Performance.
Values are mean $\pm$ standard deviation on the validation split (5 seeds for Kaggle and UCI; 3 seeds for OULAD).}
\label{tab:hyperparameter_sensitivity_all_full_appendix}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.13}
\begin{tabular}{llccccc|ccccc|ccccc}
\toprule
\multirow{2}{*}{\textbf{Hyperparameter}} & \multirow{2}{*}{\textbf{Value}}
& \multicolumn{5}{c|}{\textbf{Kaggle}}
& \multicolumn{5}{c|}{\textbf{OULAD}}
& \multicolumn{5}{c}{\textbf{UCI}} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(lr){13-17}
& & \textbf{AUC} & \textbf{ACC} & \textbf{ABROCA} & \textbf{SPD} & \textbf{EOD}
& \textbf{AUC} & \textbf{ACC} & \textbf{ABROCA} & \textbf{SPD} & \textbf{EOD}
& \textbf{AUC} & \textbf{ACC} & \textbf{ABROCA} & \textbf{SPD} & \textbf{EOD} \\
\midrule
\multirow{5}{*}{$\lambda$} & 0.0
& 0.948$\pm$0.001 & 0.908$\pm$0.002 & 0.036$\pm$0.001 & -0.233$\pm$0.011 & 0.009$\pm$0.001
& 0.791$\pm$0.000 & 0.718$\pm$0.001 & 0.058$\pm$0.000 & 0.079$\pm$0.001 & 0.041$\pm$0.002
& 0.955$\pm$0.002 & 0.901$\pm$0.005 & 0.027$\pm$0.003 & -0.048$\pm$0.026 & -0.033$\pm$0.016 \\
& 0.1
& 0.948$\pm$0.001 & 0.905$\pm$0.003 & 0.034$\pm$0.001 & -0.221$\pm$0.008 & 0.017$\pm$0.002
& 0.791$\pm$0.000 & 0.720$\pm$0.001 & 0.057$\pm$0.000 & 0.074$\pm$0.001 & 0.036$\pm$0.003
& 0.953$\pm$0.002 & 0.896$\pm$0.009 & 0.027$\pm$0.002 & -0.043$\pm$0.014 & -0.020$\pm$0.013 \\
& 0.3
& 0.945$\pm$0.001 & 0.900$\pm$0.001 & 0.031$\pm$0.001 & -0.209$\pm$0.007 & 0.009$\pm$0.005
& 0.791$\pm$0.000 & 0.720$\pm$0.000 & 0.054$\pm$0.000 & 0.062$\pm$0.002 & 0.027$\pm$0.003
& 0.945$\pm$0.001 & 0.876$\pm$0.015 & 0.026$\pm$0.004 & -0.045$\pm$0.020 & -0.009$\pm$0.028 \\
& 0.5
& 0.942$\pm$0.001 & 0.899$\pm$0.001 & 0.028$\pm$0.001 & -0.214$\pm$0.005 & -0.009$\pm$0.000
& 0.790$\pm$0.000 & 0.716$\pm$0.000 & 0.052$\pm$0.000 & 0.055$\pm$0.001 & 0.020$\pm$0.001
& 0.939$\pm$0.002 & 0.881$\pm$0.013 & 0.019$\pm$0.002 & -0.016$\pm$0.022 & 0.035$\pm$0.031 \\
& 1.0
& 0.935$\pm$0.001 & 0.889$\pm$0.002 & 0.021$\pm$0.001 & -0.196$\pm$0.004 & -0.003$\pm$0.000
& 0.786$\pm$0.001 & 0.711$\pm$0.002 & 0.045$\pm$0.001 & 0.036$\pm$0.003 & 0.006$\pm$0.003
& 0.918$\pm$0.005 & 0.848$\pm$0.000 & 0.027$\pm$0.006 & 0.087$\pm$0.000 & 0.151$\pm$0.000 \\
\midrule
\multirow{5}{*}{$\tau$} & 0.01
& 0.945$\pm$0.001 & 0.899$\pm$0.001 & 0.029$\pm$0.001 & -0.213$\pm$0.006 & 0.000$\pm$0.005
& 0.790$\pm$0.000 & 0.718$\pm$0.001 & 0.055$\pm$0.000 & 0.068$\pm$0.003 & 0.032$\pm$0.001
& 0.944$\pm$0.002 & 0.871$\pm$0.012 & 0.021$\pm$0.004 & -0.054$\pm$0.024 & -0.022$\pm$0.034 \\
& 0.05
& 0.945$\pm$0.001 & 0.900$\pm$0.001 & 0.030$\pm$0.001 & -0.211$\pm$0.005 & 0.005$\pm$0.002
& 0.791$\pm$0.000 & 0.719$\pm$0.000 & 0.055$\pm$0.000 & 0.066$\pm$0.002 & 0.029$\pm$0.001
& 0.945$\pm$0.001 & 0.873$\pm$0.014 & 0.022$\pm$0.005 & -0.049$\pm$0.023 & -0.016$\pm$0.032 \\
& 0.1
& 0.945$\pm$0.001 & 0.900$\pm$0.001 & 0.031$\pm$0.001 & -0.209$\pm$0.007 & 0.009$\pm$0.005
& 0.791$\pm$0.000 & 0.720$\pm$0.000 & 0.054$\pm$0.000 & 0.062$\pm$0.002 & 0.027$\pm$0.003
& 0.945$\pm$0.001 & 0.876$\pm$0.015 & 0.026$\pm$0.004 & -0.045$\pm$0.020 & -0.009$\pm$0.028 \\
& 0.5
& 0.947$\pm$0.001 & 0.904$\pm$0.003 & 0.034$\pm$0.001 & -0.217$\pm$0.007 & 0.016$\pm$0.004
& 0.791$\pm$0.000 & 0.721$\pm$0.000 & 0.056$\pm$0.000 & 0.071$\pm$0.002 & 0.033$\pm$0.004
& 0.952$\pm$0.002 & 0.891$\pm$0.006 & 0.025$\pm$0.002 & -0.033$\pm$0.014 & -0.011$\pm$0.004 \\
& 1.0
& 0.948$\pm$0.001 & 0.905$\pm$0.003 & 0.035$\pm$0.001 & -0.223$\pm$0.007 & 0.016$\pm$0.005
& 0.791$\pm$0.000 & 0.719$\pm$0.000 & 0.057$\pm$0.000 & 0.075$\pm$0.001 & 0.037$\pm$0.002
& 0.953$\pm$0.002 & 0.899$\pm$0.011 & 0.028$\pm$0.003 & -0.047$\pm$0.008 & -0.020$\pm$0.013 \\
\midrule
\multirow{5}{*}{$N_{\text{bins}}$} & 10
& 0.945$\pm$0.001 & 0.900$\pm$0.001 & 0.031$\pm$0.001 & -0.209$\pm$0.007 & 0.009$\pm$0.005
& 0.791$\pm$0.000 & 0.720$\pm$0.000 & 0.055$\pm$0.000 & 0.062$\pm$0.001 & 0.027$\pm$0.003
& 0.946$\pm$0.000 & 0.876$\pm$0.015 & 0.026$\pm$0.004 & -0.045$\pm$0.020 & -0.009$\pm$0.028 \\
& 25
& 0.945$\pm$0.001 & 0.900$\pm$0.001 & 0.031$\pm$0.001 & -0.209$\pm$0.007 & 0.009$\pm$0.005
& 0.791$\pm$0.000 & 0.720$\pm$0.000 & 0.054$\pm$0.000 & 0.063$\pm$0.001 & 0.027$\pm$0.002
& 0.945$\pm$0.001 & 0.876$\pm$0.015 & 0.026$\pm$0.003 & -0.045$\pm$0.020 & -0.009$\pm$0.028 \\
& 50
& 0.945$\pm$0.001 & 0.900$\pm$0.001 & 0.031$\pm$0.001 & -0.209$\pm$0.007 & 0.009$\pm$0.005
& 0.791$\pm$0.000 & 0.720$\pm$0.000 & 0.054$\pm$0.000 & 0.062$\pm$0.002 & 0.027$\pm$0.003
& 0.945$\pm$0.001 & 0.876$\pm$0.015 & 0.026$\pm$0.004 & -0.045$\pm$0.020 & -0.009$\pm$0.028 \\
& 100
& 0.945$\pm$0.001 & 0.900$\pm$0.001 & 0.031$\pm$0.001 & -0.209$\pm$0.007 & 0.009$\pm$0.005
& 0.791$\pm$0.000 & 0.719$\pm$0.000 & 0.054$\pm$0.000 & 0.062$\pm$0.002 & 0.027$\pm$0.003
& 0.945$\pm$0.001 & 0.876$\pm$0.015 & 0.026$\pm$0.004 & -0.045$\pm$0.020 & -0.009$\pm$0.028 \\
& 200
& 0.945$\pm$0.001 & 0.900$\pm$0.001 & 0.031$\pm$0.001 & -0.209$\pm$0.007 & 0.009$\pm$0.005
& 0.791$\pm$0.000 & 0.719$\pm$0.000 & 0.054$\pm$0.000 & 0.062$\pm$0.002 & 0.027$\pm$0.002
& 0.945$\pm$0.001 & 0.876$\pm$0.015 & 0.026$\pm$0.004 & -0.045$\pm$0.020 & -0.009$\pm$0.028 \\
\bottomrule
\end{tabular}

\vspace{0.35em}
\footnotesize
\begin{minipage}{0.95\textheight}
\justifying
\emph{Note.}
Fixed settings per sweep block:
$\lambda$ sweep fixes $\tau=0.1$ and $N_{\text{bins}}=50$;
$\tau$ sweep fixes $\lambda=0.3$ and $N_{\text{bins}}=50$;
$N_{\text{bins}}$ sweep fixes $\lambda=0.3$ and $\tau=0.1$.
Within each dataset, all runs use the same split rule, preprocessing, optimizer, and early-stopping criterion.
\end{minipage}
\end{sidewaystable}

---

## Appendix Figures

### RQ1 ABROCA model (validation ABROCA)
![rq1_abroca_val_abroca](figure/rq1_abroca_val_abroca.png)

### RQ1 ABROCA model (validation AUC)
![rq1_abroca_val_auc](figure/rq1_abroca_val_auc.png)

### RQ1 CE model (validation ABROCA)
![rq1_ce_val_abroca](figure/rq1_ce_val_abroca.png)

### RQ1 CE model (validation AUC)
![rq1_ce_val_auc](figure/rq1_ce_val_auc.png)

### RQ1 Dual-objective model (validation ABROCA)
![rq1_dual_val_abroca](figure/rq1_dual_val_abroca.png)

### RQ1 Dual-objective model (validation AUC)
![rq1_dual_val_auc](figure/rq1_dual_val_auc.png)


