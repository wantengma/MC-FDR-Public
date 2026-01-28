# Project Overview
This repository reproduces the results reported in the following paper:

Ma, W., Du, L., Xia, D., & Yuan, M. (2023). Multiple testing of linear forms for noisy matrix completion. arXiv preprint arXiv:2312.00305.



# Get Started

Run `Weak/MatrixSimu.m` with `TestCase = 0` to reproduce “Data aggregation under weak dependency” for blockwise matrix tests, which corresponds to Figure 2. 

Set `TestCase = 1` to reproduce "Whitening and screening" for  row tests, which corresponds to Figure 3.


# File Description
- `Variance Comparison/` This is the simulation to compare variance characterization (e.g., Figure 1, and additional figures in supplement)
  - `Draw_W_Compare.m` Plot the empirical distribution function $\bar{F}_n(z)-\Phi(z)$
  - `Draw_W.m` Plot the histogram of distribution for statistic $W_T$
- `Weak/` Simulation under weak dependence 
  - `MatrixSimu.m` Simulation for “Data aggregation under weak dependency”  and "Whitening and screening" (Figure 2,3)
  - `MatrixSimu_Hetero.m` Simulation for "Performance under Heterogeneous Noise"
  - `MatrixSimu_SDA_Weak.m` Simulation for "Performance under Very Weak Dependence"
- `Moderate/MatrixSimu_noises_moderate.m` Simulation under moderate dependence, with heavy-tailed noise (Figure 4)
- `Strong/MatrixSimu_noises_strong.m` Simulation under strong dependence, with heavy-tailed noise (Figure 5)
- `Robustness/MatrixSimu.m` Simulation for "Robustness under Model Misspecification"
- `MovieLens` Simulation for MovieLens dataset (Figure 6)
- `Rossmann/` Simulation for Rossmann dataset
  - `Rossmann.m` FDR and Power performance comparison (Figure 7)
  - `Rossmann_ROC.m` Draw ROC curve for the rossmann result (Figure 8) 