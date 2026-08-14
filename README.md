# An Epidemiological–Actuarial Analysis of HIV/AIDS Insurance in Indonesia Using a Bilayer SID Model

This repository contains the MATLAB source code developed for my undergraduate thesis and the accompanying manuscript, currently under academic review.

## Objective

This project analyzes the epidemiological–actuarial dynamics of a bilayer SID model for HIV/AIDS insurance in Indonesia. Numerical simulations are used to investigate the basic reproduction number ($R_0$), parameter estimation with bootstrap uncertainty analysis, and insurer solvency risk.

## Repository Structure

The model is implemented using a modular MATLAB framework to facilitate reproducibility and interpretation of the computational analysis.

- `main_simulation.m` — Primary execution script containing data initialization, multi-start SQP parameter estimation, and actuarial valuation.
- `Simulation_Results.pdf` — Compiled simulation figures, including model trajectories, emergent limit cycles, and solvency-related results.
- Supporting `.m` files — Model-specific ODE functions, statistical objective functions, and plotting utilities, including `ode_macro.m`, `ode_cohort.m`, and `obj_nsse.m`.

## ⚠️Copyright and Usage

> **© 2026 Sean Felix Fefri Hutagaol. All Rights Reserved.**
> 
> This repository contains code associated with a manuscript currently under academic review. The source code is made available for inspection and evaluation only. No permission is granted to copy, modify, redistribute, or use the code for other purposes without prior written permission from the author.
