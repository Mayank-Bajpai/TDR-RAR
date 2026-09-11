# PINN Groundwater Surrogate (Optuna + TDR-RAR)

This repository contains the complete code implementation for training and evaluating a Physics-Informed Neural Network (PINN) surrogate model for groundwater flow, leveraging **Optuna** for hyperparameter tuning and **TDR-RAR** (Time-Dependent Residual based Residual Adaptive Refinement) for dynamic collocation point generation.

## Directory Structure

* **	raining/**: Contains the core optuna_tdr_rar_pinn_training.py script. This script implements both Phase 1 (Optuna Hyperparameter Search) and Phase 2 (TDR-RAR dynamic point refinement loops).
* **evaluation/**: Contains scripts to evaluate the final trained checkpoints across both PDE physical errors and spatial observation (MAE/RMSE) errors.
* **isualization/**: Contains all scripts used to generate the final deliverable plots, including the Spatial RAR distributions, water budget component bar charts, and overall loss metrics.

## Usage
1. Configure your paths in the respective scripts.
2. Ensure you have the sciann and 	ensorflow libraries installed.
3. Run 	raining/optuna_tdr_rar_pinn_training.py to initiate the end-to-end training pipeline.
