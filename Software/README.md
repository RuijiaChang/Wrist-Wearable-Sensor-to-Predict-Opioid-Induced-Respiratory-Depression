# Software Guide

This document explains how to use the current Software stack to train and run the respiratory (RR) and oxygen saturation (SpO₂) predictors. The older feature-extraction modules under `Software/BIDMC_Feature_Aggregation` and related legacy scripts remain in the repository but are unchanged from the 2025Spring team. The focus here is on the new training, inference, and evaluation utilities that tie the system together.

## Environment Setup

1. Activate the shared Conda environment that already contains GPU-enabled PyTorch and CUDA:
   ```bash
   conda activate pytorch
   ```
2. Install the Python packages that the new scripts depend on:
   ```bash
   pip install -r requirement.txt
   ```

## Directory Overview

- `Software/Tuning/train_and_save_gpr_models.py` — trains Gaussian Process Regression (GPR) models for current SpO₂ and RR predictions and serializes them as reusable bundles.
- `Software/Tuning/real_future_prediction.py` — trains the Transformer + XGBoost multi-window future predictor (10–300 s horizons) and saves the checkpoint consumed during inference.
- `Software/infer.py` — unified inference entry point. Runs current-state GPR inference and/or future Transformer+XGBoost inference from JSON vectors or historical CSVs.
- `Software/test_inference_example.sh` — runnable example that wires the above models together using the sample inputs in `Software/Test_Infer/`.
stress_test_model_integration
- `Software/Tuning/inference_pipeline_evaluation.py` — stress-tests the inference entry point for latency, timing stability, and robustness using synthetic inputs.
- `Software/Checkpoints/` — staging area for `.pkl` model bundles. Populate this after training so that the inference scripts and evaluation harness can find the models.

## Data Preparation
See README.txt written by 2025Spring team.

## Training Current-State GPR Models

`Software/Tuning/train_and_save_gpr_models.py` automates the steps that previously lived in `GPR.py` and `optimize_rr_gpr.py`:

```bash
cd Software/Tuning
python train_and_save_gpr_models.py
```

What happens:

1. SpO₂ model — uses the fixed feature list defined in the script, trains a Gaussian Process Regressor, and saves the tuple `{model, scaler, feature list, metrics}` as `current_spo2_model.pkl`.
2. RR model — filters outliers, evaluates feature subsets from ReliefF, LASSO, and TreeFS, retrains the best-performing configuration, and saves it as `current_rr_model.pkl`.

Put `.pkl` files to `Software/Checkpoints/` so that inference can find them. 

## Training the Transformer + XGBoost Future Model

`Software/Tuning/real_future_prediction.py` prepares 60-step sequences from the same BIDMC CSV, trains a Transformer encoder that produces embeddings, and then fits window-specific XGBoost regressors for each target horizon (10, 20, …, 300 s). Run:

```bash
cd Software/Tuning
python real_future_prediction.py
```

Key notes:

- Edit the `future_offsets`, `seq_len`, or `input_csv` variables near the bottom of the script if your dataset differs.
- Training outputs one file, `transformer_xgboost_model_extended_10-300s.pkl`. Store it under `Software/Checkpoints/` when done so that `infer.py` can load it.
- The saved bundle contains the Transformer state_dict, feature names, offsets, and all XGBoost models. No additional assets are required at inference time.

## Running Unified Inference

The `Software/infer.py` script ties everything together. It can ingest either a single feature vector (JSON) or a CSV history, run future predictions, and optionally add the current-state GPR estimates.

Basic usage:

```bash
python Software/infer.py \
  --future_model Software/Checkpoints/transformer_xgboost_model_extended_10-300s.pkl \
  --input_json Software/Test_Infer/test_input_features.json \
  --current_spo2_model Software/Checkpoints/current_spo2_model.pkl \
  --current_rr_model Software/Checkpoints/current_rr_model.pkl \
  --output Software/Test_Infer/test_prediction_output.json \
  --print_pretty
```

To sanity-check everything end-to-end, run `Software/test_inference_example.sh`, which wires in the sample inputs under `Software/Test_Infer/` and writes the predictions back to that folder.
