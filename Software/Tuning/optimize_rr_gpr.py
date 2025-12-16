'''
Script to optimize GPR model for RR prediction using multiple feature selection methods.
- Uses ReliefF, LASSO, and TreeFS to select top features for RR target
- Evaluates each feature set with core GPR implementation
- Picks best feature set based on MAE_RC metric
- Does not modify core GPR.py; acts as a wrapper
- Outputs best configuration and summary CSV

Usage:
    python optimize_rr_gpr.py
'''

import os
import tempfile
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Import existing modules without changing them
from GPR import Run_GPR as Run_GPR_core, Print_ErrorRate as Print_ErrorRate_core
import ReliefF
import lasso
import treefs


def filter_rr_outliers_to_temp_csv(input_csv: str,
                                   target_name: str = "RR(mean)",
                                   z_threshold: float = 3.0) -> str:
    """
    Create a temporary CSV with RR outliers removed, so we don't modify the core file.
    Returns the path to the temporary CSV.
    """
    df = pd.read_csv(input_csv)
    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found in {input_csv}")

    y = df[target_name].values
    mu, sigma = np.mean(y), np.std(y)
    if sigma == 0:
        # Nothing to filter
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="rr_clean_", suffix=".csv")
        os.close(tmp_fd)
        df.to_csv(tmp_path, index=False)
        return tmp_path

    keep_mask = np.abs(y - mu) <= z_threshold * sigma
    df_clean = df.loc[keep_mask].reset_index(drop=True)

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="rr_clean_", suffix=".csv")
    os.close(tmp_fd)
    df_clean.to_csv(tmp_path, index=False)
    return tmp_path


def get_top_features_sets(input_csv: str,
                          k_values: List[int]) -> Dict[str, Dict[int, List[str]]]:
    """
    Build top-feature sets from multiple selectors for RR target.
    Returns a mapping: method -> {k -> feature_list}
    """
    target_name = "RR(mean)"
    top_sets: Dict[str, Dict[int, List[str]]] = {"ReliefF": {}, "LASSO": {}, "TreeFS": {}}

    # ReliefF (returns DataFrame with Feature column)
    for k in k_values:
        try:
            rel_df = ReliefF.Run_ReliefF(input_csv, target_name, featureNum=k)
            feats = rel_df["Feature"].tolist()
            top_sets["ReliefF"][k] = feats
        except Exception:
            top_sets["ReliefF"][k] = []

    # LASSO
    for k in k_values:
        try:
            lasso_df = lasso.Run_LASSO(input_csv, target_name, n_top=k)
            feats = lasso_df["Feature"].tolist()
            top_sets["LASSO"][k] = feats
        except Exception:
            top_sets["LASSO"][k] = []

    # TreeFS
    for k in k_values:
        try:
            tree_df = treefs.Run_TreeFS(input_csv, target_name, n_top=k)
            feats = tree_df["Feature"].tolist()
            top_sets["TreeFS"][k] = feats
        except Exception:
            top_sets["TreeFS"][k] = []

    return top_sets


def evaluate_feature_sets(input_csv: str,
                          top_sets: Dict[str, Dict[int, List[str]]],
                          output_dir: str) -> Tuple[str, int, Dict[str, float]]:
    """
    Iterate over all feature sets, run core GPR once per set, and pick best by MAE_RC.
    Returns (best_method, best_k, best_metrics)
    """
    target_name = "RR(mean)"
    best_key = (None, None)
    best_metrics: Dict[str, float] = {}
    best_score = float("inf")

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    for method, k_map in top_sets.items():
        for k, feats in k_map.items():
            if not feats:
                continue
            try:
                metrics = Run_GPR_core(
                    input_csv,
                    target_name,
                    feats,
                    Beep=False,
                    StoreComparision=True,
                    output_dir=output_dir,
                    output_File_Name=f"RR_Prediction_Comparison_{method}_k{k}.csv"
                )
                mae_rc = metrics.get("MAE_RC", metrics.get("MAE", float("inf")))
                if mae_rc < best_score:
                    best_score = mae_rc
                    best_key = (method, k)
                    best_metrics = metrics
            except Exception:
                # Skip invalid combos
                continue

    if best_key == (None, None):
        raise RuntimeError("No valid feature set produced a successful GPR run for RR.")

    return best_key[0], best_key[1], best_metrics


def main():
    # Inputs
    input_csv = "BIDMC_Regression/features/BIDMC_Segmented_features.csv"
    output_dir = "BIDMC_Regression/features"

    print("== RR Prediction Optimizer (wrapper over GPR.py) ==")
    if not os.path.exists(input_csv):
        print(f"ERROR: input not found: {input_csv}")
        return 1

    # 1) Create a temporary CSV with RR outliers removed (no change to core GPR)
    print("Step 1/3: Filtering RR outliers to a temporary CSV (z<=3)...")
    tmp_csv = filter_rr_outliers_to_temp_csv(input_csv, target_name="RR(mean)", z_threshold=3.0)
    print(f"  Temp cleaned file: {tmp_csv}")

    # 2) Prepare feature candidates from multiple selectors and various top-k sizes
    print("Step 2/3: Building RR-specific feature sets (ReliefF, LASSO, TreeFS)...")
    k_values = [8, 11, 14, 18]
    top_sets = get_top_features_sets(tmp_csv, k_values)
    for method, k_map in top_sets.items():
        filled = {k: len(v) for k, v in k_map.items()}
        print(f"  {method}: {filled}")

    # 3) Evaluate all combinations via core GPR and pick best by MAE_RC
    print("Step 3/3: Evaluating feature sets with core GPR (RR target)...")
    best_method, best_k, best_metrics = evaluate_feature_sets(tmp_csv, top_sets, output_dir)

    print("\n== BEST CONFIGURATION FOR RR ==")
    print(f"Method: {best_method}, Top-K: {best_k}")
    Print_ErrorRate_core(best_metrics)

    # Optional: write a small summary CSV for quick reference
    summary_path = os.path.join(output_dir, "RR_GPR_optimization_summary.csv")
    pd.DataFrame([
        {
            "method": best_method,
            "top_k": best_k,
            **best_metrics
        }
    ]).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Cleanup temp file
    try:
        os.remove(tmp_csv)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


