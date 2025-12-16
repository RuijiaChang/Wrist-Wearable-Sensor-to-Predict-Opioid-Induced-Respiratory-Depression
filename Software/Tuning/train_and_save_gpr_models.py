"""
Script to train and save GPR models for SpO2 and RR predictions.
- SpO2: Uses fixed feature set from GPR.py
- RR: Uses optimize_rr_gpr.py logic to find best features, then trains final model
Based on GPR.py and optimize_rr_gpr.py

Usage:
    python train_and_save_gpr_models.py
"""

import os
import sys
import tempfile
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import modules for RR optimization
import ReliefF
import lasso
import treefs

import warnings
warnings.filterwarnings('ignore')

def train_gpr_model(input_csv: str,
                   target_name: str,
                   top_features: List[str]) -> Tuple[dict, StandardScaler, GaussianProcessRegressor]:
    """
    Train GPR model following GPR.py logic.
    Returns (error_metrics, scaler, gpr_model)
    """
    # Load data
    aggregated_df = pd.read_csv(input_csv)
    
    # Separate target
    target = aggregated_df[target_name]
    y = np.round(target)
    
    # Prepare features
    X = aggregated_df.drop(columns=["wave nunmber", "segment nunmber", "SpO2(mean)", "RR(mean)"])
    X_selected = X[top_features]
    
    # Scale features
    scaler_selected = StandardScaler()
    X_selected_scaled = scaler_selected.fit_transform(X_selected)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Starting GPR training for {target_name} (this may take several minutes)...")
    
    # Define kernel (same as GPR.py)
    noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
        noise_level=0.1**2, noise_level_bounds=(1e-15, 1e5)
    )
    kernel = C(1, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2)) + noise_kernel
    
    # Initialize and fit GPR
    gpr = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=20, 
        alpha=1e-2, 
        normalize_y=True
    )
    gpr.fit(X_train, y_train)
    
    print("GPR training completed. Evaluating...")
    
    # Predict
    y_pred = gpr.predict(X_test, return_std=False)
    y_pred_RC = np.clip(np.round(y_pred), a_min=0, a_max=100)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_RC = mean_absolute_error(y_test, y_pred_RC)
    rmse_RC = np.sqrt(mean_squared_error(y_test, y_pred_RC))
    
    error_rate = {
        "MAE": mae,
        "RMSE": rmse,
        "MAE_RC": mae_RC,
        "RMSE_RC": rmse_RC
    }
    
    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"  MAE_RC: {mae_RC:.4f}, RMSE_RC: {rmse_RC:.4f}")
    
    return error_rate, scaler_selected, gpr


def filter_rr_outliers_to_temp_csv(input_csv: str,
                                   target_name: str = "RR(mean)",
                                   z_threshold: float = 3.0) -> str:
    """Create temporary CSV with RR outliers removed."""
    df = pd.read_csv(input_csv)
    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found in {input_csv}")
    
    y = df[target_name].values
    mu, sigma = np.mean(y), np.std(y)
    if sigma == 0:
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
    """Build top-feature sets from multiple selectors for RR target."""
    target_name = "RR(mean)"
    top_sets: Dict[str, Dict[int, List[str]]] = {"ReliefF": {}, "LASSO": {}, "TreeFS": {}}
    
    # ReliefF
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


def find_best_rr_features(input_csv: str) -> Tuple[str, int, List[str]]:
    """
    Find best feature set for RR using optimize_rr_gpr.py logic.
    Returns (best_method, best_k, best_features)
    """
    print("Finding best RR feature set...")
    
    # Get feature sets
    k_values = [8, 11, 14, 18]
    top_sets = get_top_features_sets(input_csv, k_values)
    
    print("Evaluating feature sets...")
    best_key = (None, None)
    best_features: List[str] = []
    best_score = float("inf")
    
    for method, k_map in top_sets.items():
        for k, feats in k_map.items():
            if not feats:
                continue
            try:
                # Train and evaluate
                metrics, _, _ = train_gpr_model(input_csv, "RR(mean)", feats)
                mae_rc = metrics.get("MAE_RC", metrics.get("MAE", float("inf")))
                if mae_rc < best_score:
                    best_score = mae_rc
                    best_key = (method, k)
                    best_features = feats
            except Exception as e:
                print(f"  Skipping {method}_k{k}: {e}")
                continue
    
    if best_key == (None, None):
        raise RuntimeError("No valid feature set found for RR.")
    
    print(f"Best RR features: {best_key[0]}, k={best_key[1]}, MAE_RC={best_score:.4f}")
    return best_key[0], best_key[1], best_features


def save_spo2_model(input_csv: str, output_path: str):
    """Train and save SpO2 GPR model."""
    print("\n" + "="*60)
    print("Training SpO2 GPR Model")
    print("="*60)
    
    # Use fixed feature set from GPR.py
    top_features = [
        'Median', 'Percentile_75', 'mean(t1/tpi)', 'mfcc_2', 
        'mean(tpp)', 'mean(sys_amp)', 'mean(tpi)', 'InterquartileRange', 
        'mfcc_8', 'mfcc_6', 'mean(foot_amp)'
    ]
    
    print(f"Using {len(top_features)} features: {top_features}")
    
    # Train model
    metrics, scaler, gpr_model = train_gpr_model(input_csv, "SpO2(mean)", top_features)
    
    # Save model bundle
    model_bundle = {
        'model': gpr_model,
        'scaler': scaler,
        'features': top_features,
        'target_name': 'SpO2(mean)',
        'metrics': metrics
    }
    
    joblib.dump(model_bundle, output_path)
    print(f"\nSpO2 model saved to: {output_path}")
    print(f"Features: {top_features}")
    print(f"Metrics: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
    
    return model_bundle


def save_rr_model(input_csv: str, output_path: str):
    """Train and save RR GPR model using optimization."""
    print("\n" + "="*60)
    print("Training RR GPR Model")
    print("="*60)
    
    # Step 1: Filter outliers
    print("Step 1: Filtering RR outliers...")
    tmp_csv = filter_rr_outliers_to_temp_csv(input_csv, target_name="RR(mean)", z_threshold=3.0)
    
    try:
        # Step 2: Find best features
        print("Step 2: Finding best feature set...")
        best_method, best_k, best_features = find_best_rr_features(tmp_csv)
        
        print(f"\nBest configuration: {best_method}, k={best_k}")
        print(f"Selected features ({len(best_features)}): {best_features}")
        
        # Step 3: Train final model with best features
        print("Step 3: Training final RR model with best features...")
        metrics, scaler, gpr_model = train_gpr_model(tmp_csv, "RR(mean)", best_features)
        
        # Save model bundle
        model_bundle = {
            'model': gpr_model,
            'scaler': scaler,
            'features': best_features,
            'target_name': 'RR(mean)',
            'best_method': best_method,
            'best_k': best_k,
            'metrics': metrics
        }
        
        joblib.dump(model_bundle, output_path)
        print(f"\nRR model saved to: {output_path}")
        print(f"Best method: {best_method}, k={best_k}")
        print(f"Features: {best_features}")
        print(f"Metrics: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
        
        return model_bundle
        
    finally:
        # Cleanup temp file
        try:
            os.remove(tmp_csv)
        except Exception:
            pass


def main():
    """Main function to train and save both models."""
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up from Software/Tuning to project root
    
    # Default paths (relative to project root)
    input_csv = os.path.join(project_root, "BIDMC_Regression/features/BIDMC_Segmented_features.csv")
    spo2_output = os.path.join(project_root, "current_spo2_model.pkl")
    rr_output = os.path.join(project_root, "current_rr_model.pkl")
    
    # Check input file
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV not found: {input_csv}")
        print("Please provide the correct path to BIDMC_Segmented_features.csv")
        return 1
    
    # Change to Tuning directory to import modules correctly
    os.chdir(script_dir)
    
    # Add current directory to path for imports
    sys.path.insert(0, script_dir)
    
    try:
        # Train and save SpO2 model
        save_spo2_model(input_csv, spo2_output)
        
        # Train and save RR model
        save_rr_model(input_csv, rr_output)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"SpO2 model: {spo2_output}")
        print(f"RR model: {rr_output}")
        print("\nYou can now use these models with the inference script.")
        
        return 0
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

