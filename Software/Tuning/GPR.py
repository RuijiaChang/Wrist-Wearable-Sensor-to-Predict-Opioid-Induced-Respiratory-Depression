import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.gaussian_process.kernels import WhiteKernel
# import winsound
import platform
import os

def Run_GPR(input_csv: str,
            targetName: str,
            TopFeatures: list[str],
            Beep: bool = False,
            StoreComparision: bool = False,
            output_dir: str = None,
            output_File_Name: str = "Prediction_Comparision.csv") -> dict[str, float]:
    # Load the aggregated features CSV (adjust the file path as needed)
    aggregated_df = pd.read_csv(input_csv)

    # Separate the target variable
    target = aggregated_df[targetName]
    # y = pd.qcut(target, q=2, labels=False)
    y = np.round(target)

    X = aggregated_df.drop(columns=["wave nunmber", "segment nunmber", "SpO2(mean)", "RR(mean)"])
    X_selected = X[TopFeatures]

    
    scaler_selected = StandardScaler()
    X_selected_scaled = scaler_selected.fit_transform(X_selected)


    X_train, X_test, y_train, y_test = train_test_split(X_selected_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print("Starting GPR training (this may take several minutes)...")
    
    # kernel = C(1, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2))
    noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-15, 1e5)             # noise_level_bounds modified from 1e-10 to 1e-15
    )
    kernel = C(1, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2)) + noise_kernel

    # Initialize and fit the Gaussian Process Regressor.
    # n_restarts_optimizer is set to 10 to ensure robust kernel parameter optimization.
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=1e-2, normalize_y=True)
    gpr.fit(X_train, y_train)
    
    print("GPR training completed. Making predictions...")

    # y_pred, sigma = gpr.predict(X_test, return_std=True)
    
    # Use return_std=False for faster prediction (we don't need uncertainty for evaluation)
    y_pred = gpr.predict(X_test, return_std=False)
    print("Predictions completed. Processing results...")
    
    y_pred_RC = np.clip(np.round(y_pred), a_min = 0, a_max = 100)

    # Calculate error
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mae_RC = mean_absolute_error(y_test, y_pred_RC)
    rmse_RC = np.sqrt(mean_squared_error(y_test, y_pred_RC))
    
    print("Error calculations completed.")

    ErrorRate = {"MAE": mae,
                 "RMSE": rmse,
                 "MAE_RC": mae_RC,
                 "RMSE_RC": rmse_RC}
    
    # Store comparision result
    if StoreComparision:
        result = pd.DataFrame()
        result[f"Target {targetName}"] = y_test
        result[f"Predicted {targetName} RC"] = y_pred_RC
        result[f"Predicted {targetName} Original"] = y_pred
        result = result.sort_values(by=f"Target {targetName}", ascending=False)
        output_Path = os.path.join(output_dir, output_File_Name)
        result.to_csv(output_Path,index=False)

    if Beep:
        # Cross-platform beep sound
        if platform.system() == "Windows":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND)
        else:
            # For macOS and Linux
            os.system("afplay /System/Library/Sounds/Ping.aiff")

    return ErrorRate

def Print_ErrorRate(ErrorRate: dict[str,float]) -> None:
    print("GPR Error Rate with Original Prediction:")
    print("Mean Absolute Error (MAE):", ErrorRate["MAE"])
    print("Root Mean Squared Error (RMSE):", ErrorRate["RMSE"])
    print("\n")
    print("GPR Error Rate with Rounded Clipped Prediction:")
    print("Mean Absolute Error (MAE):", ErrorRate["MAE_RC"])
    print("Root Mean Squared Error (RMSE):", ErrorRate["RMSE_RC"])


def main():
    # TopFeatures = ['Median', 'Percentile_75', 't1/tpi', 'mfcc_2', 'tpp', 'sys_amp', 'tpi', 'InterquartileRange', 'mfcc_8', 'mfcc_6', 'foot_amp']
    TopFeatures = ['Median', 'Percentile_75', 'mean(t1/tpi)', 'mfcc_2', 'mean(tpp)', 'mean(sys_amp)', 'mean(tpi)', 'InterquartileRange', 'mfcc_8', 'mfcc_6', 'mean(foot_amp)']
    # targetName = "SpO2(mean)" # "RR(mean)"
    targetName = "RR(mean)"
    input_csv = "BIDMC_Regression/features/BIDMC_Segmented_features.csv"
    output_dir = "BIDMC_Regression/features"
    # ErrorRate = Run_GPR(input_csv, output_dir, targetName, TopFeatures)
    ErrorRate = Run_GPR(input_csv, targetName, TopFeatures, Beep=True, StoreComparision=True, output_dir=output_dir)
    Print_ErrorRate(ErrorRate)
    return 0


if __name__ == "__main__":
    main()