import optuna
import argparse
import pandas as pd
import os
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.gaussian_process.kernels import WhiteKernel


def Use_weighted_P3(input_csv: str, weights: list[float] = [0.33, 0.33, 0.33]) -> pd.DataFrame:
    target_set = ["SpO2(mean)", "RR(mean)", "wave nunmber", "segment nunmber"]
    block_length = 31
    df = pd.read_csv(input_csv)
    features = [feature for feature in df.columns if feature not in target_set]
    
    result = []
    for i in range(54):
        start = i*block_length
        end = start + block_length
        curr_seg = df.iloc[start:end]

        for col in features:
            # calculate weighted average
            weighted_sum =(curr_seg[col].shift(1)*weights[0] 
                         + curr_seg[col].shift(2)*weights[1]
                         + curr_seg[col].shift(3)*weights[2])
            curr_seg.loc[:,col] = weighted_sum
        
        curr_seg = curr_seg.iloc[3:]

        result.append(curr_seg)

    result_pd = pd.concat(result, ignore_index=True)

    return result_pd

def Use_weighted_C3(input_csv: str, weights: list[float] = [0.33, 0.33, 0.33]) -> pd.DataFrame:
    target_set = ["SpO2(mean)", "RR(mean)", "wave nunmber", "segment nunmber"]
    # weights = [0.5, 0.3, 0.2]
    # weights = [0.4, 0.3, 0.3]
    block_length = 31
    df = pd.read_csv(input_csv)
    features = [feature for feature in df.columns if feature not in target_set]
    
    result = []
    for i in range(54):
        start = i*block_length
        end = start + block_length
        curr_seg = df.iloc[start:end]

        for col in features:

            weighted_sum =(curr_seg[col]*weights[0] 
                         + curr_seg[col].shift(1)*weights[1]
                         + curr_seg[col].shift(2)*weights[2])
            curr_seg.loc[:,col] = weighted_sum
        
        curr_seg = curr_seg.iloc[2:]

        result.append(curr_seg)

    result_pd = pd.concat(result, ignore_index=True)

    return result_pd

def Run_ReliefF(input_pd: pd.DataFrame, targetName: str, featureNum: int =11) -> list[str]:

    # Load the aggregated features CSV (adjust the file path as needed)
    aggregated_df = input_pd

    # Separate the target variable
    target = aggregated_df[targetName]
    # y = pd.qcut(target, q=2, labels=False)
    y = np.round(target)

    X = aggregated_df.drop(columns=["wave nunmber", "segment nunmber", "SpO2(mean)", "RR(mean)"])
    # print(X.shape)

    # Standardize the features (important for distance-based methods like ReliefF)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # relief = ReliefF(n_neighbors=50, n_features_to_select=X.shape[1])
    relief = ReliefF(n_neighbors=10, n_features_to_select=X.shape[1])

    # Fit ReliefF
    relief.fit(X_scaled, y)

    # Get the importance scores for each feature.
    feature_scores = relief.feature_importances_


    scores_df = pd.DataFrame({
        "Feature": X.columns,
        "Score": feature_scores
    }).sort_values(by="Score", ascending=False)
    
    top_features = scores_df.head(featureNum) if featureNum != 0 else scores_df
    return top_features["Feature"]

def Run_GPR(input_pd: pd.DataFrame,
            targetName: str,
            TopFeatures: list[str]) -> float:
    # Load the aggregated features CSV (adjust the file path as needed)
    aggregated_df = input_pd

    # Separate the target variable
    target = aggregated_df[targetName]
    # y = pd.qcut(target, q=2, labels=False)
    y = np.round(target)

    X = aggregated_df.drop(columns=["wave nunmber", "segment nunmber", "SpO2(mean)", "RR(mean)"])
    X_selected = X[TopFeatures]

    
    scaler_selected = StandardScaler()
    X_selected_scaled = scaler_selected.fit_transform(X_selected)


    X_train, X_test, y_train, y_test = train_test_split(X_selected_scaled, y, test_size=0.2, random_state=42)
    
    # kernel = C(1, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2))
    noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-15, 1e5)             # noise_level_bounds modified from 1e-10 to 1e-15
    )
    kernel = C(1, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2)) + noise_kernel

    # Initialize and fit the Gaussian Process Regressor.
    # n_restarts_optimizer is set to 10 to ensure robust kernel parameter optimization.
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=1e-2, normalize_y=True)
    gpr.fit(X_train, y_train)

    y_pred, sigma = gpr.predict(X_test, return_std=True)
    y_pred_RC = np.clip(np.round(y_pred), a_min = 0, a_max = 100)

    # Calculate error
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mae_RC = mean_absolute_error(y_test, y_pred_RC)
    rmse_RC = np.sqrt(mean_squared_error(y_test, y_pred_RC))

    return rmse_RC

"""
All functions above have the same function as in 'transform_data.py', 'ReliefF.py' and 'GPR.py'.
Inputs and outputs are modified to fit with optuna
"""

def objective(trial):
    """
    Enabe part1 or part2 to tune weights or feature number.
    May change targetName to SpO2 to tune for different target.
    """
    input_csv = r"D:\桌面\ENG573\Sp25_Project\BIDMC_Regression\features\BIDMC_Segmented_features.csv"
    targetName = "RR(mean)"

    # Part1: tune weights used in weighted average
    # make sure sum(weights) = 1
    w1 = trial.suggest_float("w1", 0, 1)
    remaining = 1 - w1
    w2 = trial.suggest_float("w2", 0, remaining)
    w3 = remaining - w2
    weights = [w1, w2, w3]

    # Part2: tune number of features used 
    # n_features = trial.suggest_int("n_features", 1, 131)

    weighted_data = Use_weighted_C3(input_csv, weights)
    
    selected_features = Run_ReliefF(weighted_data, targetName, featureNum=11)
    
    error_rate = Run_GPR(weighted_data, targetName, selected_features)
    
    return error_rate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna.db?check_same_thread=False",
        study_name="weight_CW_F11",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=args.n_trials)
