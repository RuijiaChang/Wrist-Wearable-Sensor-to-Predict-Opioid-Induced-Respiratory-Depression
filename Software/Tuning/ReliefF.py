import pandas as pd
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
import numpy as np

def Run_ReliefF(input_csv: str, targetName: str, featureNum: int =11) -> pd.DataFrame:

    # Load the aggregated features CSV (adjust the file path as needed)
    aggregated_df = pd.read_csv(input_csv)

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

    # print("ReliefF Feature Importance Scores:")
    # print(scores_df)
    
    top_features = scores_df.head(featureNum) if featureNum != 0 else scores_df
    return top_features

def Print_TopFeature(top_features: pd.DataFrame) -> None:
    featureNum = len(top_features)
    print(f"\nTop {featureNum} Selected Features:")
    print(top_features)

    top_feature_names = top_features["Feature"].tolist()
    print(f"\nList of Top {featureNum} Features:")
    print(top_feature_names)


def main():
    input_csv = "BIDMC_Regression/features/BIDMC_Segmented_features.csv"
    target = "SpO2(mean)"
    TopFeatures = Run_ReliefF(input_csv, target)
    Print_TopFeature(TopFeatures)
    return 0


if __name__ == "__main__":
    main()