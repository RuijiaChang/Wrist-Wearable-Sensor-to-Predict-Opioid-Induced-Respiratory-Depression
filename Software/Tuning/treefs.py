import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

def Run_TreeFS(input_csv: str,
               targetName: str,
               n_top: int = 8,
               n_estimators: int = 100,
               random_state: int = 0
              ) -> pd.DataFrame:
    """
    Use a tree-based model (ExtraTreesRegressor) to compute feature importances
    and return the top-n features sorted by importance.
    """
    # 1. Load aggregated features
    df = pd.read_csv(input_csv)
    y = df[targetName].values

    # 2. Drop identifiers and other targets to form X
    drop_cols = ["wave nunmber", "segment nunmber", "SpO2(mean)", "RR(mean)"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feature_names = X_df.columns.tolist()
    X = X_df.values

    # 3. Scale features (trees don't strictly need it, but it doesn't hurt)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Fit an ExtraTreesRegressor
    tree = ExtraTreesRegressor(n_estimators=n_estimators,
                               random_state=random_state)
    tree.fit(X_scaled, y)

    # 5. Grab the importances and build a DataFrame
    importances = tree.feature_importances_
    scores_df = pd.DataFrame({
        "Feature": feature_names,
        "Score": importances
    }).sort_values(by="Score", ascending=False)

    # 6. Return the top-n
    return scores_df.head(n_top)