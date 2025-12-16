import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

def Run_LASSO(input_csv: str,
              targetName: str,
              n_top: int = 21,
              cv: int = 10,
              random_state: int = 0
             ) -> pd.DataFrame:

    # 1. Load data
    df = pd.read_csv(input_csv)
    y = df[targetName].values
    y = np.round(y)
    # 2. Build feature matrix X (drop identifiers and other targets)
    drop_cols = ["wave nunmber", "segment nunmber", "SpO2(mean)", "RR(mean)"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feature_names = X_df.columns.tolist()
    X = X_df.values

    # 3. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Fit LassoCV (automatically tunes alpha via CV)
    #    We use a range of alphas on a log scale.
    lasso = LassoCV(
        alphas=np.logspace(-3, -1, 30),
        cv=cv,
        max_iter=85000,
        tol=1e-3,
        precompute=True,
        random_state=random_state
        ).fit(X_scaled, y)

    # 5. Extract absolute coefficients as importance scores
    coefs = np.abs(lasso.coef_)
    scores_df = pd.DataFrame({
        "Feature": feature_names,
        "Score": coefs
    }).sort_values(by="Score", ascending=False)

    # 6. Take top-n (and drop zero‑coef features automatically if n_top exceeds nonzeros)
    top_features = scores_df.head(n_top).reset_index(drop=True)
    return top_features


def Print_TopLasso(top_features: pd.DataFrame) -> None:
    print(f"\nTop {len(top_features)} Features by LASSO Coefficient:")
    print(top_features.to_string(index=False))

    names = top_features["Feature"].tolist()
    print("\nList of Top Features:")
    print(names)