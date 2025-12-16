'''
Inference pipeline for current and future SpO2/RR prediction using saved models.
Supports:
- Current moment prediction via GPR models (optional)
- Future multi-window prediction via Transformer+XGBoost model
- Input via single feature vector (JSON) or historical CSV data
- Output results as JSON (to file or stdout)

Usage Example:
python Software/infer.py \
  --current_spo2_model Software/Checkpoints/current_spo2_model.pkl \
  --current_rr_model Software/Checkpoints/current_rr_model.pkl \
  --future_model Software/Checkpoints/transformer_xgboost_model_extended_10-300s.pkl \
  --input_json Software/Test_Infer/test_input_features.json \
  --output Software/Test_Infer/test_prediction_output.json

'''

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd


class FutureModelWrapper:
    """
    Thin wrapper that loads the saved Transformer+XGBoost multi-window model
    (produced by `Software/Tuning/real_future_prediction.py`) and exposes a
    simple predict() API for a single sequence or feature vector.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_data: Optional[Dict[str, Any]] = None
        self.future_offsets: List[int] = []
        self.feature_names: List[str] = []
        self.seq_len: int = 60
        self.use_torch: bool = False

    def load(self) -> None:
        self.model_data = joblib.load(self.model_path)
        self.future_offsets = self.model_data["future_offsets"]
        self.feature_names = self.model_data["feature_names"]
        self.seq_len = self.model_data.get("seq_len", 60)

        # Lazy import torch only if transformer_state is present
        self.use_torch = "transformer_state" in self.model_data
        if self.use_torch:
            import torch
            import torch.nn as nn

            embed_dim = self.model_data.get("embed_dim", 128)
            num_heads = self.model_data.get("num_heads", 8)
            num_layers = self.model_data.get("num_layers", 3)
            input_dim = self.model_data.get("input_dim", len(self.feature_names))

            class TransformerEncoder(nn.Module):
                def __init__(self, input_dim: int, embed_dim: int, num_heads: int, num_layers: int, max_seq_len: int = 60):
                    super().__init__()
                    self.embed_dim = embed_dim
                    self.input_projection = nn.Linear(input_dim, embed_dim)
                    self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=512,
                        dropout=0.1,
                        batch_first=True,
                    )
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.pool = nn.AdaptiveAvgPool1d(1)
                    self.dropout = nn.Dropout(0.1)

                def forward(self, x):
                    # x: (batch, seq_len, input_dim)
                    x = self.input_projection(x)
                    seq_len = x.shape[1]
                    x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
                    x = self.dropout(x)
                    x = self.transformer(x)
                    x = x.transpose(1, 2)
                    x = self.pool(x).squeeze(-1)
                    return x

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._torch = torch
            self._device = device
            self._transformer = TransformerEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_len=self.seq_len,
            ).to(device)
            self._transformer.load_state_dict(self.model_data["transformer_state"])
            self._transformer.eval()

        # XGBoost models
        self.models_spo2: Dict[int, Any] = self.model_data["models_spo2"]
        self.models_rr: Dict[int, Any] = self.model_data["models_rr"]

    def _sequence_from_vector(self, feature_vector: List[float]) -> np.ndarray:
        seq = np.tile(np.array(feature_vector, dtype=float), (self.seq_len, 1))
        return seq

    def _sequence_from_frame_tail(self, df: pd.DataFrame) -> np.ndarray:
        # Ensure feature columns present; missing columns filled with 0.0
        available = [c for c in self.feature_names if c in df.columns]
        missing = [c for c in self.feature_names if c not in df.columns]
        frame = df[available].copy()
        for m in missing:
            frame[m] = 0.0
        # Reorder
        frame = frame[self.feature_names]
        # Take last seq_len rows (pad at top if not enough)
        values = frame.tail(self.seq_len).to_numpy(dtype=float)
        if values.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - values.shape[0], values.shape[1]), dtype=float)
            values = np.vstack([pad, values])
        return values

    def _embedding(self, sequence: np.ndarray) -> np.ndarray:
        if not self.use_torch:
            raise RuntimeError("Transformer state not found in model; cannot compute embeddings")
        torch = self._torch
        device = self._device
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            emb = self._transformer(x).cpu().numpy()
        return emb

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        input_data supports two forms:
        - {"vector": {feature_name: value, ...}}
        - {"csv": {"path": "...", "id_column": optional_id_col, "row_selector": optional}}
          If CSV is provided without selector, the last row is used to form the tail sequence.
        """

        if "vector" in input_data:
            vec_dict: Dict[str, float] = input_data["vector"]
            feature_vector = [float(vec_dict.get(f, 0.0)) for f in self.feature_names]
            sequence = self._sequence_from_vector(feature_vector)
        elif "csv" in input_data:
            csv_info = input_data["csv"]
            path = csv_info["path"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV not found: {path}")
            df = pd.read_csv(path)
            # Optional: filter/select specific rows by id or expression
            if "id_column" in csv_info and "id_value" in csv_info:
                id_col = csv_info["id_column"]
                id_val = csv_info["id_value"]
                df = df[df[id_col] == id_val]
            sequence = self._sequence_from_frame_tail(df)
        else:
            raise ValueError("input_data must contain either 'vector' or 'csv'")

        emb = self._embedding(sequence)

        results: Dict[str, Any] = {"predictions": {}}
        for offset in self.future_offsets:
            spo2 = float(self.models_spo2[offset].predict(emb)[0])
            rr = float(self.models_rr[offset].predict(emb)[0])
            # Clip to physiological ranges
            spo2 = float(np.clip(spo2, 70, 100))
            rr = float(np.clip(rr, 8, 40))
            results["predictions"][int(offset)] = {
                "future_spo2": spo2,
                "future_rr": rr,
            }
        return results


def load_optional_current_model(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Current model not found: {path}")
    return joblib.load(path)


def predict_current(models: Dict[str, Any], feature_names: List[str], vector: Dict[str, float]) -> Dict[str, float]:
    """
    Predict current SpO2 and RR using GPR models.
    Models can be either:
    - Direct sklearn models (legacy)
    - Model bundles with keys: 'model', 'scaler', 'features' (from train_and_save_gpr_models.py)
    """
    out: Dict[str, float] = {}
    
    # SpO2 prediction
    if models.get("spo2") is not None:
        try:
            model_bundle = models["spo2"]
            # Check if it's a model bundle (dict with 'model' key) or direct model
            if isinstance(model_bundle, dict) and 'model' in model_bundle:
                # GPR model bundle: extract features, scale, predict
                gpr_model = model_bundle['model']
                scaler = model_bundle['scaler']
                required_features = model_bundle['features']
                
                # Build feature vector in correct order
                X = np.array([[float(vector.get(f, 0.0)) for f in required_features]], dtype=float)
                X_scaled = scaler.transform(X)
                pred = gpr_model.predict(X_scaled, return_std=False)[0]
                out["current_spo2"] = float(np.clip(np.round(pred), 0, 100))
            else:
                # Legacy: direct model (assume it expects features in feature_names order)
                X = np.array([[float(vector.get(f, 0.0)) for f in feature_names]], dtype=float)
                out["current_spo2"] = float(model_bundle.predict(X)[0])
        except Exception as e:
            print(f"Warning: SpO2 prediction failed: {e}", file=sys.stderr)
    
    # RR prediction
    if models.get("rr") is not None:
        try:
            model_bundle = models["rr"]
            # Check if it's a model bundle (dict with 'model' key) or direct model
            if isinstance(model_bundle, dict) and 'model' in model_bundle:
                # GPR model bundle: extract features, scale, predict
                gpr_model = model_bundle['model']
                scaler = model_bundle['scaler']
                required_features = model_bundle['features']
                
                # Build feature vector in correct order
                X = np.array([[float(vector.get(f, 0.0)) for f in required_features]], dtype=float)
                X_scaled = scaler.transform(X)
                pred = gpr_model.predict(X_scaled, return_std=False)[0]
                out["current_rr"] = float(np.clip(np.round(pred), 8, 40))
            else:
                # Legacy: direct model (assume it expects features in feature_names order)
                X = np.array([[float(vector.get(f, 0.0)) for f in feature_names]], dtype=float)
                out["current_rr"] = float(model_bundle.predict(X)[0])
        except Exception as e:
            print(f"Warning: RR prediction failed: {e}", file=sys.stderr)
    
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified inference: current and future SpO2/RR predictions. "
        "Input: current moment features (single feature vector) or historical CSV data."
    )
    parser.add_argument("--future_model", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "transformer_xgboost_model_extended_10-300s.pkl")), help="Path to saved Transformer+XGBoost multi-window model .pkl")
    parser.add_argument("--input_json", type=str, help="Path to JSON file with current moment feature vector: {\"vector\": {feature_name: value, ...}}. This represents the current moment's signal features.")
    parser.add_argument("--input_csv", type=str, help="Path to CSV file containing feature history. For future prediction: last seq_len rows are used as sequence. For current prediction: last row is used as current moment features.")
    parser.add_argument("--current_spo2_model", type=str, default=None, help="Optional path to current SpO2 GPR model .pkl (from train_and_save_gpr_models.py)")
    parser.add_argument("--current_rr_model", type=str, default=None, help="Optional path to current RR GPR model .pkl (from train_and_save_gpr_models.py)")
    parser.add_argument("--output", type=str, default=None, help="Path to save prediction results as JSON file. If not specified, results are printed to stdout.")
    parser.add_argument("--print_pretty", action="store_true", help="Pretty-print JSON output (only used when --output is not specified)")

    args = parser.parse_args()

    # Load future model
    future_model = FutureModelWrapper(args.future_model)
    future_model.load()

    # Build input payload
    input_payload: Dict[str, Any]
    if args.input_json:
        with open(args.input_json, "r") as f:
            data = json.load(f)
        if "vector" not in data:
            raise ValueError("JSON must contain a 'vector' object with feature_name: value")
        input_payload = {"vector": data["vector"]}
    elif args.input_csv:
        input_payload = {"csv": {"path": args.input_csv}}
    else:
        print("Error: Provide either --input_json or --input_csv", file=sys.stderr)
        return 2

    # Run future predictions
    result = future_model.predict(input_payload)

    # Optional current predictions (require a single vector - current moment features)
    if args.current_spo2_model or args.current_rr_model:
        # Extract current moment vector from input
        if "vector" in input_payload:
            # Direct vector input (current moment features)
            current_vector = input_payload["vector"]
        elif "csv" in input_payload:
            # Extract last row from CSV as current moment features
            csv_path = input_payload["csv"]["path"]
            df = pd.read_csv(csv_path)
            # Get last row and convert to dict
            last_row = df.iloc[-1]
            current_vector = {col: float(last_row[col]) if pd.notna(last_row[col]) else 0.0 
                            for col in df.columns if col not in ["wave nunmber", "segment nunmber"]}
        else:
            current_vector = None
        
        if current_vector:
            models = {
                "spo2": load_optional_current_model(args.current_spo2_model),
                "rr": load_optional_current_model(args.current_rr_model),
            }
            current_out = predict_current(models, future_model.feature_names, current_vector)  # type: ignore[index]
            result["current"] = current_out
        else:
            result["current"] = {
                "warning": "Could not extract current moment features for prediction",
            }

    # Save or print results
    if args.output:
        # Save to JSON file
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Prediction results saved to: {args.output}", file=sys.stderr)
    else:
        # Print to stdout
        if args.print_pretty:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())


