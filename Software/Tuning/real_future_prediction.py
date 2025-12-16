'''
This script is used to predict future SpO2 and RR for multiple time windows (10-300s)
Using Transformer Encoder to extract features from PPG sequences + XGBoost for predictions
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

# PyTorch for Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PPGSequenceDataset(Dataset):
    """Dataset for PPG sequences"""
    def __init__(self, sequences, targets=None):
        self.sequences = sequences  # Shape: (n_samples, seq_len, n_features)
        self.targets = targets  # Shape: (n_samples, n_targets) or None
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        if self.targets is not None:
            target = torch.FloatTensor(self.targets[idx])
            return sequence, target
        return sequence

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for processing PPG sequences
    """
    def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=3, 
                 feedforward_dim=512, dropout=0.1, max_seq_len=60):
        super(TransformerEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, embed_dim)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Pooling layers (for sequence to vector)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
        
        Returns:
            embedding: Sequence embedding (batch_size, embed_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project to embedding dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling (CLS token style)
        # x shape: (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, embed_dim)
        
        return x

class MultiWindowFuturePrediction:
    """
    Multi-window future prediction model for SpO2 and RR
    """
    
    def __init__(self, future_offsets=[10, 20, 30, 40, 50, 60, 90, 120, 180, 240, 300], 
                 seq_len=60, use_transformer=True,
                 embed_dim=128, num_heads=8, num_layers=3):
        """
        Initialize multi-window future prediction model with Transformer
        
        Args:
            future_offsets: List of prediction time offsets in seconds
            seq_len: Length of PPG sequence to use for input
            use_transformer: Whether to use Transformer encoder
            embed_dim: Transformer embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
        """
        self.future_offsets = sorted(future_offsets)
        self.max_offset = max(self.future_offsets)
        self.seq_len = seq_len
        self.use_transformer = use_transformer
        
        # Transformer parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Store Transformer model
        self.transformer = None
        self.input_dim = None
        
        # Store models for each time window
        self.models_spo2 = {}
        self.models_rr = {}
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def prepare_training_data(self, input_csv):
        """
        Prepare training data with PPG sequences for multiple time windows
        
        Args:
            input_csv: Input feature file path
        
        Returns:
            dict: Training data for each time window
        """
        print(f"Preparing training data for multiple time windows: {self.future_offsets}s...")
        
        # Read data
        df = pd.read_csv(input_csv)
        print(f"Original data length: {len(df)}")
        
        # Enhanced feature engineering for better long-term prediction
        df['SpO2_trend'] = df['SpO2(mean)'].diff().fillna(0)
        df['RR_trend'] = df['RR(mean)'].diff().fillna(0)
        
        # Multiple moving averages for different time scales
        df['SpO2_ma3'] = df['SpO2(mean)'].rolling(window=3, min_periods=1).mean()
        df['SpO2_ma5'] = df['SpO2(mean)'].rolling(window=5, min_periods=1).mean()
        df['SpO2_ma10'] = df['SpO2(mean)'].rolling(window=10, min_periods=1).mean()
        
        df['RR_ma3'] = df['RR(mean)'].rolling(window=3, min_periods=1).mean()
        df['RR_ma5'] = df['RR(mean)'].rolling(window=5, min_periods=1).mean()
        df['RR_ma10'] = df['RR(mean)'].rolling(window=10, min_periods=1).mean()
        
        # Volatility features
        df['SpO2_volatility'] = df['SpO2(mean)'].rolling(window=5, min_periods=1).std().fillna(0)
        df['RR_volatility'] = df['RR(mean)'].rolling(window=5, min_periods=1).std().fillna(0)
        
        # Momentum features (rate of change)
        df['SpO2_momentum'] = df['SpO2(mean)'].diff(2).fillna(0)
        df['RR_momentum'] = df['RR(mean)'].diff(2).fillna(0)
        
        # Get feature columns (exclude SpO2 and RR as they are targets)
        exclude_cols = ['SpO2(mean)', 'RR(mean)']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Store feature names for later use
        self.feature_names = feature_cols
        self.input_dim = len(feature_cols)
        print(f"Number of features: {self.input_dim}")
        
        # Prepare training data for each time window
        training_data = {}
        
        for offset in self.future_offsets:
            print(f"Preparing data for {offset}s prediction...")
            
            X_sequences = []  # Input PPG sequences
            y_spo2 = []  # Future SpO2 targets
            y_rr = []    # Future RR targets
            
            # Ensure sufficient data for sequence + future prediction
            max_index = len(df) - max(offset, self.seq_len)
            
            for i in range(max_index):
                # Create sequence from historical data
                start_idx = max(0, i - self.seq_len + 1)
                sequence_data = df.iloc[start_idx:i+1][feature_cols].values
                
                # Pad if sequence is shorter than seq_len
                if len(sequence_data) < self.seq_len:
                    padding = np.zeros((self.seq_len - len(sequence_data), len(feature_cols)))
                    sequence_data = np.vstack([padding, sequence_data])
                
                # Get future targets
                future_row = df.iloc[i + offset]
                future_spo2 = future_row['SpO2(mean)']
                future_rr = future_row['RR(mean)']
                
                X_sequences.append(sequence_data)
                y_spo2.append(future_spo2)
                y_rr.append(future_rr)
            
            training_data[offset] = {
                'X': X_sequences,
                'y_spo2': y_spo2,
                'y_rr': y_rr
            }
            
            print(f"  {offset}s samples: {len(X_sequences)}")
            print(f"  SpO2 range: {min(y_spo2):.2f} - {max(y_spo2):.2f}")
            print(f"  RR range: {min(y_rr):.2f} - {max(y_rr):.2f}")
        
        return training_data
    
    def _train_transformer(self, all_sequences):
        """
        Train the Transformer encoder on all PPG sequences
        
        Args:
            all_sequences: List of all sequences from training data
        """
        print("Training Transformer encoder...")
        
        # Initialize Transformer
        self.transformer = TransformerEncoder(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        ).to(device)
        
        # For this implementation, we'll use the Transformer in a self-supervised way
        # or train it as a feature extractor with reconstruction loss
        optimizer = optim.Adam(self.transformer.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create dataset
        X = np.array(all_sequences)
        dataset = PPGSequenceDataset(X)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Add a reconstruction head for training
        reconstruction_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.input_dim * self.seq_len)
        ).to(device)
        
        optimizer_full = optim.Adam(
            list(self.transformer.parameters()) + list(reconstruction_head.parameters()),
            lr=0.001
        )
        
        # Train for a few epochs to learn representations
        n_epochs = 15
        self.transformer.train()
        reconstruction_head.train()
        
        for epoch in range(n_epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(device)
                batch_size, seq_len, n_features = batch.shape
                
                # Forward pass through transformer
                embeddings = self.transformer(batch)
                
                # Reconstruct sequence from embedding
                reconstructed = reconstruction_head(embeddings)
                reconstructed = reconstructed.view(batch_size, self.seq_len, n_features)
                
                # Reconstruction loss
                reconstruction_loss = criterion(reconstructed, batch)
                
                # L2 regularization on embeddings to prevent overfitting
                l2_reg = 0.001 * torch.mean(embeddings ** 2)
                
                loss = reconstruction_loss + l2_reg
        
                optimizer_full.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
                optimizer_full.step()
                
                total_loss += loss.item()
            
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.transformer.eval()
        print("Transformer encoder trained!")
    
    def _extract_embeddings(self, sequences):
        """
        Extract embeddings from sequences using the Transformer
        
        Args:
            sequences: Input sequences (n_samples, seq_len, n_features)
        
        Returns:
            numpy array: Embeddings (n_samples, embed_dim)
        """
        self.transformer.eval()
        
        X = np.array(sequences)
        dataset = PPGSequenceDataset(X)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        embeddings_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                embeddings = self.transformer(batch)
                embeddings_list.append(embeddings.cpu().numpy())
        
        return np.vstack(embeddings_list)
    
    def train(self, training_data):
        """
        Train Transformer + XGBoost multi-window future prediction models
        
        Args:
            training_data: Dictionary containing training data for each time window
        """
        print("Starting multi-window future prediction model training with Transformer...")
        
        # Collect all sequences for Transformer training
        all_sequences = []
        for offset in self.future_offsets:
            all_sequences.extend(training_data[offset]['X'])
        
        # Train Transformer encoder
        self._train_transformer(all_sequences)
        
        # Extract embeddings and train XGBoost models for each time window
        for offset in self.future_offsets:
            print(f"Training XGBoost models for {offset}s prediction...")
            
            # Extract embeddings
            X_sequences = training_data[offset]['X']
            X_embeddings = self._extract_embeddings(X_sequences)
            
            y_spo2 = training_data[offset]['y_spo2']
            y_rr = training_data[offset]['y_rr']
            
            # Adaptive XGBoost parameters based on prediction window
            if offset <= 20:
                # Very short-term prediction (10-20s): faster learning, less regularization
                xgb_params = dict(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=0.5,
                    objective='reg:squarederror',
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1
                )
            elif offset <= 40:
                # Short-term prediction (30-40s): balanced parameters
                xgb_params = dict(
                    n_estimators=600,
                    max_depth=4,
                    learning_rate=0.06,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective='reg:squarederror',
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1
                )
            elif offset <= 60:
                # Medium-term prediction (50-60s): more regularization
                xgb_params = dict(
                    n_estimators=700,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.75,
                    colsample_bytree=0.75,
                    reg_lambda=1.5,
                    objective='reg:squarederror',
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1
                )
            elif offset <= 120:
                # Long-term prediction (90-120s): strong regularization
                xgb_params = dict(
                    n_estimators=800,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_lambda=2.5,
                    objective='reg:squarederror',
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1
                )
            else:
                # Very long-term prediction (180s+, 3-5 minutes): maximum regularization
                xgb_params = dict(
                    n_estimators=1000,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.65,
                    colsample_bytree=0.65,
                    reg_lambda=3.0,
                    objective='reg:squarederror',
                    random_state=42,
                    tree_method='hist',
                    n_jobs=-1
                )
            
            model_spo2 = XGBRegressor(**xgb_params)
            model_rr = XGBRegressor(**xgb_params)
            
            # Train XGBoost models using Transformer embeddings
            model_spo2.fit(X_embeddings, y_spo2)
            model_rr.fit(X_embeddings, y_rr)
            
            # Store models
            self.models_spo2[offset] = model_spo2
            self.models_rr[offset] = model_rr
            
            print(f"  {offset}s models trained successfully")
        
        self.is_trained = True
        print("Multi-window Transformer+XGBoost model training completed!")
    
    def predict(self, current_sequence):
        """
        Predict future SpO2 and RR for all time windows using Transformer embeddings
        
        Args:
            current_sequence: Current PPG sequence (seq_len, n_features) or list of features
        
        Returns:
            dict: Prediction results for all time windows
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Convert input to sequence if needed
        if isinstance(current_sequence, (list, dict)):
            # If single feature vector, convert to sequence
            if isinstance(current_sequence, dict):
                # Extract features in correct order
                feature_values = [current_sequence.get(f, 0.0) for f in self.feature_names]
            else:
                feature_values = current_sequence
            
            # Create sequence by repeating the current features
            sequence = np.tile(feature_values, (self.seq_len, 1))
        else:
            sequence = np.array(current_sequence)
        
        # Ensure correct shape
        if sequence.ndim == 1:
            sequence = sequence.reshape(1, -1)
        if sequence.shape[0] != self.seq_len:
            # Pad or truncate to seq_len
            if sequence.shape[0] < self.seq_len:
                padding = np.zeros((self.seq_len - sequence.shape[0], sequence.shape[1]))
                sequence = np.vstack([padding, sequence])
            else:
                sequence = sequence[-self.seq_len:]
        
        # Extract embedding using Transformer
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        self.transformer.eval()
        
        with torch.no_grad():
            embedding = self.transformer(sequence_tensor)
            embedding = embedding.cpu().numpy()
        
        # Predict for all time windows using XGBoost
        predictions = {}
        
        for offset in self.future_offsets:
            # Prediction using Transformer embeddings
            spo2_pred = float(self.models_spo2[offset].predict(embedding)[0])
            rr_pred = float(self.models_rr[offset].predict(embedding)[0])
            
            # Apply physiological constraints
            spo2_pred = np.clip(spo2_pred, 70, 100)
            rr_pred = np.clip(rr_pred, 8, 40)
            
            predictions[offset] = {
                'future_spo2': spo2_pred,
                'future_rr': rr_pred,
                'confidence_spo2': None,
                'confidence_rr': None
            }
        
        return predictions
    
    def evaluate(self, test_data):
        """
        Evaluate model performance for all time windows
        
        Args:
            test_data: Dictionary containing test data for each time window
        
        Returns:
            dict: Evaluation metrics for all time windows
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        evaluation_results = {}
        
        for offset in self.future_offsets:
            print(f"Evaluating {offset}s prediction...")
            
            # Prepare test data
            X_sequences = test_data[offset]['X']
            y_spo2_test = test_data[offset]['y_spo2']
            y_rr_test = test_data[offset]['y_rr']
            
            # Extract embeddings using Transformer
            X_test_embeddings = self._extract_embeddings(X_sequences)
            
            # Predict using Transformer embeddings + XGBoost
            spo2_pred = self.models_spo2[offset].predict(X_test_embeddings)
            rr_pred = self.models_rr[offset].predict(X_test_embeddings)
            
            # Calculate evaluation metrics
            spo2_mae = mean_absolute_error(y_spo2_test, spo2_pred)
            spo2_rmse = np.sqrt(mean_squared_error(y_spo2_test, spo2_pred))
            spo2_r2 = r2_score(y_spo2_test, spo2_pred)
            
            rr_mae = mean_absolute_error(y_rr_test, rr_pred)
            rr_rmse = np.sqrt(mean_squared_error(y_rr_test, rr_pred))
            rr_r2 = r2_score(y_rr_test, rr_pred)
            
            evaluation_results[offset] = {
                'SpO2_MAE': spo2_mae,
                'SpO2_RMSE': spo2_rmse,
                'SpO2_R2': spo2_r2,
                'RR_MAE': rr_mae,
                'RR_RMSE': rr_rmse,
                'RR_R2': rr_r2,
                'predictions': {
                    'spo2_pred': spo2_pred,
                    'rr_pred': rr_pred,
                    'spo2_true': y_spo2_test,
                    'rr_true': y_rr_test
                }
            }
        
        return evaluation_results
    
    def save_model(self, filepath):
        """
        Save trained multi-window model including Transformer
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        model_data = {
            'future_offsets': self.future_offsets,
            'max_offset': self.max_offset,
            'models_spo2': self.models_spo2,
            'models_rr': self.models_rr,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'seq_len': self.seq_len,
            'use_transformer': self.use_transformer,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'input_dim': self.input_dim
        }
        
        # Save Transformer state separately
        if self.use_transformer and self.transformer is not None:
            model_data['transformer_state'] = self.transformer.state_dict()
        
        joblib.dump(model_data, filepath)
        print(f"Multi-window Transformer+XGBoost model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained multi-window model including Transformer
        """
        model_data = joblib.load(filepath)
        
        self.future_offsets = model_data['future_offsets']
        self.max_offset = model_data['max_offset']
        self.models_spo2 = model_data['models_spo2']
        self.models_rr = model_data['models_rr']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        self.seq_len = model_data.get('seq_len', 60)
        self.use_transformer = model_data.get('use_transformer', True)
        
        # Load Transformer if present
        if 'transformer_state' in model_data:
            self.embed_dim = model_data.get('embed_dim', 128)
            self.num_heads = model_data.get('num_heads', 8)
            self.num_layers = model_data.get('num_layers', 3)
            self.input_dim = model_data.get('input_dim', len(self.feature_names))
            
            self.transformer = TransformerEncoder(
                input_dim=self.input_dim,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers
            ).to(device)
            self.transformer.load_state_dict(model_data['transformer_state'])
            self.transformer.eval()
        
        print(f"Multi-window Transformer+XGBoost model loaded from {filepath}")

def main():
    """
    Main function: demonstrate multi-window future prediction with Transformer + XGBoost
    """
    print("=== Multi-Window Future Prediction with Transformer + XGBoost ===")
    
    # Set parameters
    # Extended time windows: 10-60s (short-term), 90-300s (medium to long-term)
    future_offsets = [10, 20, 30, 40, 50, 60, 90, 120, 180, 240, 300]
    input_csv = "BIDMC_Regression/features/BIDMC_Segmented_features.csv"
    
    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"Error: File not found {input_csv}")
        print("Please ensure BIDMC_Segmented_features.csv exists")
        return
    
    # Initialize model
    model = MultiWindowFuturePrediction(future_offsets)
    
    # Prepare training data
    print("Preparing training data...")
    training_data = model.prepare_training_data(input_csv)
    
    # Split training and test data for each time window
    test_data = {}
    for offset in future_offsets:
        X = training_data[offset]['X']
        y_spo2 = training_data[offset]['y_spo2']
        y_rr = training_data[offset]['y_rr']
        
        X_train, X_test, y_spo2_train, y_spo2_test, y_rr_train, y_rr_test = train_test_split(
            X, y_spo2, y_rr, test_size=0.2, random_state=42
        )
        
        # Update training data with train split
        training_data[offset] = {
            'X': X_train,
            'y_spo2': y_spo2_train,
            'y_rr': y_rr_train
        }
        
        # Store test data
        test_data[offset] = {
            'X': X_test,
            'y_spo2': y_spo2_test,
            'y_rr': y_rr_test
        }
        
        print(f"{offset}s - Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    model.train(training_data)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    evaluation_results = model.evaluate(test_data)
    
    # Display results for each time window
    print(f"\n=== Multi-Window Prediction Performance Evaluation ===")
    print(f"{'Time Window':<12} {'SpO2 MAE':<10} {'SpO2 R²':<10} {'RR MAE':<10} {'RR R²':<10}")
    print("-" * 60)
    
    # Store results for analysis
    spo2_maes = []
    rr_maes = []
    
    for offset in future_offsets:
        eval_result = evaluation_results[offset]
        spo2_maes.append(eval_result['SpO2_MAE'])
        rr_maes.append(eval_result['RR_MAE'])
        print(f"{offset}s{'':<8} {eval_result['SpO2_MAE']:<10.4f} {eval_result['SpO2_R2']:<10.4f} "
              f"{eval_result['RR_MAE']:<10.4f} {eval_result['RR_R2']:<10.4f}")
    
    # Performance analysis
    print(f"\n=== Performance Analysis ===")
    spo2_maes = np.array(spo2_maes)
    rr_maes = np.array(rr_maes)
    
    print(f"SpO2 MAE - Best: {min(spo2_maes):.4f} ({future_offsets[np.argmin(spo2_maes)]}s), "
          f"Worst: {max(spo2_maes):.4f} ({future_offsets[np.argmax(spo2_maes)]}s)")
    print(f"RR MAE - Best: {min(rr_maes):.4f} ({future_offsets[np.argmin(rr_maes)]}s), "
          f"Worst: {max(rr_maes):.4f} ({future_offsets[np.argmax(rr_maes)]}s)")
    
    # Identify problematic windows
    spo2_threshold = np.mean(spo2_maes) + np.std(spo2_maes)
    rr_threshold = np.mean(rr_maes) + np.std(rr_maes)
    
    problematic_windows = []
    for i, offset in enumerate(future_offsets):
        if spo2_maes[i] > spo2_threshold or rr_maes[i] > rr_threshold:
            problematic_windows.append(offset)
    
    if problematic_windows:
        print(f"Problematic windows (MAE > mean+std): {problematic_windows}")
        print("These windows may need special attention or different modeling approaches.")
    
    # Show prediction examples for different time windows
    print(f"\n=== Multi-Window Prediction Examples ===")
    
    # Use first time window's test data for examples (all time windows have same input features)
    first_offset = future_offsets[0]
    X_test_examples = test_data[first_offset]['X']
    
    for i in range(min(3, len(X_test_examples))):
        print(f"\nSample {i+1}:")
        
        # For demonstration, create some mock historical data
        # In real usage, this would come from previous measurements
        historical_data = []
        if i > 0:
            # Use previous samples as historical context
            for j in range(max(1, i-5), i):
                if j < len(X_test_examples):
                    historical_data.append(X_test_examples[j])
        
        predictions = model.predict(X_test_examples[i])
        
        for offset in future_offsets:
            pred = predictions[offset]
            # Get true values for this specific time window
            true_spo2 = test_data[offset]['y_spo2'][i]
            true_rr = test_data[offset]['y_rr'][i]
            
            print(f"  {offset}s prediction:")
            
            # SpO2 prediction with confidence level (N/A for XGBoost)
            if pred['confidence_spo2'] is None:
                print(f"    SpO2: {pred['future_spo2']:.2f}% (confidence: N/A)")
            else:
                spo2_conf_level = "High" if pred['confidence_spo2'] < 0.5 else "Medium" if pred['confidence_spo2'] < 1.0 else "Low"
                print(f"    SpO2: {pred['future_spo2']:.2f}% (confidence: {pred['confidence_spo2']:.3f} - {spo2_conf_level})")
            print(f"    True SpO2: {true_spo2:.2f}% (error: {abs(true_spo2 - pred['future_spo2']):.2f}%)")
            
            # RR prediction with confidence level (N/A for XGBoost)
            if pred['confidence_rr'] is None:
                print(f"    RR: {pred['future_rr']:.2f} breaths/min (confidence: N/A)")
            else:
                rr_conf_level = "High" if pred['confidence_rr'] < 0.5 else "Medium" if pred['confidence_rr'] < 1.0 else "Low"
                print(f"    RR: {pred['future_rr']:.2f} breaths/min (confidence: {pred['confidence_rr']:.3f} - {rr_conf_level})")
            print(f"    True RR: {true_rr:.2f} breaths/min (error: {abs(true_rr - pred['future_rr']):.2f})")
            print()
    
    # Save model with a concise name
    model_path = f"transformer_xgboost_model_extended_10-300s.pkl"
    model.save_model(model_path)
    
    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()