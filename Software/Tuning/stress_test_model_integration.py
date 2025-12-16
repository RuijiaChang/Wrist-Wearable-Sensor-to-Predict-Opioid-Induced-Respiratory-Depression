'''
Integration of Stress Testing Framework (stress_testing_framework.py) with Prediction Models
Tests model robustness under various data quality conditions
GPR model and future prediction model

Usage:
cd Software
python Tuning/stress_test_model_integration.py \
    --test_both \
    --model Checkpoints/transformer_xgboost_model_extended_10-300s.pkl \
    --gpr_spo2 Checkpoints/current_spo2_model.pkl \
    --gpr_rr Checkpoints/current_rr_model.pkl \
    --synthetic \
    --duration 30.0 \
    --visualize
'''

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Add Software directory to path for imports
software_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, software_dir)

from Tuning.stress_testing_framework import PPGStressTester, generate_synthetic_ppg_signal, load_ppg_signal_from_csv
from BIDMC_Feature_Aggregation.BIDMC_Preprocess import butter_filter, VMD_deMA
from BIDMC_Feature_Aggregation.BIDMIC_Aggregation import wave_segmentation
from BIDMC_Feature_Aggregation.BIDMC_FE_statistical import extract_statistical_features
from BIDMC_Feature_Aggregation.BIDMC_FE_freq import frequency_features
from BIDMC_Feature_Aggregation.BIDMC_FE_time import extract_key_time_domain_features
from BIDMC_Feature_Aggregation.BIDMC_FE_derivative import derivative_features
from infer import FutureModelWrapper

import neurokit2 as nk


class ModelStressTester:
    """
    Integrates stress testing with prediction models to evaluate robustness
    """
    
    def __init__(self, 
                 future_model_path: Optional[str] = None,
                 sampling_rate: int = 125,
                 segment_length: float = 30.0,
                 segment_distance: float = 15.0):
        """
        Initialize model stress tester
        
        Args:
            future_model_path: Path to saved Transformer+XGBoost model (optional if only testing GPR)
            sampling_rate: Sampling rate in Hz
            segment_length: Length of segments for feature extraction (seconds)
            segment_distance: Distance between segments (seconds)
        """
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.segment_distance = segment_distance
        self.samples_per_seg = int(segment_length * sampling_rate)
        self.step_samples = int(segment_distance * sampling_rate)
        
        # Initialize stress tester
        self.stress_tester = PPGStressTester(sampling_rate=sampling_rate)
        
        # Load prediction model (optional)
        self.model = None
        if future_model_path:
            print(f"Loading prediction model from: {future_model_path}")
            self.model = FutureModelWrapper(future_model_path)
            self.model.load()
            print("Model loaded successfully")
        
    def extract_features_from_signal(self, ppg_signal: np.ndarray, 
                                     required_features: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract features from PPG signal (same pipeline as training)
        
        Args:
            ppg_signal: PPG signal array
            required_features: Optional list of required feature names (if None, uses model.feature_names)
        
        Returns:
            Dictionary of extracted features
        """
        # Preprocessing (same as training pipeline)
        clean_PLETH = VMD_deMA(butter_filter(ppg_signal, freq=self.sampling_rate))
        
        # Segmentation
        try:
            segList, idxList = wave_segmentation(clean_PLETH)
        except:
            # If segmentation fails, return default features
            return self._get_default_features(required_features)
        
        if len(segList) == 0:
            return self._get_default_features(required_features)
        
        # Extract features
        try:
            FE_stat = extract_statistical_features(clean_PLETH)
            FE_freq = frequency_features(clean_PLETH, freq=self.sampling_rate)
            FE_time = extract_key_time_domain_features(segList, idxList, sampling_rate=self.sampling_rate)
            FE_deriv = derivative_features(segList)
            
            # Combine all features
            features = {**FE_stat, **FE_freq, **FE_time, **FE_deriv}
            
            # Ensure all required features are present
            if required_features is None:
                if self.model is not None:
                    required_features = self.model.feature_names
                else:
                    # If no model and no required features, return all extracted features
                    return features
            
            for feature_name in required_features:
                if feature_name not in features:
                    features[feature_name] = 0.0
            
            return features
        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            return self._get_default_features(required_features)
    
    def _get_default_features(self, required_features: Optional[List[str]] = None) -> Dict[str, float]:
        """Return default feature values (zeros)"""
        if required_features is None:
            if self.model is not None:
                required_features = self.model.feature_names
            else:
                return {}
        return {name: 0.0 for name in required_features}
    
    def test_model_under_stress(self, 
                                original_signal: np.ndarray,
                                stress_scenario: Dict[str, any],
                                true_spo2: Optional[float] = None,
                                true_rr: Optional[float] = None) -> Dict[str, any]:
        """
        Test future prediction model under stress conditions
        
        Args:
            original_signal: Original clean PPG signal
            stress_scenario: Stress scenario dictionary
            true_spo2: True SpO2 value (if known, for evaluation)
            true_rr: True RR value (if known, for evaluation)
        
        Returns:
            Dictionary with test results
        """
        if self.model is None:
            return {
                'scenario_name': stress_scenario.get('name', 'Unknown'),
                'success': False,
                'error': 'Future prediction model not loaded',
                'metadata': {}
            }
        
        # Apply stress to signal
        stressed_signal, metadata = self.stress_tester.apply_stress_scenario(
            original_signal, stress_scenario
        )
        
        # Extract features from stressed signal
        try:
            features = self.extract_features_from_signal(stressed_signal)
        except Exception as e:
            return {
                'scenario_name': stress_scenario.get('name', 'Unknown'),
                'success': False,
                'error': str(e),
                'metadata': metadata
            }
        
        # Make predictions
        try:
            predictions = self.model.predict({'vector': features})
        except Exception as e:
            return {
                'scenario_name': stress_scenario.get('name', 'Unknown'),
                'success': False,
                'error': str(e),
                'metadata': metadata
            }
        
        # Prepare results
        result = {
            'scenario_name': stress_scenario.get('name', 'Unknown'),
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'features_extracted': len([v for v in features.values() if v != 0.0])
        }
        
        # Add evaluation metrics if true values provided
        if true_spo2 is not None or true_rr is not None:
            result['evaluation'] = {}
            
            # Evaluate for each prediction window
            for offset, pred in predictions['predictions'].items():
                eval_dict = {}
                
                if true_spo2 is not None:
                    spo2_error = abs(pred['future_spo2'] - true_spo2)
                    spo2_error_percent = (spo2_error / true_spo2) * 100 if true_spo2 > 0 else 0
                    eval_dict['spo2_error'] = float(spo2_error)
                    eval_dict['spo2_error_percent'] = float(spo2_error_percent)
                
                if true_rr is not None:
                    rr_error = abs(pred['future_rr'] - true_rr)
                    rr_error_percent = (rr_error / true_rr) * 100 if true_rr > 0 else 0
                    eval_dict['rr_error'] = float(rr_error)
                    eval_dict['rr_error_percent'] = float(rr_error_percent)
                
                result['evaluation'][str(offset)] = eval_dict
        
        return result
    
    def run_comprehensive_stress_test(self,
                                     test_signal: np.ndarray,
                                     true_spo2: Optional[float] = None,
                                     true_rr: Optional[float] = None) -> pd.DataFrame:
        """
        Run comprehensive stress test with all scenarios
        
        Args:
            test_signal: Test PPG signal
            true_spo2: True SpO2 value (optional)
            true_rr: True RR value (optional)
        
        Returns:
            DataFrame with test results
        """
        scenarios = self.stress_tester.generate_test_scenarios()
        results = []
        
        print(f"\nRunning comprehensive stress test with {len(scenarios)} scenarios...")
        print("=" * 60)
        
        for i, scenario in enumerate(scenarios):
            print(f"\n[{i+1}/{len(scenarios)}] Testing: {scenario['name']}")
            
            result = self.test_model_under_stress(
                test_signal, scenario, true_spo2, true_rr
            )
            
            if result['success']:
                # Extract key metrics for summary
                summary = {
                    'scenario': result['scenario_name'],
                    'success': True,
                    'features_extracted': result['features_extracted']
                }
                
                # Add prediction values for first few time windows
                if 'predictions' in result and 'predictions' in result['predictions']:
                    preds = result['predictions']['predictions']
                    # Get first few offsets
                    offsets = sorted([int(k) for k in preds.keys()])[:5]
                    for offset in offsets:
                        if str(offset) in preds:
                            summary[f'spo2_{offset}s'] = preds[str(offset)]['future_spo2']
                            summary[f'rr_{offset}s'] = preds[str(offset)]['future_rr']
                
                # Add evaluation metrics if available
                if 'evaluation' in result:
                    eval_data = result['evaluation']
                    # Get first offset for summary
                    first_offset = sorted([int(k) for k in eval_data.keys()])[0] if eval_data else None
                    if first_offset and str(first_offset) in eval_data:
                        eval_first = eval_data[str(first_offset)]
                        if 'spo2_error' in eval_first:
                            summary['spo2_error'] = eval_first['spo2_error']
                        if 'rr_error' in eval_first:
                            summary['rr_error'] = eval_first['rr_error']
                
                results.append(summary)
                print(f"  ✓ Success - Features: {result['features_extracted']}")
            else:
                results.append({
                    'scenario': result['scenario_name'],
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
                print(f"  ✗ Failed - {result.get('error', 'Unknown error')}")
        
        return pd.DataFrame(results)
    
    def test_gpr_models_under_stress(self,
                                    original_signal: np.ndarray,
                                    stress_scenario: Dict[str, any],
                                    gpr_spo2_path: Optional[str] = None,
                                    gpr_rr_path: Optional[str] = None,
                                    true_spo2: Optional[float] = None,
                                    true_rr: Optional[float] = None) -> Dict[str, any]:
        """
        Test GPR models (current prediction) under stress conditions
        
        Args:
            original_signal: Original clean PPG signal
            stress_scenario: Stress scenario dictionary
            gpr_spo2_path: Path to GPR SpO2 model (optional)
            gpr_rr_path: Path to GPR RR model (optional)
            true_spo2: True SpO2 value (if known, for evaluation)
            true_rr: True RR value (if known, for evaluation)
        
        Returns:
            Dictionary with test results
        """
        # Apply stress to signal
        stressed_signal, metadata = self.stress_tester.apply_stress_scenario(
            original_signal, stress_scenario
        )
        
        # Load GPR models first to get required features
        from infer import load_optional_current_model, predict_current
        
        models = {}
        spo2_features = []
        rr_features = []
        all_required_features = set()
        
        if gpr_spo2_path:
            spo2_model = load_optional_current_model(gpr_spo2_path)
            models['spo2'] = spo2_model
            # Get required features from model
            if isinstance(spo2_model, dict) and 'features' in spo2_model:
                spo2_features = spo2_model['features']
                all_required_features.update(spo2_features)
        
        if gpr_rr_path:
            rr_model = load_optional_current_model(gpr_rr_path)
            models['rr'] = rr_model
            # Get required features from model
            if isinstance(rr_model, dict) and 'features' in rr_model:
                rr_features = rr_model['features']
                all_required_features.update(rr_features)
        
        # Extract features from stressed signal using required features
        try:
            required_features_list = list(all_required_features) if all_required_features else None
            features = self.extract_features_from_signal(stressed_signal, required_features_list)
        except Exception as e:
            return {
                'scenario_name': stress_scenario.get('name', 'Unknown'),
                'model_type': 'GPR',
                'success': False,
                'error': str(e),
                'metadata': metadata
            }
        
        # Count features actually used by each model
        spo2_features_count = len(spo2_features) if spo2_features else 0
        rr_features_count = len(rr_features) if rr_features else 0
        total_unique_features_used = len(all_required_features) if all_required_features else 0
        
        # Count all extracted features (for reference)
        all_extracted_features_count = len([v for v in features.values() if v != 0.0])
        
        result = {
            'scenario_name': stress_scenario.get('name', 'Unknown'),
            'model_type': 'GPR',
            'success': True,
            'metadata': metadata,
            'spo2_features_count': spo2_features_count,
            'rr_features_count': rr_features_count,
            'total_features_used': total_unique_features_used,  # Model actually uses (merged unique)
            'total_features_extracted': all_extracted_features_count,  # All features extracted from signal
            'predictions': {}
        }
        
        try:
            if models:
                # Use feature names from models or fallback to extracted features
                feature_names_for_pred = list(all_required_features) if all_required_features else list(features.keys())
                current_predictions = predict_current(models, feature_names_for_pred, features)
                result['predictions'] = current_predictions
                
                # Add evaluation metrics if true values provided
                if true_spo2 is not None and 'current_spo2' in current_predictions:
                    spo2_error = abs(current_predictions['current_spo2'] - true_spo2)
                    spo2_error_percent = (spo2_error / true_spo2) * 100 if true_spo2 > 0 else 0
                    result['evaluation'] = {
                        'spo2_error': float(spo2_error),
                        'spo2_error_percent': float(spo2_error_percent)
                    }
                
                if true_rr is not None and 'current_rr' in current_predictions:
                    rr_error = abs(current_predictions['current_rr'] - true_rr)
                    rr_error_percent = (rr_error / true_rr) * 100 if true_rr > 0 else 0
                    if 'evaluation' not in result:
                        result['evaluation'] = {}
                    result['evaluation']['rr_error'] = float(rr_error)
                    result['evaluation']['rr_error_percent'] = float(rr_error_percent)
            else:
                result['success'] = False
                result['error'] = 'No GPR models provided'
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def run_gpr_stress_test(self,
                            test_signal: np.ndarray,
                            gpr_spo2_path: str,
                            gpr_rr_path: str,
                            true_spo2: Optional[float] = None,
                            true_rr: Optional[float] = None) -> pd.DataFrame:
        """
        Run comprehensive stress test for GPR models
        
        Args:
            test_signal: Test PPG signal
            gpr_spo2_path: Path to GPR SpO2 model
            gpr_rr_path: Path to GPR RR model
            true_spo2: True SpO2 value (optional)
            true_rr: True RR value (optional)
        
        Returns:
            DataFrame with test results
        """
        # Load models once to get feature information
        from infer import load_optional_current_model
        
        spo2_model = load_optional_current_model(gpr_spo2_path)
        rr_model = load_optional_current_model(gpr_rr_path)
        
        spo2_features = []
        rr_features = []
        
        if isinstance(spo2_model, dict) and 'features' in spo2_model:
            spo2_features = spo2_model['features']
        
        if isinstance(rr_model, dict) and 'features' in rr_model:
            rr_features = rr_model['features']
        
        # Display model information
        print(f"\nGPR Model Information:")
        print(f"  SpO2 Model: {len(spo2_features)} features")
        if len(spo2_features) <= 15:
            print(f"    Features: {spo2_features}")
        else:
            print(f"    Features: {spo2_features[:10]}... (+{len(spo2_features)-10} more)")
        
        print(f"  RR Model: {len(rr_features)} features")
        if len(rr_features) <= 15:
            print(f"    Features: {rr_features}")
        else:
            print(f"    Features: {rr_features[:10]}... (+{len(rr_features)-10} more)")
        
        # Calculate unique features
        all_features = set(spo2_features) | set(rr_features)
        common_features = set(spo2_features) & set(rr_features)
        print(f"  Total unique features: {len(all_features)}")
        print(f"  Common features: {len(common_features)}")
        print()
        
        scenarios = self.stress_tester.generate_test_scenarios()
        results = []
        
        print(f"Running GPR model stress test with {len(scenarios)} scenarios...")
        print("=" * 60)
        
        for i, scenario in enumerate(scenarios):
            print(f"\n[{i+1}/{len(scenarios)}] Testing: {scenario['name']}")
            
            result = self.test_gpr_models_under_stress(
                test_signal, scenario, gpr_spo2_path, gpr_rr_path, true_spo2, true_rr
            )
            
            if result['success']:
                summary = {
                    'scenario': result['scenario_name'],
                    'model_type': 'GPR',
                    'success': True
                }
                
                # Only add evaluation metrics if available
                if 'evaluation' in result:
                    eval_data = result['evaluation']
                    if 'spo2_error' in eval_data:
                        summary['spo2_error'] = eval_data['spo2_error']
                    if 'rr_error' in eval_data:
                        summary['rr_error'] = eval_data['rr_error']
                
                results.append(summary)
                spo2_feat = result.get('spo2_features_count', 0)
                rr_feat = result.get('rr_features_count', 0)
                print(f"  ✓ Success - SpO2 features: {spo2_feat}, RR features: {rr_feat}")
            else:
                results.append({
                    'scenario': result['scenario_name'],
                    'model_type': 'GPR',
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
                print(f"  ✗ Failed - {result.get('error', 'Unknown error')}")
        
        return pd.DataFrame(results)
    
    def visualize_stress_impact(self,
                                original_signal: np.ndarray,
                                scenario: Dict[str, any],
                                save_path: Optional[str] = None):
        """
        Visualize impact of stress on signal
        
        Args:
            original_signal: Original signal
            scenario: Stress scenario
            save_path: Optional path to save figure
        """
        stressed_signal, metadata = self.stress_tester.apply_stress_scenario(
            original_signal, scenario
        )
        
        self.stress_tester.visualize_stress_test(
            original_signal, stressed_signal, metadata,
            duration=5.0, save_path=save_path
        )
    
    def visualize_gpr_predictions(self,
                                  results_df: pd.DataFrame,
                                  true_spo2: Optional[float] = None,
                                  true_rr: Optional[float] = None,
                                  save_path: Optional[str] = None):
        """
        Visualize GPR model predictions across all stress scenarios
        
        Args:
            results_df: DataFrame with test results (must contain 'scenario')
            true_spo2: True SpO2 value (optional, for reference line)
            true_rr: True RR value (optional, for reference line)
            save_path: Optional path to save figure
        
        Note: This function requires prediction values to be stored separately
        or passed via a different mechanism since they're not in the CSV.
        """
        import matplotlib.pyplot as plt
        
        # Filter successful results
        successful_results = results_df[results_df['success'] == True].copy()
        
        if len(successful_results) == 0:
            print("No successful results to visualize")
            return
        
        # Check if prediction columns exist (they may not be in CSV but needed for visualization)
        if 'current_spo2' not in successful_results.columns or 'current_rr' not in successful_results.columns:
            print("Warning: Prediction values not found in DataFrame. Skipping prediction visualization.")
            print("Note: Prediction values are not saved to CSV but are needed for visualization.")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        scenarios = successful_results['scenario'].values
        spo2_predictions = successful_results['current_spo2'].values
        rr_predictions = successful_results['current_rr'].values
        
        # Plot SpO2 predictions
        x_pos = np.arange(len(scenarios))
        colors_spo2 = ['green' if s == 'Baseline (No Stress)' else 'blue' for s in scenarios]
        
        bars1 = ax1.bar(x_pos, spo2_predictions, color=colors_spo2, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_xlabel('Stress Scenario', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Predicted SpO2 (%)', fontsize=11, fontweight='bold')
        ax1.set_title('SpO2 Predictions Across Stress Scenarios', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([max(0, min(spo2_predictions) - 5), min(100, max(spo2_predictions) + 5)])
        
        # Add true value line if available
        if true_spo2 is not None:
            ax1.axhline(y=true_spo2, color='red', linestyle='--', linewidth=2, 
                       label=f'True SpO2: {true_spo2:.1f}%', alpha=0.8)
            ax1.legend(loc='best')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, spo2_predictions)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add error annotations if available
        if 'spo2_error' in successful_results.columns:
            for i, (idx, row) in enumerate(successful_results.iterrows()):
                if pd.notna(row.get('spo2_error')):
                    error = row['spo2_error']
                    ax1.text(i, spo2_predictions[i] + 1,
                            f'±{error:.1f}',
                            ha='center', va='bottom', fontsize=7, color='red', style='italic')
        
        # Plot RR predictions
        colors_rr = ['green' if s == 'Baseline (No Stress)' else 'orange' for s in scenarios]
        
        bars2 = ax2.bar(x_pos, rr_predictions, color=colors_rr, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.set_xlabel('Stress Scenario', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Predicted RR (breaths/min)', fontsize=11, fontweight='bold')
        ax2.set_title('RR Predictions Across Stress Scenarios', fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([max(0, min(rr_predictions) - 2), max(rr_predictions) + 2])
        
        # Add true value line if available
        if true_rr is not None:
            ax2.axhline(y=true_rr, color='red', linestyle='--', linewidth=2,
                       label=f'True RR: {true_rr:.1f} breaths/min', alpha=0.8)
            ax2.legend(loc='best')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars2, rr_predictions)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add error annotations if available
        if 'rr_error' in successful_results.columns:
            for i, (idx, row) in enumerate(successful_results.iterrows()):
                if pd.notna(row.get('rr_error')):
                    error = row['rr_error']
                    ax2.text(i, rr_predictions[i] + 0.5,
                            f'±{error:.1f}',
                            ha='center', va='bottom', fontsize=7, color='red', style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def load_test_signal_from_bidmc(csv_path: str, 
                                duration: float = 30.0) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """
    Load test signal from BIDMC dataset CSV
    
    Args:
        csv_path: Path to BIDMC CSV file
        duration: Duration to load in seconds
    
    Returns:
        Tuple of (signal, SpO2, RR) - SpO2 and RR may be None if not available
    """
    df = pd.read_csv(csv_path)
    
    # Extract signal
    time_col = 'Time [s]'
    value_col = ' PLETH'
    
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Required columns not found in CSV")
    
    # Filter by duration
    start_time = df[time_col].iloc[0]
    end_time = start_time + duration
    df_segment = df[(df[time_col] >= start_time) & (df[time_col] < end_time)]
    
    signal = df_segment[value_col].values
    
    # Extract SpO2 and RR if available
    spo2 = None
    rr = None
    
    if ' SpO2' in df.columns:
        spo2 = df_segment[' SpO2'].mean()
    
    if ' RESP' in df.columns:
        rr = df_segment[' RESP'].mean()
    
    return signal, spo2, rr


def main():
    """
    Main function for model stress testing
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stress test prediction models with various data quality scenarios"
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Path to saved Transformer+XGBoost future prediction model (required for future model testing)')
    parser.add_argument('--gpr_spo2', type=str, default=None,
                       help='Path to GPR SpO2 model for current prediction (required for GPR testing)')
    parser.add_argument('--gpr_rr', type=str, default=None,
                       help='Path to GPR RR model for current prediction (required for GPR testing)')
    parser.add_argument('--test_future', action='store_true',
                       help='Test future prediction model (Transformer+XGBoost)')
    parser.add_argument('--test_gpr', action='store_true',
                       help='Test GPR models (current prediction)')
    parser.add_argument('--test_both', action='store_true',
                       help='Test both future prediction and GPR models')
    parser.add_argument('--signal_csv', type=str, default=None,
                       help='Path to BIDMC CSV file with PPG signal')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic signal instead of CSV')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Signal duration in seconds')
    parser.add_argument('--output', type=str, default='model_stress_test_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Determine test mode
    test_future = args.test_future or args.test_both
    test_gpr = args.test_gpr or args.test_both
    
    # If no test mode specified, default to future model testing
    if not test_future and not test_gpr:
        test_future = True
        if args.model is None:
            args.model = 'Software/Checkpoints/transformer_xgboost_model_extended_10-300s.pkl'
    
    # Validate model files based on test mode
    if test_future:
        if args.model is None:
            print("=" * 60)
            print("ERROR: --model is required when testing future prediction model")
            print("=" * 60)
            print("\nPlease provide --model path or use --test_gpr to test GPR models only")
            print("=" * 60)
            return 1
        if not os.path.exists(args.model):
            print("=" * 60)
            print(f"ERROR: Future prediction model file not found: {args.model}")
            print("=" * 60)
            print("\nPlease check the model path and try again.")
            print("Common locations:")
            print("  - Software/Checkpoints/transformer_xgboost_model_extended_10-300s.pkl")
            print("  - ../../transformer_xgboost_model_extended_10-300s.pkl")
            print("=" * 60)
            return 1
    
    if test_gpr:
        if not args.gpr_spo2 or not args.gpr_rr:
            print("=" * 60)
            print("ERROR: Missing required GPR model parameters")
            print("=" * 60)
            print("\nWhen testing GPR models, you must provide:")
            print("  --gpr_spo2: Path to GPR SpO2 model")
            print("  --gpr_rr: Path to GPR RR model")
            print("\nExample:")
            print("  python Tuning/stress_test_model_integration.py \\")
            print("    --test_gpr \\")
            print("    --gpr_spo2 Software/Checkpoints/current_spo2_model.pkl \\")
            print("    --gpr_rr Software/Checkpoints/current_rr_model.pkl \\")
            print("    --synthetic --visualize")
            print("=" * 60)
            return 1
        if not os.path.exists(args.gpr_spo2):
            print(f"ERROR: GPR SpO2 model not found: {args.gpr_spo2}")
            return 1
        if not os.path.exists(args.gpr_rr):
            print(f"ERROR: GPR RR model not found: {args.gpr_rr}")
            return 1
    
    # Load or generate test signal
    if args.synthetic:
        print("Generating synthetic PPG signal...")
        test_signal = generate_synthetic_ppg_signal(
            duration=args.duration, 
            sampling_rate=125, 
            heart_rate=72.0
        )
        true_spo2 = None
        true_rr = None
    elif args.signal_csv:
        print(f"Loading signal from: {args.signal_csv}")
        test_signal, true_spo2, true_rr = load_test_signal_from_bidmc(
            args.signal_csv, duration=args.duration
        )
        print(f"Loaded signal: {len(test_signal)} samples")
        if true_spo2:
            print(f"True SpO2: {true_spo2:.2f}%")
        if true_rr:
            print(f"True RR: {true_rr:.2f} breaths/min")
    else:
        print("Using synthetic signal (use --synthetic or --signal_csv)")
        test_signal = generate_synthetic_ppg_signal(
            duration=args.duration, 
            sampling_rate=125, 
            heart_rate=72.0
        )
        true_spo2 = None
        true_rr = None
    
    print()
    
    all_results = {}
    
    # Test Future prediction model
    if test_future:
        print("=== Future Prediction Model Stress Testing ===")
        print(f"Model: {args.model}")
        print(f"Duration: {args.duration}s")
        print()
        
        # Initialize tester
        tester_future = ModelStressTester(args.model)
        
        # Run stress tests
        results_df_future = tester_future.run_comprehensive_stress_test(
            test_signal, true_spo2, true_rr
        )
        
        all_results['future'] = {
            'tester': tester_future,
            'results': results_df_future,
            'output': args.output.replace('.csv', '_future.csv') if args.output else 'model_stress_test_results_future.csv'
        }
    
    # Test GPR models
    if test_gpr:
        print("\n" + "=" * 60)
        print("=== GPR Model Stress Testing ===")
        print(f"GPR SpO2 Model: {args.gpr_spo2}")
        print(f"GPR RR Model: {args.gpr_rr}")
        print(f"Duration: {args.duration}s")
        print("=" * 60)
        print()
        
        # Initialize tester (no future model needed - GPR models contain their own feature lists)
        tester_gpr = ModelStressTester(future_model_path=None)
        
        # Run GPR stress tests
        results_df_gpr = tester_gpr.run_gpr_stress_test(
            test_signal, args.gpr_spo2, args.gpr_rr, true_spo2, true_rr
        )
        
        all_results['gpr'] = {
            'tester': tester_gpr,
            'results': results_df_gpr,
            'output': args.output.replace('.csv', '_gpr.csv') if args.output else 'model_stress_test_results_gpr.csv'
        }
    
    # Save results
    for model_type, result_data in all_results.items():
        output_file = result_data['output']

        output_dir = "stress_test/model_stress_results/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}{output_file}"
                    
        result_data['results'].to_csv(output_path, index=False)
        print(f"\n{model_type.upper()} model results saved to: {output_path}")
        
        # Print summary
        print(f"\n=== {model_type.upper()} Model Stress Test Summary ===")
        print(result_data['results'].to_string(index=False))
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # Signal-level visualizations (for all testers)
        tester_for_viz = all_results.get('future', {}).get('tester') or all_results.get('gpr', {}).get('tester')
        if tester_for_viz:
            scenarios = tester_for_viz.stress_tester.generate_test_scenarios()
            visualize_indices = [0, 1, 4, 5, 9]  # Key scenarios
            
            for idx in visualize_indices:
                if idx < len(scenarios):
                    scenario = scenarios[idx]
                    save_dir = "stress_test/signals/"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = f"{save_dir}stress_visualization_{scenario['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
                    tester_for_viz.visualize_stress_impact(test_signal, scenario, save_path=save_path)
        
        # GPR prediction visualizations (if GPR models were tested)
        if test_gpr and 'gpr' in all_results:
            gpr_results = all_results['gpr']['results']
            gpr_tester = all_results['gpr']['tester']
            
            # Visualize SpO2 and RR predictions
            pred_viz_path = save_dir + "gpr_predictions_stress_comparison.png"
            gpr_tester.visualize_gpr_predictions(
                gpr_results, true_spo2, true_rr, save_path=pred_viz_path
            )
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

