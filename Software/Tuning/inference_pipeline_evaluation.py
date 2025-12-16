'''
Inference Pipeline Evaluation: Timing Stability, Latency, and Robustness
Evaluates the inference pipeline under variable conditions

The evaluation generates several output files:
1. inference_timing_stability.csv: Detailed timing statistics for each input
2. inference_timing_distribution.png: Visualization of timing distribution
3. inference_robustness_results.csv: Results from robustness tests
4. inference_robustness_analysis.png: Visualization of robustness analysis
5. inference_evaluation_summary.txt: Text summary of all results

Usage Example:
cd Software
python Tuning/inference_pipeline_evaluation.py \
    --future_model Checkpoints/transformer_xgboost_model_extended_10-300s.pkl \
    --current_spo2 Checkpoints/current_spo2_model.pkl \
    --current_rr Checkpoints/current_rr_model.pkl \
    --test_input_json Test_Infer/test_input_features.json \
    --include_current \
    --output_dir ./infer_eval_results
'''

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import warnings
from collections import defaultdict
import concurrent.futures
import threading
import random
warnings.filterwarnings('ignore')

# Add Software directory to path
software_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, software_dir)

from infer import FutureModelWrapper, load_optional_current_model, predict_current
from Tuning.stress_testing_framework import PPGStressTester, generate_synthetic_ppg_signal


class InferencePipelineEvaluator:
    """
    Comprehensive evaluation of inference pipeline:
    - Timing stability (consistency of inference time)
    - Latency (mean, min, max, percentiles)
    - Robustness (handling variable conditions)
    """
    
    def __init__(self, 
                 future_model_path: str,
                 current_spo2_model_path: Optional[str] = None,
                 current_rr_model_path: Optional[str] = None):
        """
        Initialize evaluator
        
        Args:
            future_model_path: Path to Transformer+XGBoost future prediction model
            current_spo2_model_path: Optional path to current SpO2 GPR model
            current_rr_model_path: Optional path to current RR GPR model
        """
        self.future_model_path = future_model_path
        self.current_spo2_model_path = current_spo2_model_path
        self.current_rr_model_path = current_rr_model_path
        
        # Load models
        print("Loading models...")
        self.future_model = FutureModelWrapper(future_model_path)
        self.future_model.load()
        print(f"  Future model loaded: {len(self.future_model.feature_names)} features")
        
        self.current_models = {}
        if current_spo2_model_path:
            self.current_models['spo2'] = load_optional_current_model(current_spo2_model_path)
            print(f"  Current SpO2 model loaded")
        if current_rr_model_path:
            self.current_models['rr'] = load_optional_current_model(current_rr_model_path)
            print(f"  Current RR model loaded")
        
        # Initialize stress tester for robustness tests
        self.stress_tester = PPGStressTester(sampling_rate=125)
        
        # Results storage
        self.timing_results = []
        self.robustness_results = []
        
    def _time_inference(self, 
                       input_data: Dict[str, Any],
                       include_current: bool = False) -> Tuple[Dict[str, Any], float]:
        """
        Time a single inference call
        
        Args:
            input_data: Input data for inference
            include_current: Whether to include current predictions
            
        Returns:
            Tuple of (prediction results, elapsed time in seconds)
        """
        start_time = time.perf_counter()
        
        # Future prediction
        future_result = self.future_model.predict(input_data)
        
        # Current prediction (if requested and models available)
        if include_current and self.current_models:
            # Extract current moment vector
            if "vector" in input_data:
                current_vector = input_data["vector"]
            elif "csv" in input_data:
                import pandas as pd
                df = pd.read_csv(input_data["csv"]["path"])
                last_row = df.iloc[-1]
                current_vector = {
                    col: float(last_row[col]) if pd.notna(last_row[col]) else 0.0
                    for col in df.columns if col not in ["wave nunmber", "segment nunmber"]
                }
            else:
                current_vector = None
            
            if current_vector:
                current_result = predict_current(
                    self.current_models, 
                    self.future_model.feature_names,
                    current_vector
                )
                future_result["current"] = current_result
        
        elapsed_time = time.perf_counter() - start_time
        
        return future_result, elapsed_time
    
    def evaluate_timing_stability(self,
                                  test_inputs: List[Dict[str, Any]],
                                  n_runs_per_input: int = 10,
                                  include_current: bool = False,
                                  warmup_runs: int = 3) -> pd.DataFrame:
        """
        Evaluate timing stability across multiple runs
        
        Args:
            test_inputs: List of input data dictionaries to test
            n_runs_per_input: Number of runs per input for stability measurement
            include_current: Whether to include current predictions
            warmup_runs: Number of warmup runs to discard (for JIT compilation, etc.)
            
        Returns:
            DataFrame with timing statistics
        """
        print(f"\n=== Timing Stability Evaluation ===")
        print(f"Testing {len(test_inputs)} inputs with {n_runs_per_input} runs each")
        print(f"Warmup runs: {warmup_runs}")
        
        all_timings = []
        
        for input_idx, input_data in enumerate(test_inputs):
            print(f"\nInput {input_idx + 1}/{len(test_inputs)}")
            
            # Warmup runs
            for _ in range(warmup_runs):
                try:
                    self._time_inference(input_data, include_current=False)
                except:
                    pass
            
            # Actual timing runs
            timings = []
            for run in range(n_runs_per_input):
                try:
                    _, elapsed = self._time_inference(input_data, include_current=include_current)
                    timings.append(elapsed)
                    if (run + 1) % 10 == 0:
                        print(f"  Run {run + 1}/{n_runs_per_input}: {elapsed*1000:.2f}ms")
                except Exception as e:
                    print(f"  Run {run + 1} failed: {e}")
            
            if timings:
                timing_stats = {
                    'input_id': input_idx,
                    'mean_ms': np.mean(timings) * 1000,
                    'std_ms': np.std(timings) * 1000,
                    'min_ms': np.min(timings) * 1000,
                    'max_ms': np.max(timings) * 1000,
                    'median_ms': np.median(timings) * 1000,
                    'p25_ms': np.percentile(timings, 25) * 1000,
                    'p75_ms': np.percentile(timings, 75) * 1000,
                    'p95_ms': np.percentile(timings, 95) * 1000,
                    'p99_ms': np.percentile(timings, 99) * 1000,
                    'cv': np.std(timings) / np.mean(timings) if np.mean(timings) > 0 else 0,  # Coefficient of variation
                    'n_runs': len(timings)
                }
                all_timings.append(timing_stats)
                print(f"  Mean: {timing_stats['mean_ms']:.2f}ms, Std: {timing_stats['std_ms']:.2f}ms, CV: {timing_stats['cv']:.4f}")
        
        df = pd.DataFrame(all_timings)
        self.timing_results.append(df)
        return df
    
    def evaluate_latency(self,
                        test_inputs: List[Dict[str, Any]],
                        n_samples: int = 100,
                        include_current: bool = False) -> pd.DataFrame:
        """
        Evaluate latency characteristics
        
        Args:
            test_inputs: List of input data dictionaries
            n_samples: Total number of inference samples to collect
            include_current: Whether to include current predictions
            
        Returns:
            DataFrame with latency statistics
        """
        print(f"\n=== Latency Evaluation ===")
        print(f"Collecting {n_samples} inference samples")
        
        timings = []
        input_ids = []
        
        # Distribute samples across inputs
        samples_per_input = max(1, n_samples // len(test_inputs))
        
        for input_idx, input_data in enumerate(test_inputs):
            n_for_this_input = samples_per_input if input_idx < len(test_inputs) - 1 else n_samples - len(timings)
            
            for _ in range(n_for_this_input):
                try:
                    _, elapsed = self._time_inference(input_data, include_current=include_current)
                    timings.append(elapsed)
                    input_ids.append(input_idx)
                except Exception as e:
                    print(f"  Warning: Inference failed: {e}")
        
        if not timings:
            print("  No successful inferences!")
            return pd.DataFrame()
        
        latency_stats = {
            'metric': [
                'mean', 'std', 'min', 'max', 'median',
                'p25', 'p75', 'p90', 'p95', 'p99',
                'cv', 'n_samples'
            ],
            'latency_ms': [
                np.mean(timings) * 1000,
                np.std(timings) * 1000,
                np.min(timings) * 1000,
                np.max(timings) * 1000,
                np.median(timings) * 1000,
                np.percentile(timings, 25) * 1000,
                np.percentile(timings, 75) * 1000,
                np.percentile(timings, 90) * 1000,
                np.percentile(timings, 95) * 1000,
                np.percentile(timings, 99) * 1000,
                (np.std(timings) / np.mean(timings)) * 1000 if np.mean(timings) > 0 else 0,
                len(timings)
            ]
        }
        
        df = pd.DataFrame(latency_stats)
        print(f"\nLatency Statistics:")
        print(df.to_string(index=False))
        
        return df
    
    def evaluate_robustness(self,
                           base_input: Dict[str, Any],
                           stress_scenarios: Optional[List[Dict[str, Any]]] = None,
                           edge_cases: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Evaluate robustness under variable conditions
        
        Args:
            base_input: Base input data (vector or CSV path)
            stress_scenarios: Optional list of stress scenarios to apply
            edge_cases: Optional list of edge case inputs to test
            
        Returns:
            DataFrame with robustness test results
        """
        print(f"\n=== Robustness Evaluation ===")
        
        results = []
        
        # Test 1: Baseline (no stress)
        print("\n1. Baseline test (no stress)")
        try:
            result, elapsed = self._time_inference(base_input, include_current=True)
            results.append({
                'test_type': 'baseline',
                'test_name': 'Baseline (No Stress)',
                'success': True,
                'latency_ms': elapsed * 1000,
                'error': None,
                'predictions_count': len(result.get('predictions', {}))
            })
            print(f"  ✓ Success - Latency: {elapsed*1000:.2f}ms")
        except Exception as e:
            results.append({
                'test_type': 'baseline',
                'test_name': 'Baseline (No Stress)',
                'success': False,
                'latency_ms': None,
                'error': str(e),
                'predictions_count': 0
            })
            print(f"  ✗ Failed: {e}")
        
        # Test 2: Stress scenarios (if provided)
        if stress_scenarios:
            print(f"\n2. Stress scenarios ({len(stress_scenarios)} tests)")
            for scenario in stress_scenarios:
                test_name = scenario.get('name', 'Unknown Stress')
                print(f"  Testing: {test_name}")
                
                # Apply stress to input if it's a signal
                # For vector inputs, we'll add noise to feature values
                try:
                    if "vector" in base_input:
                        # Add noise to feature vector
                        stressed_vector = base_input["vector"].copy()
                        noise_level = scenario.get('noise_level', 0.1)
                        for key in stressed_vector:
                            if isinstance(stressed_vector[key], (int, float)):
                                noise = np.random.normal(0, abs(stressed_vector[key]) * noise_level)
                                stressed_vector[key] = stressed_vector[key] + noise
                        
                        stressed_input = {"vector": stressed_vector}
                    else:
                        # For CSV, we'd need to apply stress to signal first
                        # For now, just use base input
                        stressed_input = base_input
                    
                    result, elapsed = self._time_inference(stressed_input, include_current=True)
                    results.append({
                        'test_type': 'stress',
                        'test_name': test_name,
                        'success': True,
                        'latency_ms': elapsed * 1000,
                        'error': None,
                        'predictions_count': len(result.get('predictions', {}))
                    })
                    print(f"    ✓ Success - Latency: {elapsed*1000:.2f}ms")
                except Exception as e:
                    results.append({
                        'test_type': 'stress',
                        'test_name': test_name,
                        'success': False,
                        'latency_ms': None,
                        'error': str(e),
                        'predictions_count': 0
                    })
                    print(f"    ✗ Failed: {e}")
        
        # Test 3: Edge cases
        if edge_cases:
            print(f"\n3. Edge cases ({len(edge_cases)} tests)")
            for i, edge_case in enumerate(edge_cases):
                test_name = edge_case.get('name', f'Edge Case {i+1}')
                input_data = edge_case.get('input', base_input)
                print(f"  Testing: {test_name}")
                
                try:
                    result, elapsed = self._time_inference(input_data, include_current=True)
                    results.append({
                        'test_type': 'edge_case',
                        'test_name': test_name,
                        'success': True,
                        'latency_ms': elapsed * 1000,
                        'error': None,
                        'predictions_count': len(result.get('predictions', {}))
                    })
                    print(f"    ✓ Success - Latency: {elapsed*1000:.2f}ms")
                except Exception as e:
                    results.append({
                        'test_type': 'edge_case',
                        'test_name': test_name,
                        'success': False,
                        'latency_ms': None,
                        'error': str(e),
                        'predictions_count': 0
                    })
                    print(f"    ✗ Failed: {e}")
        
        # Test 4: Missing features
        print(f"\n4. Missing features test")
        if "vector" in base_input:
            # Remove some features
            partial_vector = base_input["vector"].copy()
            feature_names = list(partial_vector.keys())
            n_to_remove = min(5, len(feature_names) // 4)
            removed_features = random.sample(feature_names, n_to_remove)
            for feat in removed_features:
                del partial_vector[feat]
            
            try:
                partial_input = {"vector": partial_vector}
                result, elapsed = self._time_inference(partial_input, include_current=True)
                results.append({
                    'test_type': 'missing_features',
                    'test_name': f'Missing {n_to_remove} features',
                    'success': True,
                    'latency_ms': elapsed * 1000,
                    'error': None,
                    'predictions_count': len(result.get('predictions', {})),
                    'removed_features': ', '.join(removed_features)
                })
                print(f"  ✓ Success - Latency: {elapsed*1000:.2f}ms (removed: {', '.join(removed_features[:3])}...)")
            except Exception as e:
                results.append({
                    'test_type': 'missing_features',
                    'test_name': f'Missing {n_to_remove} features',
                    'success': False,
                    'latency_ms': None,
                    'error': str(e),
                    'predictions_count': 0
                })
                print(f"  ✗ Failed: {e}")
        
        # Test 5: Invalid values
        print(f"\n5. Invalid values test")
        if "vector" in base_input:
            invalid_vector = base_input["vector"].copy()
            # Set some features to NaN or extreme values
            feature_names = list(invalid_vector.keys())
            n_to_corrupt = min(3, len(feature_names) // 5)
            corrupted_features = random.sample(feature_names, n_to_corrupt)
            for feat in corrupted_features:
                if random.random() > 0.5:
                    invalid_vector[feat] = float('inf')
                else:
                    invalid_vector[feat] = float('nan')
            
            try:
                invalid_input = {"vector": invalid_vector}
                result, elapsed = self._time_inference(invalid_input, include_current=True)
                results.append({
                    'test_type': 'invalid_values',
                    'test_name': f'Invalid values in {n_to_corrupt} features',
                    'success': True,
                    'latency_ms': elapsed * 1000,
                    'error': None,
                    'predictions_count': len(result.get('predictions', {}))
                })
                print(f"  ✓ Success - Latency: {elapsed*1000:.2f}ms")
            except Exception as e:
                results.append({
                    'test_type': 'invalid_values',
                    'test_name': f'Invalid values in {n_to_corrupt} features',
                    'success': False,
                    'latency_ms': None,
                    'error': str(e),
                    'predictions_count': 0
                })
                print(f"  ✗ Failed: {e}")
        
        # Test 6: Concurrent requests (simulated)
        print(f"\n6. Concurrent requests test")
        n_concurrent = 5
        print(f"  Simulating {n_concurrent} concurrent requests")
        
        def run_inference(input_data):
            try:
                result, elapsed = self._time_inference(input_data, include_current=False)
                return {'success': True, 'latency_ms': elapsed * 1000, 'error': None}
            except Exception as e:
                return {'success': False, 'latency_ms': None, 'error': str(e)}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            futures = [executor.submit(run_inference, base_input) for _ in range(n_concurrent)]
            concurrent_results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        success_count = sum(1 for r in concurrent_results if r['success'])
        avg_latency = np.mean([r['latency_ms'] for r in concurrent_results if r['success']]) if success_count > 0 else None
        
        results.append({
            'test_type': 'concurrent',
            'test_name': f'{n_concurrent} concurrent requests',
            'success': success_count == n_concurrent,
            'latency_ms': avg_latency,
            'error': None if success_count == n_concurrent else f'{n_concurrent - success_count} failed',
            'predictions_count': success_count,
            'success_rate': success_count / n_concurrent
        })
        print(f"  {'✓' if success_count == n_concurrent else '⚠'} Success rate: {success_count}/{n_concurrent}, Avg latency: {avg_latency:.2f}ms" if avg_latency else f"  ✗ All failed")
        
        df = pd.DataFrame(results)
        self.robustness_results.append(df)
        return df
    
    def generate_timing_report(self, output_dir: str = ".") -> Dict[str, str]:
        """
        Generate comprehensive timing and latency report
        
        Args:
            output_dir: Directory to save reports
            
        Returns:
            Dictionary with paths to generated files
        """
        print(f"\n=== Generating Timing Report ===")
        
        report_files = {}
        
        # Combine all timing results
        if self.timing_results:
            all_timing = pd.concat(self.timing_results, ignore_index=True)
            
            # Save timing statistics
            timing_file = os.path.join(output_dir, "inference_timing_stability.csv")
            all_timing.to_csv(timing_file, index=False)
            report_files['timing_stability'] = timing_file
            print(f"  Timing stability saved to: {timing_file}")
            
            # Generate timing visualization
            if len(all_timing) > 0:
                self._plot_timing_distribution(all_timing, output_dir)
                report_files['timing_plot'] = os.path.join(output_dir, "inference_timing_distribution.png")
        
        # Robustness results
        if self.robustness_results:
            all_robustness = pd.concat(self.robustness_results, ignore_index=True)
            
            robustness_file = os.path.join(output_dir, "inference_robustness_results.csv")
            all_robustness.to_csv(robustness_file, index=False)
            report_files['robustness'] = robustness_file
            print(f"  Robustness results saved to: {robustness_file}")
            
            # Generate robustness visualization
            self._plot_robustness_results(all_robustness, output_dir)
            report_files['robustness_plot'] = os.path.join(output_dir, "inference_robustness_analysis.png")
        
        # Summary report
        summary_file = os.path.join(output_dir, "inference_evaluation_summary.txt")
        self._generate_summary_report(summary_file)
        report_files['summary'] = summary_file
        print(f"  Summary report saved to: {summary_file}")
        
        return report_files
    
    def _plot_timing_distribution(self, timing_df: pd.DataFrame, output_dir: str):
        """Plot timing distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Mean latency per input
        axes[0, 0].bar(range(len(timing_df)), timing_df['mean_ms'], alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Input ID')
        axes[0, 0].set_ylabel('Mean Latency (ms)')
        axes[0, 0].set_title('Mean Latency per Input')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Coefficient of variation (stability)
        axes[0, 1].bar(range(len(timing_df)), timing_df['cv'], alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Input ID')
        axes[0, 1].set_ylabel('Coefficient of Variation')
        axes[0, 1].set_title('Timing Stability (CV)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Latency range (min-max)
        x_pos = range(len(timing_df))
        axes[1, 0].bar(x_pos, timing_df['min_ms'], alpha=0.5, color='lightblue', label='Min')
        axes[1, 0].bar(x_pos, timing_df['max_ms'] - timing_df['min_ms'], 
                      bottom=timing_df['min_ms'], alpha=0.5, color='darkblue', label='Max')
        axes[1, 0].set_xlabel('Input ID')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].set_title('Latency Range (Min-Max)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Percentile distribution
        percentiles = ['p25_ms', 'median_ms', 'p75_ms', 'p95_ms']
        percentile_data = timing_df[percentiles].mean()
        axes[1, 1].bar(range(len(percentiles)), percentile_data.values, alpha=0.7, color='orange')
        axes[1, 1].set_xticks(range(len(percentiles)))
        axes[1, 1].set_xticklabels(['P25', 'Median', 'P75', 'P95'])
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].set_title('Average Percentile Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "inference_timing_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Timing distribution plot saved to: {plot_path}")
    
    def _plot_robustness_results(self, robustness_df: pd.DataFrame, output_dir: str):
        """Plot robustness analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Success rate by test type
        success_by_type = robustness_df.groupby('test_type')['success'].agg(['sum', 'count'])
        success_by_type['rate'] = success_by_type['sum'] / success_by_type['count']
        
        axes[0, 0].bar(success_by_type.index, success_by_type['rate'] * 100, alpha=0.7, color='green')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].set_title('Success Rate by Test Type')
        axes[0, 0].set_ylim([0, 105])
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Latency by test type (only successful tests)
        successful = robustness_df[robustness_df['success'] == True]
        if len(successful) > 0:
            latency_by_type = successful.groupby('test_type')['latency_ms'].mean()
            axes[0, 1].bar(latency_by_type.index, latency_by_type.values, alpha=0.7, color='blue')
            axes[0, 1].set_ylabel('Mean Latency (ms)')
            axes[0, 1].set_title('Mean Latency by Test Type')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Test results overview
        test_names = robustness_df['test_name'].values
        success_status = robustness_df['success'].values
        colors = ['green' if s else 'red' for s in success_status]
        
        axes[1, 0].barh(range(len(test_names)), [1] * len(test_names), color=colors, alpha=0.7)
        axes[1, 0].set_yticks(range(len(test_names)))
        axes[1, 0].set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in test_names], fontsize=8)
        axes[1, 0].set_xlabel('Status')
        axes[1, 0].set_title('Test Results Overview')
        axes[1, 0].set_xlim([0, 1.2])
        
        # 4. Latency comparison (successful tests only)
        if len(successful) > 0:
            test_names_short = [name[:20] + '...' if len(name) > 20 else name for name in successful['test_name'].values]
            axes[1, 1].barh(range(len(test_names_short)), successful['latency_ms'].values, alpha=0.7, color='blue')
            axes[1, 1].set_yticks(range(len(test_names_short)))
            axes[1, 1].set_yticklabels(test_names_short, fontsize=7)
            axes[1, 1].set_xlabel('Latency (ms)')
            axes[1, 1].set_title('Latency by Test (Successful Only)')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "inference_robustness_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Robustness analysis plot saved to: {plot_path}")
    
    def _generate_summary_report(self, output_file: str):
        """Generate text summary report"""
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("INFERENCE PIPELINE EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            # Timing stability summary
            if self.timing_results:
                all_timing = pd.concat(self.timing_results, ignore_index=True)
                f.write("TIMING STABILITY:\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Number of inputs tested: {len(all_timing)}\n")
                f.write(f"  Overall mean latency: {all_timing['mean_ms'].mean():.2f} ms\n")
                f.write(f"  Overall std latency: {all_timing['std_ms'].mean():.2f} ms\n")
                f.write(f"  Overall CV (stability): {all_timing['cv'].mean():.4f}\n")
                f.write(f"  Min latency: {all_timing['min_ms'].min():.2f} ms\n")
                f.write(f"  Max latency: {all_timing['max_ms'].max():.2f} ms\n")
                f.write(f"  P95 latency: {all_timing['p95_ms'].mean():.2f} ms\n")
                f.write(f"  P99 latency: {all_timing['p99_ms'].mean():.2f} ms\n\n")
            
            # Robustness summary
            if self.robustness_results:
                all_robustness = pd.concat(self.robustness_results, ignore_index=True)
                f.write("ROBUSTNESS:\n")
                f.write("-" * 70 + "\n")
                total_tests = len(all_robustness)
                successful_tests = len(all_robustness[all_robustness['success'] == True])
                success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
                f.write(f"  Total tests: {total_tests}\n")
                f.write(f"  Successful: {successful_tests}\n")
                f.write(f"  Failed: {total_tests - successful_tests}\n")
                f.write(f"  Success rate: {success_rate:.1f}%\n\n")
                
                # By test type
                f.write("  Results by test type:\n")
                for test_type in all_robustness['test_type'].unique():
                    type_df = all_robustness[all_robustness['test_type'] == test_type]
                    type_success = len(type_df[type_df['success'] == True])
                    f.write(f"    {test_type}: {type_success}/{len(type_df)} successful\n")
                
                # Latency for successful tests
                successful = all_robustness[all_robustness['success'] == True]
                if len(successful) > 0:
                    f.write(f"\n  Mean latency (successful tests): {successful['latency_ms'].mean():.2f} ms\n")
                    f.write(f"  Std latency (successful tests): {successful['latency_ms'].std():.2f} ms\n")
            
            f.write("\n" + "=" * 70 + "\n")


def generate_test_inputs_from_json(json_path: str, n_variations: int = 5) -> List[Dict[str, Any]]:
    """
    Generate test inputs from a JSON file with variations
    
    Args:
        json_path: Path to JSON file with feature vector
        n_variations: Number of variations to generate
        
    Returns:
        List of input dictionaries
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    base_vector = data.get("vector", {})
    inputs = []
    
    # Original input
    inputs.append({"vector": base_vector.copy()})
    
    # Generate variations with small random noise
    for i in range(n_variations - 1):
        variation = base_vector.copy()
        for key in variation:
            if isinstance(variation[key], (int, float)):
                noise = np.random.normal(0, abs(variation[key]) * 0.05)  # 5% noise
                variation[key] = variation[key] + noise
        inputs.append({"vector": variation})
    
    return inputs


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate inference pipeline: timing stability, latency, and robustness"
    )
    parser.add_argument('--future_model', type=str, required=True,
                       help='Path to Transformer+XGBoost future prediction model')
    parser.add_argument('--current_spo2', type=str, default=None,
                       help='Optional path to current SpO2 GPR model')
    parser.add_argument('--current_rr', type=str, default=None,
                       help='Optional path to current RR GPR model')
    parser.add_argument('--test_input_json', type=str, default=None,
                       help='Path to JSON file with test input features')
    parser.add_argument('--test_input_csv', type=str, default=None,
                       help='Path to CSV file with test input features')
    parser.add_argument('--n_timing_runs', type=int, default=10,
                       help='Number of runs per input for timing stability (default: 10)')
    parser.add_argument('--n_latency_samples', type=int, default=100,
                       help='Number of samples for latency evaluation (default: 100)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for reports (default: current directory)')
    parser.add_argument('--include_current', action='store_true',
                       help='Include current predictions in timing tests')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = InferencePipelineEvaluator(
        args.future_model,
        args.current_spo2,
        args.current_rr
    )
    
    # Prepare test inputs
    test_inputs = []
    if args.test_input_json:
        test_inputs = generate_test_inputs_from_json(args.test_input_json, n_variations=5)
        print(f"Generated {len(test_inputs)} test inputs from JSON")
    elif args.test_input_csv:
        test_inputs = [{"csv": {"path": args.test_input_csv}}]
        print(f"Using CSV input: {args.test_input_csv}")
    else:
        # Use default test input
        default_json = os.path.join(software_dir, "Test_Infer", "test_input_features.json")
        if os.path.exists(default_json):
            test_inputs = generate_test_inputs_from_json(default_json, n_variations=5)
            print(f"Using default test input from: {default_json}")
        else:
            print("ERROR: No test input provided and default not found")
            print("Please provide --test_input_json or --test_input_csv")
            return 1
    
    # Run evaluations
    # 1. Timing stability
    timing_df = evaluator.evaluate_timing_stability(
        test_inputs,
        n_runs_per_input=args.n_timing_runs,
        include_current=args.include_current
    )
    
    # 2. Latency
    latency_df = evaluator.evaluate_latency(
        test_inputs,
        n_samples=args.n_latency_samples,
        include_current=args.include_current
    )
    
    # 3. Robustness
    base_input = test_inputs[0]
    stress_scenarios = [
        {'name': 'Light Noise', 'noise_level': 0.05},
        {'name': 'Moderate Noise', 'noise_level': 0.1},
        {'name': 'Heavy Noise', 'noise_level': 0.2}
    ]
    
    edge_cases = [
        {
            'name': 'Empty vector',
            'input': {'vector': {}}
        },
        {
            'name': 'Single feature',
            'input': {'vector': {list(base_input.get('vector', {}).keys())[0]: 1.0} if 'vector' in base_input else {}}
        }
    ]
    
    robustness_df = evaluator.evaluate_robustness(
        base_input,
        stress_scenarios=stress_scenarios,
        edge_cases=edge_cases
    )
    
    # Generate reports
    os.makedirs(args.output_dir, exist_ok=True)
    report_files = evaluator.generate_timing_report(args.output_dir)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    for key, path in report_files.items():
        print(f"  {key}: {path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

