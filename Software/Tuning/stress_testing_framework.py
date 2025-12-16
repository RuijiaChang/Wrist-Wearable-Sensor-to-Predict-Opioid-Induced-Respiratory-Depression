'''
Stress Testing Framework for 125 Hz PPG Data
Simulates controlled noise, missing samples, and signal distortion
Outputs:
- Stressed PPG signals 
- Metadata about applied stresses

Usage:
python Software/Tuning/stress_testing_framework.py
output_dir (default: 'stress_test/')
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import json
from scipy import signal
from scipy.interpolate import interp1d
import warnings
import os
warnings.filterwarnings('ignore')


class PPGStressTester:
    """
    Framework for stress testing PPG signal processing and prediction models
    with controlled noise, missing samples, and signal distortion
    """
    
    def __init__(self, sampling_rate: int = 125):
        """
        Initialize stress testing framework
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 125 Hz)
        """
        self.sampling_rate = sampling_rate
        self.test_results = []
        
    def add_white_noise(self, signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """
        Add white Gaussian noise to signal with specified SNR
        
        Args:
            signal: Input PPG signal
            snr_db: Signal-to-noise ratio in dB (higher = less noise)
        
        Returns:
            Noisy signal
        """
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise
    
    def add_motion_artifact(self, signal: np.ndarray, 
                           artifact_duration: float = 0.5,
                           artifact_amplitude: float = 2.0,
                           artifact_frequency: float = 2.0,
                           artifact_start: Optional[int] = None) -> np.ndarray:
        """
        Add motion artifact (low-frequency, high-amplitude disturbance)
        
        Args:
            signal: Input PPG signal
            artifact_duration: Duration of artifact in seconds
            artifact_amplitude: Amplitude multiplier for artifact
            artifact_frequency: Frequency of artifact oscillation in Hz
            artifact_start: Optional start index for artifact (if None, random)
        
        Returns:
            Signal with motion artifact
        """
        n_samples = len(signal)
        t = np.arange(n_samples) / self.sampling_rate
        
        # Create motion artifact as low-frequency oscillation
        artifact = artifact_amplitude * np.sin(2 * np.pi * artifact_frequency * t)
        
        # Apply artifact to segment (random or specified)
        if artifact_start is None:
            artifact_start = np.random.randint(0, int(n_samples - artifact_duration * self.sampling_rate))
        else:
            artifact_start = int(artifact_start)
            # Ensure it's within valid range
            max_start = int(n_samples - artifact_duration * self.sampling_rate)
            artifact_start = min(artifact_start, max(0, max_start))
        
        artifact_end = int(artifact_start + artifact_duration * self.sampling_rate)
        
        # Create smooth transition
        transition_samples = int(0.1 * self.sampling_rate)  # 100ms transition
        window = np.ones(n_samples)
        
        # Fade in
        if artifact_start + transition_samples <= n_samples:
            window[artifact_start:artifact_start+transition_samples] = np.linspace(0, 1, transition_samples)
        # Fade out
        if artifact_end - transition_samples >= 0:
            window[artifact_end-transition_samples:artifact_end] = np.linspace(1, 0, transition_samples)
        # Zero outside artifact region
        window[:artifact_start] = 0
        window[artifact_end:] = 0
        
        artifact_signal = artifact * window
        return signal + artifact_signal
    
    def add_baseline_drift(self, signal: np.ndarray, 
                          drift_rate: float = 0.1,
                          drift_type: str = 'linear') -> np.ndarray:
        """
        Add baseline drift to signal
        
        Args:
            signal: Input PPG signal
            drift_rate: Rate of drift (amplitude per second)
            drift_type: Type of drift ('linear', 'exponential', 'sinusoidal')
        
        Returns:
            Signal with baseline drift
        """
        n_samples = len(signal)
        t = np.arange(n_samples) / self.sampling_rate
        
        if drift_type == 'linear':
            drift = drift_rate * t
        elif drift_type == 'exponential':
            drift = drift_rate * (np.exp(t / 10) - 1)
        elif drift_type == 'sinusoidal':
            drift = drift_rate * np.sin(2 * np.pi * 0.1 * t)  # Very slow oscillation
        else:
            drift = drift_rate * t
        
        # Randomize drift direction
        if np.random.random() > 0.5:
            drift = -drift
        
        return signal + drift
    
    def add_amplitude_variation(self, signal: np.ndarray,
                               variation_percent: float = 20.0,
                               variation_type: str = 'gradual') -> np.ndarray:
        """
        Add amplitude variation (simulating sensor contact changes)
        
        Args:
            signal: Input PPG signal
            variation_percent: Percentage variation (0-100)
            variation_type: 'gradual', 'sudden', or 'periodic'
        
        Returns:
            Signal with amplitude variation
        """
        n_samples = len(signal)
        t = np.arange(n_samples) / self.sampling_rate
        
        variation_factor = variation_percent / 100.0
        
        if variation_type == 'gradual':
            # Gradual change over time
            multiplier = 1.0 + variation_factor * (2 * np.random.random() - 1) * (t / t[-1])
        elif variation_type == 'sudden':
            # Sudden change at random point
            change_point = np.random.randint(n_samples // 4, 3 * n_samples // 4)
            multiplier = np.ones(n_samples)
            multiplier[change_point:] = 1.0 + variation_factor * (2 * np.random.random() - 1)
        elif variation_type == 'periodic':
            # Periodic variation
            freq = 0.1  # 0.1 Hz = 10 second period
            multiplier = 1.0 + variation_factor * np.sin(2 * np.pi * freq * t)
        else:
            multiplier = np.ones(n_samples)
        
        return signal * multiplier
    
    def add_missing_samples(self, signal: np.ndarray,
                           missing_percent: float = 5.0,
                           missing_pattern: str = 'random') -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove samples from signal (simulating data loss)
        
        Args:
            signal: Input PPG signal
            missing_percent: Percentage of samples to remove (0-100)
            missing_pattern: 'random', 'burst', or 'periodic'
        
        Returns:
            Tuple of (signal with missing samples, mask indicating missing samples)
        """
        n_samples = len(signal)
        n_missing = int(n_samples * missing_percent / 100.0)
        
        if missing_pattern == 'random':
            # Random missing samples
            missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        elif missing_pattern == 'burst':
            # Burst of missing samples (simulating temporary disconnection)
            burst_start = np.random.randint(0, n_samples - n_missing)
            missing_indices = np.arange(burst_start, burst_start + n_missing)
        elif missing_pattern == 'periodic':
            # Periodic missing samples
            period = n_samples // n_missing if n_missing > 0 else n_samples
            missing_indices = np.arange(0, n_samples, period)[:n_missing]
        else:
            missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        
        # Create mask (True = valid, False = missing)
        mask = np.ones(n_samples, dtype=bool)
        mask[missing_indices] = False
        
        # Set missing samples to NaN
        signal_with_missing = signal.copy()
        signal_with_missing[~mask] = np.nan
        
        return signal_with_missing, mask
    
    def interpolate_missing_samples(self, signal: np.ndarray, 
                                   method: str = 'linear') -> np.ndarray:
        """
        Interpolate missing samples (NaN values)
        
        Args:
            signal: Signal with NaN values
            method: Interpolation method ('linear', 'cubic', 'spline')
        
        Returns:
            Interpolated signal
        """
        valid_mask = ~np.isnan(signal)
        valid_indices = np.where(valid_mask)[0]
        missing_indices = np.where(~valid_mask)[0]
        
        if len(missing_indices) == 0:
            return signal
        
        if len(valid_indices) < 2:
            # Not enough valid samples for interpolation, use mean
            return np.full_like(signal, np.nanmean(signal))
        
        if method == 'linear':
            interp_func = interp1d(valid_indices, signal[valid_indices], 
                                  kind='linear', 
                                  bounds_error=False, 
                                  fill_value='extrapolate')
        elif method == 'cubic':
            if len(valid_indices) >= 4:
                interp_func = interp1d(valid_indices, signal[valid_indices], 
                                      kind='cubic', 
                                      bounds_error=False, 
                                      fill_value='extrapolate')
            else:
                interp_func = interp1d(valid_indices, signal[valid_indices], 
                                      kind='linear', 
                                      bounds_error=False, 
                                      fill_value='extrapolate')
        else:  # spline
            from scipy.interpolate import UnivariateSpline
            interp_func = UnivariateSpline(valid_indices, signal[valid_indices], 
                                          s=0, ext='extrapolate')
        
        signal_interpolated = signal.copy()
        signal_interpolated[missing_indices] = interp_func(missing_indices)
        
        return signal_interpolated
    
    def add_frequency_distortion(self, signal: np.ndarray,
                                distortion_type: str = 'aliasing',
                                cutoff_factor: float = 0.5) -> np.ndarray:
        """
        Add frequency domain distortion
        
        Args:
            signal: Input PPG signal
            distortion_type: 'aliasing', 'filtering', or 'harmonic'
            cutoff_factor: Factor for cutoff frequency (0-1)
        
        Returns:
            Distorted signal
        """
        if distortion_type == 'aliasing':
            # Simulate aliasing by downsampling and upsampling
            downsample_factor = int(1 / cutoff_factor)
            downsampled = signal[::downsample_factor]
            # Upsample back using linear interpolation
            upsampled_indices = np.linspace(0, len(downsampled) - 1, len(signal))
            interp_func = interp1d(np.arange(len(downsampled)), downsampled, 
                                 kind='linear', 
                                 bounds_error=False, 
                                 fill_value='extrapolate')
            return interp_func(upsampled_indices)
        
        elif distortion_type == 'filtering':
            # Apply aggressive low-pass filtering
            nyquist = self.sampling_rate / 2
            cutoff = nyquist * cutoff_factor
            sos = signal.butter(4, cutoff, 'lp', fs=self.sampling_rate, output='sos')
            return signal.sosfiltfilt(sos, signal)
        
        elif distortion_type == 'harmonic':
            # Add harmonic distortion
            t = np.arange(len(signal)) / self.sampling_rate
            # Add second and third harmonics
            harmonic2 = 0.1 * np.sin(4 * np.pi * 1.0 * t)  # 2 Hz
            harmonic3 = 0.05 * np.sin(6 * np.pi * 1.0 * t)  # 3 Hz
            return signal + harmonic2 + harmonic3
        
        else:
            return signal
    
    def apply_stress_scenario(self, signal: np.ndarray,
                             scenario: Dict[str, any]) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Apply a complete stress scenario to signal
        
        Args:
            signal: Input PPG signal
            scenario: Dictionary defining stress scenario parameters
        
        Returns:
            Tuple of (stressed signal, metadata about applied stresses)
        """
        stressed_signal = signal.copy()
        metadata = {
            'original_length': len(signal),
            'applied_stresses': []
        }
        
        # Apply noise
        if scenario.get('add_noise', False):
            snr_db = scenario.get('noise_snr_db', 20.0)
            stressed_signal = self.add_white_noise(stressed_signal, snr_db)
            metadata['applied_stresses'].append(f'White noise (SNR: {snr_db} dB)')
        
        # Apply motion artifact
        if scenario.get('add_motion_artifact', False):
            artifact_duration = scenario.get('artifact_duration', 0.5)
            artifact_amplitude = scenario.get('artifact_amplitude', 2.0)
            artifact_frequency = scenario.get('artifact_frequency', 2.0)
            artifact_start = scenario.get('artifact_start', None)  # Allow specifying start position
            stressed_signal = self.add_motion_artifact(
                stressed_signal, artifact_duration, artifact_amplitude, artifact_frequency, artifact_start
            )
            metadata['applied_stresses'].append(
                f'Motion artifact (duration: {artifact_duration}s, amplitude: {artifact_amplitude}x)'
            )
            # Store artifact position for visualization
            if artifact_start is not None:
                metadata['artifact_start_time'] = artifact_start / self.sampling_rate
        
        # Apply baseline drift
        if scenario.get('add_baseline_drift', False):
            drift_rate = scenario.get('drift_rate', 0.1)
            drift_type = scenario.get('drift_type', 'linear')
            stressed_signal = self.add_baseline_drift(stressed_signal, drift_rate, drift_type)
            metadata['applied_stresses'].append(f'Baseline drift ({drift_type}, rate: {drift_rate})')
        
        # Apply amplitude variation
        if scenario.get('add_amplitude_variation', False):
            variation_percent = scenario.get('variation_percent', 20.0)
            variation_type = scenario.get('variation_type', 'gradual')
            stressed_signal = self.add_amplitude_variation(
                stressed_signal, variation_percent, variation_type
            )
            metadata['applied_stresses'].append(
                f'Amplitude variation ({variation_type}, {variation_percent}%)'
            )
        
        # Apply missing samples
        if scenario.get('add_missing_samples', False):
            missing_percent = scenario.get('missing_percent', 5.0)
            missing_pattern = scenario.get('missing_pattern', 'random')
            stressed_signal, mask = self.add_missing_samples(
                stressed_signal, missing_percent, missing_pattern
            )
            metadata['applied_stresses'].append(
                f'Missing samples ({missing_pattern}, {missing_percent}%)'
            )
            metadata['missing_mask'] = mask
            metadata['missing_count'] = np.sum(~mask)
            
            # Interpolate if requested
            if scenario.get('interpolate_missing', True):
                interpolation_method = scenario.get('interpolation_method', 'linear')
                stressed_signal = self.interpolate_missing_samples(
                    stressed_signal, interpolation_method
                )
                metadata['applied_stresses'].append(
                    f'Interpolation ({interpolation_method})'
                )
        
        # Apply frequency distortion
        if scenario.get('add_frequency_distortion', False):
            distortion_type = scenario.get('distortion_type', 'aliasing')
            cutoff_factor = scenario.get('cutoff_factor', 0.5)
            stressed_signal = self.add_frequency_distortion(
                stressed_signal, distortion_type, cutoff_factor
            )
            metadata['applied_stresses'].append(
                f'Frequency distortion ({distortion_type}, cutoff: {cutoff_factor})'
            )
        
        metadata['final_length'] = len(stressed_signal)
        metadata['signal_change'] = np.std(stressed_signal - signal)
        
        return stressed_signal, metadata
    
    def generate_test_scenarios(self) -> List[Dict[str, any]]:
        """
        Generate predefined test scenarios
        
        Returns:
            List of scenario dictionaries
        """
        scenarios = [
            # Baseline: no stress
            {
                'name': 'Baseline (No Stress)',
                'add_noise': False,
                'add_motion_artifact': False,
                'add_baseline_drift': False,
                'add_amplitude_variation': False,
                'add_missing_samples': False,
                'add_frequency_distortion': False
            },
            
            # Light noise
            {
                'name': 'Light Noise',
                'add_noise': True,
                'noise_snr_db': 30.0,
                'add_motion_artifact': False,
                'add_baseline_drift': False,
                'add_amplitude_variation': False,
                'add_missing_samples': False,
                'add_frequency_distortion': False
            },
            
            # Moderate noise
            {
                'name': 'Moderate Noise',
                'add_noise': True,
                'noise_snr_db': 20.0,
                'add_motion_artifact': False,
                'add_baseline_drift': False,
                'add_amplitude_variation': False,
                'add_missing_samples': False,
                'add_frequency_distortion': False
            },
            
            # Heavy noise
            {
                'name': 'Heavy Noise',
                'add_noise': True,
                'noise_snr_db': 10.0,
                'add_motion_artifact': False,
                'add_baseline_drift': False,
                'add_amplitude_variation': False,
                'add_missing_samples': False,
                'add_frequency_distortion': False
            },
            
            # Motion artifact
            {
                'name': 'Motion Artifact',
                'add_noise': False,
                'add_motion_artifact': True,
                'artifact_duration': 1.0,
                'artifact_amplitude': 2.0,
                'artifact_frequency': 2.0,
                'add_baseline_drift': False,
                'add_amplitude_variation': False,
                'add_missing_samples': False,
                'add_frequency_distortion': False
            },
            
            # Missing samples - random
            {
                'name': 'Missing Samples (Random 5%)',
                'add_noise': False,
                'add_motion_artifact': False,
                'add_baseline_drift': False,
                'add_amplitude_variation': False,
                'add_missing_samples': True,
                'missing_percent': 5.0,
                'missing_pattern': 'random',
                'interpolate_missing': True,
                'interpolation_method': 'linear',
                'add_frequency_distortion': False
            },
            
            # Missing samples - burst
            {
                'name': 'Missing Samples (Burst 10%)',
                'add_noise': False,
                'add_motion_artifact': False,
                'add_baseline_drift': False,
                'add_amplitude_variation': False,
                'add_missing_samples': True,
                'missing_percent': 10.0,
                'missing_pattern': 'burst',
                'interpolate_missing': True,
                'interpolation_method': 'linear',
                'add_frequency_distortion': False
            },
            
            # Baseline drift
            {
                'name': 'Baseline Drift',
                'add_noise': False,
                'add_motion_artifact': False,
                'add_baseline_drift': True,
                'drift_rate': 0.2,
                'drift_type': 'linear',
                'add_amplitude_variation': False,
                'add_missing_samples': False,
                'add_frequency_distortion': False
            },
            
            # Amplitude variation
            {
                'name': 'Amplitude Variation',
                'add_noise': False,
                'add_motion_artifact': False,
                'add_baseline_drift': False,
                'add_amplitude_variation': True,
                'variation_percent': 30.0,
                'variation_type': 'gradual',
                'add_missing_samples': False,
                'add_frequency_distortion': False
            },
            
            # Combined: moderate stress
            {
                'name': 'Combined Moderate Stress',
                'add_noise': True,
                'noise_snr_db': 20.0,
                'add_motion_artifact': True,
                'artifact_duration': 0.5,
                'artifact_amplitude': 1.5,
                'add_baseline_drift': True,
                'drift_rate': 0.1,
                'add_amplitude_variation': False,
                'add_missing_samples': True,
                'missing_percent': 3.0,
                'missing_pattern': 'random',
                'interpolate_missing': True,
                'add_frequency_distortion': False
            },
            
            # Combined: extreme stress
            {
                'name': 'Combined Extreme Stress',
                'add_noise': True,
                'noise_snr_db': 10.0,
                'add_motion_artifact': True,
                'artifact_duration': 2.0,
                'artifact_amplitude': 3.0,
                'add_baseline_drift': True,
                'drift_rate': 0.3,
                'drift_type': 'exponential',
                'add_amplitude_variation': True,
                'variation_percent': 40.0,
                'variation_type': 'sudden',
                'add_missing_samples': True,
                'missing_percent': 15.0,
                'missing_pattern': 'burst',
                'interpolate_missing': True,
                'add_frequency_distortion': True,
                'distortion_type': 'aliasing',
                'cutoff_factor': 0.6
            }
        ]
        
        return scenarios
    
    def visualize_stress_test(self, original: np.ndarray, stressed: np.ndarray,
                             metadata: Dict[str, any], 
                             duration: float = 5.0,
                             save_path: Optional[str] = None,
                             start_time: Optional[float] = None):
        """
        Visualize original vs stressed signal
        
        Args:
            original: Original signal
            stressed: Stressed signal
            metadata: Metadata about applied stresses
            duration: Duration to plot in seconds
            save_path: Optional path to save figure
            start_time: Optional start time in seconds (if None, starts from beginning or finds interesting region)
        """
        n_samples = min(len(original), len(stressed))
                
        # Smart start time selection: if not specified, try to find interesting region
        if start_time is None:
            # Special handling for missing samples: show region with missing samples
            if 'missing_mask' in metadata:
                missing_mask = metadata['missing_mask']
                missing_indices = np.where(~missing_mask)[0]
                if len(missing_indices) > 0:
                    # Find region with most missing samples
                    window_size = int(duration * self.sampling_rate)
                    max_missing_start = 0
                    max_missing_count = 0
                    for i in range(0, n_samples - window_size, int(0.5 * self.sampling_rate)):
                        missing_count = np.sum(~missing_mask[i:i+window_size])
                        if missing_count > max_missing_count:
                            max_missing_count = missing_count
                            max_missing_start = i
                    # If found region with missing samples, use it
                    if max_missing_count > 0:
                        start_idx = max_missing_start
                    else:
                        # Fall back to difference-based selection
                        diff = np.abs(stressed - original)
                        max_diff_start = 0
                        max_diff_sum = 0
                        for i in range(0, n_samples - window_size, int(0.5 * self.sampling_rate)):
                            diff_sum = np.sum(diff[i:i+window_size])
                            if diff_sum > max_diff_sum:
                                max_diff_sum = diff_sum
                                max_diff_start = i
                        start_idx = max_diff_start
                else:
                    # No missing samples, use difference-based selection
                    diff = np.abs(stressed - original)
                    window_size = int(duration * self.sampling_rate)
                    max_diff_start = 0
                    max_diff_sum = 0
                    for i in range(0, n_samples - window_size, int(0.5 * self.sampling_rate)):
                        diff_sum = np.sum(diff[i:i+window_size])
                        if diff_sum > max_diff_sum:
                            max_diff_sum = diff_sum
                            max_diff_start = i
                    start_idx = max_diff_start
            else:
                # Check if there's a significant difference (e.g., motion artifact)
                diff = np.abs(stressed - original)
                # Find region with maximum difference
                window_size = int(duration * self.sampling_rate)
                if window_size < n_samples:
                    max_diff_start = 0
                    max_diff_sum = 0
                    for i in range(0, n_samples - window_size, int(0.5 * self.sampling_rate)):  # Check every 0.5s
                        diff_sum = np.sum(diff[i:i+window_size])
                        if diff_sum > max_diff_sum:
                            max_diff_sum = diff_sum
                            max_diff_start = i
                    start_idx = max_diff_start
                else:
                    start_idx = 0
        else:
            start_idx = int(start_time * self.sampling_rate)
        
        start_idx = max(0, min(start_idx, n_samples - 1))
        samples_to_plot = int(duration * self.sampling_rate)
        end_idx = min(start_idx + samples_to_plot, n_samples)
        actual_samples = end_idx - start_idx
        
        t_original = np.arange(len(original)) / self.sampling_rate
        t_stressed = np.arange(len(stressed)) / self.sampling_rate
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot original
        axes[0].plot(t_original[start_idx:end_idx], original[start_idx:end_idx], 
                    'b-', linewidth=1.5, label='Original Signal')
        axes[0].set_title('Original Signal', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot stressed
        axes[1].plot(t_stressed[start_idx:end_idx], stressed[start_idx:end_idx], 
                    'r-', linewidth=1.5, label='Stressed Signal', alpha=0.8)
        title_stresses = metadata.get("applied_stresses", ["None"])
        if isinstance(title_stresses, list) and len(title_stresses) > 0:
            title_text = ', '.join(str(s) for s in title_stresses[:2])  # Show first 2 stresses
            if len(title_stresses) > 2:
                title_text += '...'
        else:
            title_text = str(title_stresses)
        axes[1].set_title(f'Stressed Signal: {title_text}', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot overlay
        axes[2].plot(t_original[start_idx:end_idx], original[start_idx:end_idx], 
                    'b-', linewidth=1.5, label='Original', alpha=0.6)
        axes[2].plot(t_stressed[start_idx:end_idx], stressed[start_idx:end_idx], 
                    'r-', linewidth=1.5, label='Stressed', alpha=0.8)
        
        # Highlight missing samples if available
        if 'missing_mask' in metadata:
            missing_mask = metadata['missing_mask']
            missing_in_range = ~missing_mask[start_idx:end_idx]
            if np.any(missing_in_range):
                missing_indices_in_range = np.where(missing_in_range)[0] + start_idx
                missing_times = t_original[missing_indices_in_range]
                missing_values_original = original[missing_indices_in_range]
                axes[2].scatter(missing_times, missing_values_original, 
                              c='orange', s=30, marker='x', linewidths=2,
                              label='Missing Samples (Original)', zorder=5)
        
        axes[2].set_title('Overlay Comparison', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def load_ppg_signal_from_csv(csv_path: str, 
                            time_col: str = 'Time [s]',
                            value_col: str = ' PLETH',
                            duration: Optional[float] = None) -> np.ndarray:
    """
    Load PPG signal from CSV file
    
    Args:
        csv_path: Path to CSV file
        time_col: Name of time column
        value_col: Name of signal value column
        duration: Optional duration to load in seconds (None = load all)
    
    Returns:
        PPG signal array
    """
    df = pd.read_csv(csv_path)
    
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Columns '{time_col}' or '{value_col}' not found in CSV")
    
    if duration:
        max_time = df[time_col].iloc[0] + duration
        df = df[df[time_col] <= max_time]
    
    signal = df[value_col].values
    return signal


def generate_synthetic_ppg_signal(duration: float = 30.0,
                                  sampling_rate: int = 125,
                                  heart_rate: float = 72.0) -> np.ndarray:
    """
    Generate synthetic PPG signal for testing
    
    Args:
        duration: Signal duration in seconds
        sampling_rate: Sampling rate in Hz
        heart_rate: Heart rate in beats per minute
    
    Returns:
        Synthetic PPG signal
    """
    n_samples = int(duration * sampling_rate)
    t = np.arange(n_samples) / sampling_rate
    
    # Generate basic PPG waveform
    # PPG typically has a systolic peak and diastolic notch
    hr_rad_per_sec = 2 * np.pi * heart_rate / 60.0
    
    # Main cardiac cycle
    signal = np.sin(hr_rad_per_sec * t)
    
    # Add systolic peak (sharper)
    systolic_phase = (hr_rad_per_sec * t) % (2 * np.pi)
    systolic_peak = 0.5 * np.exp(-((systolic_phase - np.pi/2) ** 2) / 0.1)
    
    # Add diastolic notch
    diastolic_phase = (hr_rad_per_sec * t) % (2 * np.pi)
    diastolic_notch = 0.2 * np.exp(-((diastolic_phase - 3*np.pi/4) ** 2) / 0.05)
    
    # Combine components
    signal = signal + systolic_peak + diastolic_notch
    
    # Add some baseline variation
    baseline = 0.1 * np.sin(2 * np.pi * 0.1 * t)  # Very slow variation
    
    # Normalize
    signal = signal + baseline
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    return signal


def main():
    """
    Example usage of stress testing framework
    """
    print("=== PPG Stress Testing Framework ===\n")
    
    # Initialize tester
    tester = PPGStressTester(sampling_rate=125)
    
    # Generate or load test signal
    print("Generating synthetic PPG signal...")
    test_signal = generate_synthetic_ppg_signal(duration=30.0, sampling_rate=125, heart_rate=72.0)
    print(f"Generated signal: {len(test_signal)} samples ({len(test_signal)/125:.1f} seconds)\n")
    
    # Get test scenarios
    scenarios = tester.generate_test_scenarios()
    print(f"Generated {len(scenarios)} test scenarios\n")
    
    # Run stress tests
    results = []
    for i, scenario in enumerate(scenarios):
        print(f"Running scenario {i+1}/{len(scenarios)}: {scenario['name']}")
        
        # Apply stress
        stressed_signal, metadata = tester.apply_stress_scenario(test_signal, scenario)
        
        # Calculate metrics
        mse = np.mean((test_signal[:len(stressed_signal)] - stressed_signal) ** 2)
        correlation = np.corrcoef(test_signal[:len(stressed_signal)], stressed_signal)[0, 1]
        
        result = {
            'scenario_name': scenario['name'],
            'mse': float(mse),
            'correlation': float(correlation),
            'metadata': metadata
        }
        results.append(result)
        
        print(f"  MSE: {mse:.6f}, Correlation: {correlation:.4f}")
        print(f"  Applied stresses: {len(metadata['applied_stresses'])}")
        print()
    
    # Visualize a few scenarios
    print("Generating visualizations...")
    visualize_scenarios = [0, 1, 4, 5, 9]  # Baseline, Light Noise, Motion, Missing, Combined
    
    for idx in visualize_scenarios:
        if idx < len(scenarios):
            scenario = scenarios[idx]
            stressed_signal, metadata = tester.apply_stress_scenario(test_signal, scenario)
            
            output_dir = "stress_test/signals/"
            os.makedirs(output_dir, exist_ok=True)

            save_path = f"{output_dir}stress_test_{scenario['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
            tester.visualize_stress_test(test_signal, stressed_signal, metadata, 
                                       duration=5.0, save_path=save_path)
    
    # Save results
    results_df = pd.DataFrame([
        {
            'scenario': r['scenario_name'],
            'mse': r['mse'],
            'correlation': r['correlation'],
            'num_stresses': len(r['metadata']['applied_stresses'])
        }
        for r in results
    ])
    
    results_df.to_csv(output_dir+'stress_test_results.csv', index=False)
    print(f"\nResults saved to: stress_test_results.csv")
    
    # Print summary
    print("\n=== Stress Test Summary ===")
    print(results_df.to_string(index=False))
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

