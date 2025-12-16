# **Wearable SpO₂ & RR Forecasting – Software System Overview**

This project develops a **wrist-wearable physiological forecasting system** that leverages photoplethysmography (PPG) signals to perform **real-time and multi-horizon prediction** of blood oxygen saturation (SpO₂) and respiratory rate (RR), enabling early detection of **opioid-induced respiratory depression (OIRD)**.

The system builds upon a **Gaussian Process Regression (GPR) core model** developed by the Spring 2025 team, which is **currently under patent review**. To ensure IP compliance, all software contributions described below were designed as **modular, externally attachable components** that **do not modify or expose the protected GPR logic**.

I was solely responsible for the **software system design, implementation, evaluation, and integration**.

## **Summary of Software Contributions**

1. **Feature-Optimized GPR Wrapper for RR Estimation**
   Built a non-invasive RR optimization pipeline using automated feature selection (ReliefF, LASSO, TreeFS) and z-score outlier filtering, evaluated strictly via the original `Run_GPR()` interface; **reduced RR MAE from 1.07 → 0.86 (~20% improvement)** without modifying the patented GPR core.

2. **XGBoost-Based Multi-Horizon Forecasting (10–300s)**
   Implemented an XGBoost forecasting model trained on the **BIDMC dataset** to predict future SpO₂ and RR across **10–300+ second horizons**, enabling early physiological risk detection beyond instantaneous monitoring.

3. **Transformer-Based Temporal Encoder for PPG Sequences**
   Designed a Transformer-based temporal encoder to extract **multi-scale temporal features** from high-frequency PPG sequences, improving **long-horizon prediction stability** under noisy and distorted inputs.

4. **Unified Real-Time Inference Pipeline**
   Developed a production-ready inference pipeline integrating **GPR (instantaneous prediction)** and **Transformer–XGBoost (future forecasting)**, supporting **single-vector and time-series inputs** and outputting **standardized JSON predictions** for downstream system integration.

5. **Variability & Stress-Testing Framework (125 Hz Simulation)**
   Built a comprehensive stress-testing framework simulating **125 Hz wearable input** under controlled noise, 5% missing samples, and motion artifacts; validated robustness across **11 stress scenarios** with **stable feature extraction (125–131 features)** and zero inference failures.

6. **Latency, Stability, and Reliability Evaluation**
   Evaluated end-to-end pipeline performance under diverse stress conditions, achieving **low-latency inference (mean 4.6–5.5 ms)**, **low timing jitter (CV 0.03–0.14)**, **bounded latency (4.2–6.8 ms)**, and **100% functional reliability** across all test cases.
