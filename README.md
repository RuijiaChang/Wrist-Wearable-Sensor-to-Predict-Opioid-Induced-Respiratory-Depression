## **Wearable Spo2 RR Forecasting**

This project aims to develop a wrist-wearable prediction system based on photoplethysmography (PPG) signals for the early detection of **Opioid-Induced Respiratory Depression (OIRD)**. The long-term goal is to build an integrated wearable platform that combines **hardware sensing** and **software prediction modules**, enabling real-time and multi-horizon forecasting of **blood oxygen saturation (SpO₂)** and **respiratory rate (RR)** to support continuous patient monitoring and timely clinical intervention.

For the software component, the **core algorithm** developed by the Spring 2025 team—based on **Gaussian Process Regression (GPR)** —is fully protected and currently under patent review. As such, this semester’s work focuses exclusively on **modular algorithmic enhancement** without modifying the core logic. All extensions must be externally attachable modules that operate independently of the protected model.

### **Summary of Software Contributions (Fall 2025)**

1. **RR Feature Optimization (Non-invasive enhancement of core GPR model)**
   Using the existing company-provided codebase—without altering any protected components—I developed an automated RR optimization pipeline (`optimize_rr_gpr.py`).

   * Applied z-score outlier filtering and three feature selection methods (ReliefF, LASSO, TreeFS).
   * Evaluated each candidate feature set using the original `Run_GPR()` interface.
   * **Reduced RR MAE from 1.07 → 0.86**, demonstrating improved RR prediction *without modifying the core GPR model*.

2. **Future Multi-Window Prediction Module (Transformer + XGBoost)**
   Implemented a new modular forecasting framework (`real_future_prediction.py`) that predicts SpO₂ and RR across multiple future horizons (10–300+ seconds).

   * Designed a Transformer-based temporal encoder to capture long-range dependencies in PPG-derived features.
   * Combined Transformer embeddings with XGBoost regressors to improve multi-horizon predictive performance.
   * Fully modular: operates as an external component without altering the protected GPR structure.

3. **Unified Inference Pipeline**
   Developed `infer.py` as a consolidated inference interface that:

   * Loads the trained multi-window model.
   * Optionally incorporates the original GPR model for “current value” prediction.
   * Generates future SpO₂ and RR predictions from either a single feature vector or a time-sequence CSV.
   * Outputs standardized JSON-format predictions for downstream system integration.

Together, these enhancements directly support the semester’s goals—
**(i) extending prediction beyond 30 seconds with preserved accuracy**,
**(ii) improving RR prediction performance**, and
**(iii) maintaining strict modularity under the project’s IP protection constraints.**
