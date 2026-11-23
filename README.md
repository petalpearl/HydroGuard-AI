# HydroGuard-AI
# 🛡️ HydroGuard AI — Predictive Maintenance for Industrial Hydraulic Systems

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-189AB4)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-9ACD32)](https://lightgbm.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/UCI%20Dataset-ID%20447%20(Real%20Sensors)-success)](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)

A leak-free, multi-sensor predictive maintenance pipeline for diagnosing internal hydraulic pump leakage using real experimental test-rig data (`HydroGuardAI.ipynb`). Built for **Artificial Intelligence in MAE (BMA 304)** under mentor **Dr. Pooja Bhati**.

---

## 🎯 Problem & Motivation

Industrial hydraulic systems power automated manufacturing lines, presses, and packaging machinery. Undetected **internal pump leakage** reduces volumetric efficiency, increases electric motor power draw (`EPS1`), and triggers unplanned downtime.

**HydroGuard AI** addresses three core engineering challenges:
* **Multi-Rate Sensor Fusion:** Combines 17 physical sensors sampled at 100 Hz, 10 Hz, and 1 Hz across 60-second operating cycles.
* **Zero Data Leakage:** Fits collinearity pruning ($|r| > 0.95$), `StandardScaler`, and **SMOTE** class balancing strictly on the training split, keeping validation and test sets untouched.
* **ML vs. Deep Learning Benchmark:** Evaluates regularized tabular models against a **2-Layer PyTorch LSTM** on 1 Hz time-series sequences and auto-selects the best model on validation metrics.

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------+
| 1. Raw UCI Hydraulic Rig Data (ID: 447)                               |
|    2,205 real 60s cycles x 17 sensors (PS1-PS6, EPS1, FS1-FS2,         |
|    TS1-TS4, VS1, CE, CP, SE) | Target: Pump_Leakage (0, 1, 2)       |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| 2. Dual Feature Engineering                                           |
|    - 85 Cycle Stats (mean, std, min, max, drift)                      |
|    - 4 Fluid-Power Physics Features (Power Proxy, dP, dT, Power/Flow) |
|    - (2205, 60, 17) 1 Hz Sequence Tensor for PyTorch LSTM             |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| 3. Stratified Split (70% Train / 15% Val / 15% Test)                  |
|    Train: 1,543 cycles  |  Val: 331 cycles  |  Test: 331 cycles       |
+-----------------------------------------------------------------------+
          |                                                 |
          v (Tabular path)                                v (Sequence path)
+-----------------------------------+     +-----------------------------+
| 4. Train-Only Preprocessing       |     | 2-Layer PyTorch LSTM        |
|    - Drop 48 collinear (|r|>0.95) |     | Dropout (0.3) + Weight      |
|      -> 41 features kept          |     | Decay + Class Weights       |
|    - StandardScaler (fit on train)|     +-----------------------------+
|    - SMOTE -> 855/class on train  |                    |
+-----------------------------------+                    |
                  |                                      |
                  +------------------+-------------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
| 5. Validation Leaderboard & Auto-Selection                            |
|    Winner: Random Forest (Regularized) -> 99.094% Val Acc, 0.988 F1 |
+-----------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
| 6. Held-Out Test & Live Plant-Floor Alerts                            |
|    Test Acc: 98.19% (325/331) | Macro F1: 0.9742                      |
|    Alerts: HEALTHY | WARNING (Weak Leak) | CRITICAL (Severe Leak)     |
+-----------------------------------------------------------------------+
