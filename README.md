# ✈️ Aircraft Engine Failure Prediction & RUL Estimation

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end prognostic health monitoring system designed to forecast the Remaining Useful Life (RUL) of aircraft turbofan engines using sensor time-series data, Random Forest regressors, and PyTorch LSTM recurrent sequence models.

---

## 📌 Project Summary

| Metric / Parameter | Project Specifications |
| :--- | :--- |
| **Domain** | Aerospace Predictive Maintenance (Fleet Prognostics) |
| **Dataset** | Multi-sensor Turbofan Engine Degradation Simulation (FD001) |
| **Sensor Features** | 21 sensor channels + 3 operational settings |
| **Baseline Architecture** | Random Forest Regressor with rolling temporal features |
| **Sequential Model** | 2-layer Deep LSTM Network (PyTorch) |
| **Evaluation Metrics** | RMSE, MAE, NASA Asymmetric Penalty Score, R² Score |
| **User Interface** | Interactive Gradio Fleet Health Dashboard |

---

## 📊 Performance Benchmarks

Evaluated across a test fleet of 100 aircraft turbofan engines:

| Architecture | Test RMSE (Cycles) | MAE (Cycles) | NASA Score | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 25.14 | 20.32 | 1140.5 | 0.5821 |
| **Random Forest Baseline** | 18.32 | 14.21 | 512.4 | 0.7780 |
| **Deep LSTM (Sequential)** | **14.76** | **11.08** | **338.2** | **0.8562** |

---

## 📈 Evaluation Trajectory Preview

![RUL Predictions](assets/rul_comparison_chart.png)

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["Raw C-MAPSS Telemetry<br>(21 Sensors + 3 Flight Settings)"] --> B["Sensor Variance Filtering<br>(Drop Uninformative Channels)"]
    B --> C["Temporal Feature Engineering<br>(Rolling Mean & Std: 5, 15 Cycles)"]
    C --> D["Piecewise Linear RUL Target<br>(Capped at 125 Cycles)"]
    D --> E1["Random Forest Regressor<br>(RMSE: 18.32 | Score: 512.4)"]
    D --> E2["PyTorch Sequential LSTM<br>(RMSE: 14.76 | Score: 338.2)"]
    E1 --> F["Aerospace Evaluation<br>(RMSE, MAE, Asymmetric NASA Score)"]
    E2 --> F
    F --> G["Gradio Fleet Monitoring Dashboard<br>(app.py)"]
```

---

## 📁 Repository Structure

```
├── configs/
│   └── config.yaml             # Model hyperparameters and sensor channel mappings
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py        # Dataset ingestion, synthetic generator, RUL calculation
│   ├── feature_extraction.py   # Rolling window statistics & sliding sequence tensor builder
│   ├── models.py               # Random Forest baseline & PyTorch TurbofanLSTM class
│   └── evaluation.py           # RMSE, MAE, R², and asymmetric NASA scoring metrics
├── sample_data/                # Sample turbofan test datasets
├── notebooks/
│   └── aircraft_engine_rul_prediction.ipynb  # End-to-end Colab GPU training notebook
├── assets/
│   └── rul_comparison_chart.png# Benchmark visualization
├── app.py                      # Interactive Gradio fleet health application
├── requirements.txt            # Project dependencies
├── LICENSE                     # MIT License
└── README.md                   # Technical documentation
```

---

## ⚡ Quickstart

```bash
# Clone the repository
git clone https://github.com/adityars5/aircraft-engine-failure-prediction.git
cd aircraft-engine-failure-prediction

# Install requirements
pip install -r requirements.txt

# Launch interactive Gradio fleet dashboard
python app.py
```

Open your browser at `http://127.0.0.1:7861` to test the live fleet diagnostic monitor.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.

Developed by **[adityars5](https://github.com/adityars5)**.
