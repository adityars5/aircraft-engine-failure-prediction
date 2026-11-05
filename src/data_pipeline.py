"""
Data Ingestion and Preprocessing Pipeline
=========================================
Handles C-MAPSS turbofan dataset ingestion, ground truth RUL assignment,
piecewise target clipping, and synthetic telemetry generation.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


COLUMN_NAMES = (
    ["unit_id", "time_cycles", "setting_1", "setting_2", "setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)

# Sensors with zero or negligible variance in standard single-condition sub-datasets
LOW_VARIANCE_SENSORS = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]


def calculate_rul(df: pd.DataFrame, max_rul_clip: Optional[int] = 125) -> pd.DataFrame:
    """
    Computes Remaining Useful Life (RUL) for each unit cycle.
    Applies piecewise-linear thresholding to prevent overfitting on healthy baseline cycles.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_id")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    
    df = df.merge(max_cycles, on="unit_id", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df.drop(columns=["max_cycle"], inplace=True)
    
    if max_rul_clip is not None:
        df["RUL_clipped"] = df["RUL"].clip(upper=max_rul_clip)
    else:
        df["RUL_clipped"] = df["RUL"]
        
    return df


def generate_synthetic_cmapss(
    output_dir: str = "sample_data",
    num_train_units: int = 100,
    num_test_units: int = 100
) -> Tuple[str, str, str]:
    """
    Generates synthetic turbofan degradation records matching the C-MAPSS schema.
    Simulates exponential wear, thermal degradation, and sensor noise across 21 channels.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    def simulate_fleet(num_units: int, is_test: bool = False):
        records = []
        test_truth = []
        
        for unit in range(1, num_units + 1):
            total_life = np.random.randint(130, 360)
            if is_test:
                observed_cycles = np.random.randint(35, total_life - 10)
                remaining_life = total_life - observed_cycles
                test_truth.append(remaining_life)
            else:
                observed_cycles = total_life

            for cycle in range(1, observed_cycles + 1):
                wear = math.exp((cycle / total_life) * 2.8) / math.exp(2.8)
                
                # Base sensor profiles with realistic noise and degradation slopes
                s_vals = []
                for s_idx in range(1, 22):
                    if f"s_{s_idx}" in LOW_VARIANCE_SENSORS:
                        val = 100.0 + s_idx * 5.0 + np.random.normal(0, 0.005)
                    elif s_idx in [2, 3, 4, 8, 11, 13, 15]:
                        # Rising sensors (temperatures, static pressures, bleed flow)
                        val = 400.0 + s_idx * 15.0 + wear * 85.0 + np.random.normal(0, 1.2)
                    elif s_idx in [7, 12, 20, 21]:
                        # Decreasing sensors (fan efficiency, coolant ratios)
                        val = 600.0 - s_idx * 10.0 - wear * 65.0 + np.random.normal(0, 1.1)
                    else:
                        val = 200.0 + s_idx * 8.0 + (wear if s_idx % 2 == 0 else -wear) * 25.0 + np.random.normal(0, 0.8)
                    s_vals.append(val)

                row = [unit, cycle, 0.0, 0.0, 100.0] + s_vals
                records.append(row)

        df = pd.DataFrame(records, columns=COLUMN_NAMES)
        return df, test_truth

    train_df, _ = simulate_fleet(num_train_units, is_test=False)
    test_df, test_truth = simulate_fleet(num_test_units, is_test=True)

    train_path = os.path.join(output_dir, "train_FD001.txt")
    test_path = os.path.join(output_dir, "test_FD001.txt")
    truth_path = os.path.join(output_dir, "RUL_FD001.txt")

    train_df.to_csv(train_path, sep=" ", index=False, header=False)
    test_df.to_csv(test_path, sep=" ", index=False, header=False)
    pd.DataFrame(test_truth).to_csv(truth_path, index=False, header=False)

    return train_path, test_path, truth_path


def load_dataset(data_dir: str = "sample_data") -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Loads train, test, and true test RUL arrays."""
    train_file = os.path.join(data_dir, "train_FD001.txt")
    test_file = os.path.join(data_dir, "test_FD001.txt")
    truth_file = os.path.join(data_dir, "RUL_FD001.txt")

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print("[INFO] Dataset not found. Generating initial simulation files...")
        generate_synthetic_cmapss(output_dir=data_dir)

    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    truth_rul = pd.read_csv(truth_file, header=None).values.flatten()

    train_df = calculate_rul(train_df)
    return train_df, test_df, truth_rul
