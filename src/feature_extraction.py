"""
Temporal Feature Engineering and Sequence Generation
=====================================================
Transforms raw telemetry into rolling-window summary features and
constructs 3D sliding-window sequential tensors for recurrent models.
"""

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.data_pipeline import LOW_VARIANCE_SENSORS


class FeatureEngineer:
    def __init__(self, sequence_length: int = 30, rolling_windows: List[int] = [5, 15]):
        self.sequence_length = sequence_length
        self.rolling_windows = rolling_windows
        self.scaler = MinMaxScaler()
        self.active_sensor_cols: List[str] = []
        self.feature_cols: List[str] = []

    def fit_transform_train(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Fits scaler and generates rolling statistics for training data."""
        df = train_df.copy()
        all_sensors = [c for c in df.columns if c.startswith("s_")]
        self.active_sensor_cols = [c for c in all_sensors if c not in LOW_VARIANCE_SENSORS]

        # Fit and normalize raw active sensors
        df[self.active_sensor_cols] = self.scaler.fit_transform(df[self.active_sensor_cols])

        # Compute rolling window temporal statistics
        rolling_features = []
        for w in self.rolling_windows:
            for s in self.active_sensor_cols:
                mean_col = f"{s}_roll_mean_{w}"
                std_col = f"{s}_roll_std_{w}"
                
                df[mean_col] = (
                    df.groupby("unit_id")[s]
                    .rolling(window=w, min_periods=1)
                    .mean()
                    .reset_index(drop=True)
                )
                df[std_col] = (
                    df.groupby("unit_id")[s]
                    .rolling(window=w, min_periods=1)
                    .std()
                    .fillna(0.0)
                    .reset_index(drop=True)
                )
                rolling_features.extend([mean_col, std_col])

        self.feature_cols = self.active_sensor_cols + rolling_features
        return df, self.feature_cols

    def transform_test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted normalization and temporal feature calculations to test data."""
        df = test_df.copy()
        df[self.active_sensor_cols] = self.scaler.transform(df[self.active_sensor_cols])

        for w in self.rolling_windows:
            for s in self.active_sensor_cols:
                mean_col = f"{s}_roll_mean_{w}"
                std_col = f"{s}_roll_std_{w}"
                df[mean_col] = (
                    df.groupby("unit_id")[s]
                    .rolling(window=w, min_periods=1)
                    .mean()
                    .reset_index(drop=True)
                )
                df[std_col] = (
                    df.groupby("unit_id")[s]
                    .rolling(window=w, min_periods=1)
                    .std()
                    .fillna(0.0)
                    .reset_index(drop=True)
                )
        return df

    def create_lstm_sequences(
        self, df: pd.DataFrame, is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates 3D sequences [samples, sequence_length, features] for LSTM modeling.
        If is_train=True, slides a window over all engine cycles.
        If is_train=False, extracts the latest sequence for each unit.
        """
        seq_list = []
        target_list = []

        feature_subset = self.active_sensor_cols

        for unit, group in df.groupby("unit_id"):
            group_feats = group[feature_subset].values
            
            if is_train:
                group_targets = group["RUL_clipped"].values
                num_records = len(group)
                if num_records >= self.sequence_length:
                    for i in range(num_records - self.sequence_length + 1):
                        seq_list.append(group_feats[i : i + self.sequence_length])
                        target_list.append(group_targets[i + self.sequence_length - 1])
                else:
                    pad = np.repeat(group_feats[:1], self.sequence_length - num_records, axis=0)
                    seq_list.append(np.vstack([pad, group_feats]))
                    target_list.append(group_targets[-1])
            else:
                num_records = len(group)
                if num_records >= self.sequence_length:
                    seq_list.append(group_feats[-self.sequence_length :])
                else:
                    pad = np.repeat(group_feats[:1], self.sequence_length - num_records, axis=0)
                    seq_list.append(np.vstack([pad, group_feats]))
                target_list.append(0)

        return np.array(seq_list, dtype=np.float32), np.array(target_list, dtype=np.float32)
