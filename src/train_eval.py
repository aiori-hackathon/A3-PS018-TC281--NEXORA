
"""Synthetic-data model training and baseline evaluation."""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit


FEATURES = [
    "distance_km",
    "frequency_ghz",
    "elevation_deg",
    "rain_mm_h",
    "cloud_loss_db",
    "gas_loss_db",
    "drift_m",
    "jitter_deg",
    "step",
]


def make_dataset(n=3500, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for scenario_id in range(n):
        distance = rng.uniform(30, 450)
        frequency = rng.choice([2., 6., 12., 20., 28., 38.])
        elevation = rng.uniform(8, 90)
        rain = rng.uniform(0, 80)
        cloud = rng.uniform(0, 8)
        gas = rng.uniform(0, 3)
        drift = rng.uniform(0, 180)
        jitter = rng.uniform(0, 2.5)

        noise_dbw = (
            -228.6
            + 10 * np.log10(290.0)
            + 10 * np.log10(20e6)
        )

        base = (
            50 + 15
            - (
                92.45
                + 20 * np.log10(distance)
                + 20 * np.log10(frequency)
            )
            - noise_dbw
        )

        rain_loss = min(
            35,
            0.018 * rain**0.85
            * (frequency / 20)**1.35
            / max(
                np.sin(np.radians(max(elevation, 5))),
                0.087,
            ),
        )

        point = min(12, 0.15 * jitter**2 + 0.008 * drift)

        for step in range(10):
            snr = (
                base
                - rain_loss
                - cloud
                - gas
                - point
                + 0.8 * np.sin(step * 0.17)
                + rng.normal(0, 0.5)
            )

            rows.append({
                "scenario_id": scenario_id,
                "distance_km": distance,
                "frequency_ghz": frequency,
                "elevation_deg": elevation,
                "rain_mm_h": rain,
                "cloud_loss_db": cloud,
                "gas_loss_db": gas,
                "drift_m": drift,
                "jitter_deg": jitter,
                "step": step,
                "snr_db": snr,
            })

    return pd.DataFrame(rows)


def train_and_evaluate(seed=42):
    df = make_dataset(seed=seed)

    # Keep complete scenarios separate between training and testing.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df["scenario_id"])
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(train[FEATURES], train["snr_db"])
    pred = model.predict(test[FEATURES])

    # Mean baseline: predicts the training-set average SNR.
    mean_pred = np.full(len(test), train["snr_db"].mean())

    threshold = 5.0

    actual_out = (
        test["snr_db"].to_numpy() < threshold
    ).astype(int)

    pred_out = (pred < threshold).astype(int)

    baseline_out = (mean_pred < threshold).astype(int)

    metrics = pd.DataFrame([
        {
            "Model": "Training-set mean baseline",
            "MAE (dB)": mean_absolute_error(
                test["snr_db"], mean_pred
            ),
            "RMSE (dB)": np.sqrt(
                mean_squared_error(test["snr_db"], mean_pred)
            ),
            "Outage F1": f1_score(
                actual_out, baseline_out, zero_division=0
            ),
            "Outage recall": recall_score(
                actual_out, baseline_out, zero_division=0
            ),
        },
        {
            "Model": "Random Forest regressor",
            "MAE (dB)": mean_absolute_error(
                test["snr_db"], pred
            ),
            "RMSE (dB)": np.sqrt(
                mean_squared_error(test["snr_db"], pred)
            ),
            "Outage F1": f1_score(
                actual_out, pred_out, zero_division=0
            ),
            "Outage recall": recall_score(
                actual_out, pred_out, zero_division=0
            ),
        },
    ])

    return {
        "model": model,
        "mae_db": float(
            mean_absolute_error(test["snr_db"], pred)
        ),
        "metrics": metrics.round(3),
        "features": FEATURES,
    }


def forecast_next(series, model, horizon=5):
    last = series.iloc[-1]
    rows = []

    for i in range(horizon):
        row = {
            name: float(last[name])
            for name in FEATURES
            if name != "step"
        }

        row["step"] = float(last["step"] + i + 1)
        rows.append(row)

    x = pd.DataFrame(rows, columns=FEATURES)
    values = model.predict(x).tolist()

    return {
        "values": values,
        "min_forecast_snr_db": float(min(values)),
    }
