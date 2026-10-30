"""Documented, simplified HAPS link-budget simulator for prototype use.

This is an educational approximation, not a complete ITU-R P.618/P.676/P.840
implementation. The attenuation coefficients are intentionally transparent
demo assumptions and must be calibrated/replaced for scientific evaluation.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

@dataclass
class Scenario:
    distance_km: float = 80.0
    frequency_ghz: float = 28.0
    elevation_deg: float = 45.0
    rain_mm_h: float = 10.0
    cloud_loss_db: float = 1.0
    gas_loss_db: float = 0.5
    drift_m: float = 15.0
    jitter_deg: float = 0.4
    eirp_dbw: float = 50.0
    gt_db_k: float = 15.0
    bandwidth_hz: float = 20e6

def fspl_db(distance_km: float, frequency_ghz: float) -> float:
    """Free-space path loss in dB, distance in km and frequency in GHz."""
    d = max(float(distance_km), 0.001)
    f = max(float(frequency_ghz), 0.001)
    return 92.45 + 20.0 * np.log10(d) + 20.0 * np.log10(f)

def simplified_rain_loss_db(rain_mm_h: float, frequency_ghz: float, elevation_deg: float) -> float:
    """Transparent heuristic rain loss; replace with ITU-R P.618/P.837 calculations."""
    rain = max(0.0, float(rain_mm_h))
    freq_factor = (max(frequency_ghz, 1.0) / 20.0) ** 1.35
    path_factor = 1.0 / max(np.sin(np.radians(max(elevation_deg, 5.0))), 0.087)
    return min(35.0, 0.018 * (rain ** 0.85) * freq_factor * path_factor)

def link_budget_snr_db(s: Scenario, step: int = 0, rng=None) -> float:
    """Calculate a simplified SNR estimate from a link budget."""
    rng = rng or np.random.default_rng(0)
    # Small deterministic time-varying motion terms make a series to forecast.
    phase = step * 0.17
    drift_variation = abs(s.drift_m * (0.55 + 0.45 * np.sin(phase)))
    motion_distance_km = np.sqrt(s.distance_km**2 + (drift_variation / 1000.0)**2)
    fspl = fspl_db(motion_distance_km, s.frequency_ghz)
    rain_loss = simplified_rain_loss_db(s.rain_mm_h, s.frequency_ghz, s.elevation_deg)
    pointing_loss = min(12.0, 0.15 * s.jitter_deg**2 + 0.008 * drift_variation)
    bandwidth_noise_dbw = -228.6 + 10 * np.log10(290.0) + 10 * np.log10(max(s.bandwidth_hz, 1.0))
    snr = (s.eirp_dbw + s.gt_db_k - fspl - rain_loss - s.cloud_loss_db
           - s.gas_loss_db - pointing_loss - bandwidth_noise_dbw)
    return float(snr + rng.normal(0.0, 0.35))

def simulate_series(scenario: Scenario, steps: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic SNR time series."""
    rng = np.random.default_rng(seed)
    rows = []
    for step in range(steps):
        snr = link_budget_snr_db(scenario, step, rng)
        rows.append({"step": step, **asdict(scenario), "snr_db": snr,
                     "outage": int(snr < 5.0)})
    return pd.DataFrame(rows)
