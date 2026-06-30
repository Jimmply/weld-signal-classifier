"""
Laser Weld In-Process Signal Simulator
=======================================
Simulates real-time photodiode and acoustic emission (AE) signals
captured during pulsed Nd:YAG laser welding.

Physical basis:
  Photodiode    — captures optical emission from weld pool and plasma plume.
                  Amplitude correlates with melt pool size and keyhole stability.
  Acoustic RMS  — captures elastic waves from spatter events, solidification
                  cracking, and keyhole collapse.
  Back-reflection — fraction of laser light reflected back from weld pool surface;
                  rises when keyhole collapses (porosity precursor).

Defect signatures:
  Good          — stable photodiode, low AE background
  Spatter       — sharp transient spikes in photodiode + AE
  Porosity      — periodic oscillation in back-reflection (keyhole instability)
  Cracking      — high-frequency AE burst during solidification phase
  Lack_of_Fusion— reduced photodiode amplitude (insufficient melt pool)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFECT_TYPES   = ["Good", "Spatter", "Porosity", "Cracking", "Lack_of_Fusion"]
SIGNAL_COLS    = ["photodiode_v", "acoustic_rms_mv", "back_reflection_pct"]
SAMPLE_RATE_HZ = 10_000          # 10 kHz — typical for Nd:YAG in-process monitoring
WELD_DURATION_S = 0.50           # 500 ms weld pass

# Frequency band boundaries (Hz) chosen to match physical defect signatures:
#   Low  0–500 Hz   — keyhole oscillation band (porosity precursor)
#   Mid  500–2000 Hz — intermediate transient zone
#   High 2000–5000 Hz — solidification cracking, spatter impulse content
_BAND_LOW_HZ  = (10,   500)
_BAND_HIGH_HZ = (2000, 5000)


def _spectral_features(signal: np.ndarray, sample_rate: int) -> dict:
    """
    Compute frequency-domain features via real FFT.

    Removing the DC component before FFT prevents the mean amplitude from
    dominating the spectral centroid and band power calculations.
    """
    sig_ac   = signal - signal.mean()
    fft_mag  = np.abs(np.fft.rfft(sig_ac))
    freqs    = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)

    total_power = fft_mag.sum() + 1e-12

    # Dominant frequency — skip DC bin (index 0)
    dominant_freq = float(freqs[1:][np.argmax(fft_mag[1:])]) if len(fft_mag) > 1 else 0.0

    # Spectral centroid — power-weighted mean frequency
    spectral_centroid = float((fft_mag * freqs).sum() / total_power)

    # Spectral entropy — how spread or tonal the spectrum is
    psd_norm = fft_mag / total_power
    spectral_entropy = float(
        -np.sum(psd_norm * np.log(psd_norm + 1e-12)) / np.log(max(len(psd_norm), 2))
    )

    # Band power fractions
    low_mask  = (freqs >= _BAND_LOW_HZ[0])  & (freqs <= _BAND_LOW_HZ[1])
    high_mask = (freqs >= _BAND_HIGH_HZ[0]) & (freqs <= _BAND_HIGH_HZ[1])
    low_band_pct  = float(fft_mag[low_mask].sum()  / total_power)
    high_band_pct = float(fft_mag[high_mask].sum() / total_power)

    return {
        "dominant_freq_hz":   dominant_freq,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_entropy":   spectral_entropy,
        "low_band_power_pct": low_band_pct,
        "high_band_power_pct": high_band_pct,
    }


@dataclass
class SignalConfig:
    sample_rate_hz: int   = SAMPLE_RATE_HZ
    weld_duration_s: float = WELD_DURATION_S
    photodiode_nominal_v: float  = 2.5   # V
    acoustic_nominal_mv: float   = 15.0  # mV
    back_ref_nominal_pct: float  = 8.0   # %


@dataclass
class WeldSignal:
    weld_id: str
    defect_type: str
    df: pd.DataFrame      # time-series: time_s, photodiode_v, acoustic_rms_mv, back_reflection_pct, is_event


class WeldSignalGenerator:
    """
    Generates time-series weld signals for a single weld pass.

    Parameters
    ----------
    config : SignalConfig, optional
    defect_type : str
        One of DEFECT_TYPES. None → random weighted selection.
    random_seed : int
    """

    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        defect_type: Optional[str] = None,
        random_seed: int = 42,
    ) -> None:
        self.config = config or SignalConfig()
        self.defect_type = defect_type
        self._rng = np.random.default_rng(random_seed)

    def generate(self, weld_id: str = "WELD-000") -> WeldSignal:
        cfg = self.config
        n   = int(cfg.sample_rate_hz * cfg.weld_duration_s)
        t   = np.linspace(0, cfg.weld_duration_s, n)

        # Baseline signals
        photo = self._base_signal(cfg.photodiode_nominal_v,   sigma=0.05, n=n)
        acous = self._base_signal(cfg.acoustic_nominal_mv,    sigma=1.0,  n=n)
        backr = self._base_signal(cfg.back_ref_nominal_pct,   sigma=0.5,  n=n)
        event = np.zeros(n, dtype=bool)

        defect = self.defect_type or self._random_defect()

        if defect == "Spatter":
            photo, acous, event = self._inject_spatter(photo, acous, event, n)
        elif defect == "Porosity":
            backr, event = self._inject_porosity(backr, event, n)
        elif defect == "Cracking":
            acous, event = self._inject_cracking(acous, event, n)
        elif defect == "Lack_of_Fusion":
            photo, event = self._inject_lof(photo, event, n)

        df = pd.DataFrame({
            "time_s":             t.round(6),
            "photodiode_v":       np.clip(photo, 0, 10).round(4),
            "acoustic_rms_mv":    np.clip(acous, 0, 500).round(3),
            "back_reflection_pct": np.clip(backr, 0, 100).round(3),
            "is_event":           event,
        })
        return WeldSignal(weld_id=weld_id, defect_type=defect, df=df)

    # ------------------------------------------------------------------

    def _base_signal(self, mean: float, sigma: float, n: int) -> np.ndarray:
        noise = self._rng.normal(0, sigma, n)
        # Slow OU drift
        drift = np.zeros(n)
        for i in range(1, n):
            drift[i] = drift[i-1] * 0.98 + self._rng.normal(0, sigma * 0.1)
        return mean + noise + drift

    def _inject_spatter(self, photo, acous, event, n):
        """Spatter: sharp transient spikes in both photodiode and AE."""
        n_events = self._rng.integers(5, 20)
        locs = self._rng.integers(int(n * 0.05), n, size=n_events)
        for loc in locs:
            width = self._rng.integers(3, 15)
            amp   = self._rng.uniform(1.5, 5.0)
            idx   = slice(loc, min(loc + width, n))
            photo[idx] += amp * np.exp(-np.linspace(0, 3, width)[:idx.stop-idx.start])
            acous[idx] += amp * 60 * self._rng.uniform(0.5, 1.5)
            event[idx]  = True
        return photo, acous, event

    def _inject_porosity(self, backr, event, n):
        """Porosity: periodic keyhole oscillation visible in back-reflection."""
        freq_hz = self._rng.uniform(200, 800)
        t = np.linspace(0, self.config.weld_duration_s, n)
        start = self._rng.integers(int(n * 0.2), int(n * 0.5))
        duration = self._rng.integers(int(n * 0.2), int(n * 0.5))
        end = min(start + duration, n)
        osc = 3.5 * np.sin(2 * np.pi * freq_hz * t[start:end])
        backr[start:end] += osc
        event[start:end]  = True
        return backr, event

    def _inject_cracking(self, acous, event, n):
        """Solidification cracking: high-frequency AE burst near end of weld."""
        burst_start = self._rng.integers(int(n * 0.65), int(n * 0.85))
        burst_len   = self._rng.integers(int(n * 0.05), int(n * 0.15))
        burst_end   = min(burst_start + burst_len, n)
        # Exponentially decaying burst
        burst = 120 * np.exp(-np.linspace(0, 4, burst_end - burst_start))
        burst += self._rng.normal(0, 15, burst_end - burst_start)
        acous[burst_start:burst_end] += burst
        event[burst_start:burst_end]  = True
        return acous, event

    def _inject_lof(self, photo, event, n):
        """Lack of fusion: progressively declining photodiode amplitude."""
        start = self._rng.integers(int(n * 0.1), int(n * 0.4))
        ramp  = np.linspace(0, 1, n - start)
        photo[start:] -= 1.8 * ramp + self._rng.normal(0, 0.05, n - start)
        event[start:]  = True
        return photo, event

    def _random_defect(self) -> str:
        weights = [0.40, 0.20, 0.20, 0.10, 0.10]
        return self._rng.choice(DEFECT_TYPES, p=weights)


class WeldFleetSignalGenerator:
    """Generates a fleet of labeled weld signals for model training."""

    def __init__(self, n_welds: int = 500, random_seed: int = 42) -> None:
        self.n_welds = n_welds
        self._rng = np.random.default_rng(random_seed)

    def generate_summary(self) -> pd.DataFrame:
        """
        Return a feature-extracted summary (one row per weld) suitable for
        XGBoost classification without storing full time series.
        """
        rows = []
        defect_weights = [0.40, 0.20, 0.20, 0.10, 0.10]

        for i in range(self.n_welds):
            defect = self._rng.choice(DEFECT_TYPES, p=defect_weights)
            seed   = int(self._rng.integers(0, 99999))
            gen    = WeldSignalGenerator(defect_type=defect, random_seed=seed)
            ws     = gen.generate(weld_id=f"WELD-{i:04d}")
            rows.append(self._extract_features(ws))

        df = pd.DataFrame(rows)
        logger.info(
            "Generated %d weld signals. Defect distribution:\n%s",
            self.n_welds,
            df["defect_type"].value_counts().to_string(),
        )
        return df

    def _extract_features(self, ws: WeldSignal) -> dict:
        d = ws.df
        feats = {"weld_id": ws.weld_id, "defect_type": ws.defect_type}
        for col in SIGNAL_COLS:
            s = d[col]
            # Time domain
            feats[f"{col}_mean"]      = float(s.mean())
            feats[f"{col}_std"]       = float(s.std())
            feats[f"{col}_max"]       = float(s.max())
            feats[f"{col}_kurtosis"]  = float(s.kurtosis())
            feats[f"{col}_skewness"]  = float(s.skew())
            feats[f"{col}_rms"]       = float(np.sqrt((s**2).mean()))
            feats[f"{col}_p2p"]       = float(s.max() - s.min())
            # Frequency domain
            spec = _spectral_features(s.values, SAMPLE_RATE_HZ)
            for k, v in spec.items():
                feats[f"{col}_{k}"] = v
        feats["event_rate"]           = float(d["is_event"].mean())
        feats["n_events"]             = int(d["is_event"].sum())
        return feats
