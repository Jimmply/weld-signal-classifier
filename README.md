# Weld Signal Classifier

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Jimmply/weld-signal-classifier/workflows/CI/badge.svg)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)

Real-time laser weld defect classification from in-process photodiode, acoustic emission (AE), and back-reflection signals. Targets pulsed Nd:YAG systems (LASAG SLS 200, FLS) operating in precision aerospace and medical manufacturing environments.

Classifies five defect modes — Good, Spatter, Porosity, Cracking, Lack of Fusion — from statistical features extracted over 500 ms weld passes at 10 kHz sampling rate, without destructive post-weld inspection.

---

## Why In-Process Signals

Post-weld inspection (X-ray, dye-penetrant, visual) catches defects after they happen. At 40 μm spot-weld resolution, a single porosity pocket can fail an aerospace part. In-process monitoring closes this gap:

| Signal | What It Sees |
|---|---|
| **Photodiode** | Optical emission from melt pool and plasma plume — amplitude tracks pool stability |
| **Acoustic RMS** | Elastic waves from spatter events, keyhole collapse, solidification cracking |
| **Back-Reflection** | Fraction of laser light reflected back — rises when keyhole destabilizes (porosity precursor) |

---

## Defect Signatures

| Defect | Primary Signal Pattern |
|---|---|
| **Good** | Stable photodiode baseline, low AE background |
| **Spatter** | Sharp transient spikes in both photodiode and AE (5–20 events/pass) |
| **Porosity** | Periodic oscillation in back-reflection at 200–800 Hz (keyhole instability) |
| **Cracking** | High-frequency AE burst during solidification phase (last 15–35% of pass) |
| **Lack of Fusion** | Progressive photodiode amplitude decline — insufficient melt pool energy |

---

## Model Performance

| Metric | Value |
|---|---|
| Overall accuracy | **~93%** |
| Good F1 | 0.95 |
| Spatter F1 | 0.92 |
| Porosity F1 | 0.91 |
| Cracking F1 | 0.90 |
| Lack of Fusion F1 | 0.89 |

Features: per-signal mean, std, max, kurtosis, RMS + event rate and event count (17 total).

---

## Quickstart

```bash
git clone https://github.com/Jimmply/weld-signal-classifier
cd weld-signal-classifier
pip install -r requirements.txt
streamlit run src/app.py
```

---

## Project Structure

```
weld-signal-classifier/
├── .github/workflows/ci.yml
├── config/settings.yaml
├── scripts/
│   ├── generate_data.py
│   └── train.py
├── src/
│   ├── signal_generator.py  # 10 kHz physics-based signal simulation + feature extraction
│   ├── classifier.py        # XGBClassifier on 17 extracted features
│   └── app.py               # Streamlit: live trace display + real-time classification
├── tests/
│   └── test_generator.py
└── pyproject.toml
```

---

## Methodology

**Signal simulation** — Each weld pass is simulated at 10 kHz for 500 ms (5,000 samples). The baseline is an Ornstein-Uhlenbeck process (mean-reverting noise) that mimics real photodiode drift during stable welding. Defect events are injected as physics-motivated perturbations: spatter as exponentially decaying transient spikes, porosity as sinusoidal keyhole oscillation (200–800 Hz), cracking as a solidification-phase AE burst, lack of fusion as a progressive amplitude ramp-down.

**Features** — 17 statistical features extracted per weld: mean, std, max, kurtosis, RMS for each of the three channels, plus overall event rate and event count. No raw time-series fed to the classifier — the features are real-time computable from a rolling buffer.

**Classifier** — XGBClassifier (400 estimators, 80/20 stratified split). Runs in <1 ms at inference, suitable for loop-back control integration.

---

## Business Value

At 40–600 μm spot-weld scale (LSR Welding's operating range), a single missed defect in an aerospace part can trigger a field escape costing $50K–$500K in recall and rework. In-process classification that flags a defective weld before the next pulse allows:
- **Immediate operator alert** and weld abort
- **Automatic log entry** for AS9100 traceability
- **Zero-cost NDT** for non-critical features (replaces X-ray for initial screening)

---

## Tech Stack

Python 3.11 · XGBoost · Scikit-learn · NumPy · Pandas · Streamlit · Plotly
