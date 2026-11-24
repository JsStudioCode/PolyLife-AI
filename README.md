# PolyLife AI — AI-Based Plastic Degradation Assessment & Sustainable Recycling Decision Support System

Real-time interactive web application combining chemistry-informed photo-oxidation kinetics, Scikit-Learn predictive models, multi-criteria **TOPSIS** recycling pathway optimization, **Kaggle** dataset integration, and **Firebase Cloud Firestore** live synchronization.

## Architecture

- **Module 1 (Degradation Stage Classifier)**: `RandomForestClassifier` predicting `Low` / `Moderate` / `Severe` degradation from polymer type (`HDPE`, `LDPE`, `PP`, `PS`, `PET`, `PVC`), Carbonyl Index (`CI`), Hydroxyl Index (`HI`), and crystallinity gain (`ΔXc`).
- **Module 2 ($M_w$ Retention & Kinetics)**: `GradientBoostingRegressor` predicting molecular weight retention and $t_{50}$ half-life from exposure time `t`, UV irradiance `I`, temperature `T`, and relative humidity `RH`, benchmarked against Arrhenius photo-oxidation and dose-only kinetics.
- **Module 3 (TOPSIS Decision Support)**: Real-time multi-criteria ranking across `Mechanical recycling`, `Solvent purification`, `Chemical recycling`, and `Energy recovery` using customizable weights (`Yield`, `Cost`, `CO₂e`, `Energy`, `Quality`).
- **Real-Time Sync**: Firebase Cloud Firestore (`onSnapshot`) via [firebase-bridge.js](static/firebase-bridge.js) + local Server-Sent Events (`/api/stream`) and SQLite persistence.
- **Kaggle Dataset Hub**: Direct Kaggle API downloader (`owner/dataset-name`), CSV drag-and-drop uploader, automatic feature mapping, and live ML retraining.

## Quick Start

```bash
pip install -r requirements.txt
python3 app.py
```

Open `http://127.0.0.1:8765` in your browser.

## Firebase Setup

1. Create or select a Firebase project and enable **Cloud Firestore**.
2. Deploy the included rules and indexes (`firestore.rules` & `firestore.indexes.json`):
   ```bash
   npx -y firebase-tools@latest login
   npx -y firebase-tools@latest use <YOUR_PROJECT_ID>
   npx -y firebase-tools@latest deploy --only firestore
   ```
3. In the PolyLife AI UI, click **Sync: Local Real-Time (SSE)** in the top bar and paste your Firebase Web App `firebaseConfig` JSON.
