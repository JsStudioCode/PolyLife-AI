"""
PolyLife AI: Core ML, Photo-Oxidation Kinetics, TOPSIS Decision Engine, and Kaggle Dataset Adapter.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"
DB_PATH = BASE_DIR / "polylife_rt.db"
FIREBASE_CFG_PATH = BASE_DIR / "firebase_config.json"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

R = 8.314
EA = 30e3
K0 = 2.0e-4

# Chemistry-informed polymer photo-oxidation parameters:
# (k_rel, CI_max, HI_max, dXc_max, density_g_cm3, resin_code, full_name)
POLY_PARAMS: dict[str, dict[str, Any]] = {
    "HDPE": {"k_rel": 0.80, "ci_max": 2.2, "hi_max": 0.9, "dxc_max": 6.0, "code": 2, "name": "High-Density Polyethylene", "color": "#0ea5e9"},
    "LDPE": {"k_rel": 1.00, "ci_max": 2.6, "hi_max": 1.1, "dxc_max": 5.0, "code": 4, "name": "Low-Density Polyethylene", "color": "#06b6d4"},
    "PP":   {"k_rel": 1.60, "ci_max": 3.2, "hi_max": 1.3, "dxc_max": 7.0, "code": 5, "name": "Polypropylene", "color": "#10b981"},
    "PS":   {"k_rel": 1.30, "ci_max": 1.8, "hi_max": 0.8, "dxc_max": 0.0, "code": 6, "name": "Polystyrene", "color": "#f59e0b"},
    "PET":  {"k_rel": 0.50, "ci_max": 1.2, "hi_max": 0.6, "dxc_max": 8.0, "code": 1, "name": "Polyethylene Terephthalate", "color": "#6366f1"},
    "PVC":  {"k_rel": 0.70, "ci_max": 1.5, "hi_max": 0.7, "dxc_max": 2.0, "code": 3, "name": "Polyvinyl Chloride", "color": "#ec4899"},
}
POLY_NAMES = list(POLY_PARAMS.keys())
STAGES = ["Low", "Moderate", "Severe"]
ROUTES = ["Mechanical recycling", "Solvent purification", "Chemical recycling", "Energy recovery"]
DEFAULT_WEIGHTS = [0.30, 0.15, 0.20, 0.10, 0.25]  # yield, cost, CO2e, energy, quality
CRITERIA_NAMES = ["Yield", "Cost (INR/kg)", "CO2e (kg/kg)", "Energy (MJ/kg)", "Quality"]
BENEFICIAL = np.array([True, False, False, False, True], dtype=bool)

KAGGLE_CURATED_CATALOG = [
    {
        "id": "xlpe-aging-prediction",
        "title": "XLPE Polymer Aging & Degradation Dataset",
        "slug": "search:xlpe+aging+prediction",
        "kaggle_url": "https://www.kaggle.com/datasets?search=XLPE+Aging+Prediction",
        "category": "Polymer Aging & Kinetics",
        "description": "Hydrothermal & chemical aging records tracking mechanical, electrical, and physicochemical degradation over exposure hours.",
        "expected_columns": ["polymer", "aging_hours", "temperature", "humidity", "retention"],
        "polylife_mapping": "Maps aging time, temperature, and property retention directly to Module 2 Mw-retention regression.",
    },
    {
        "id": "polymer-composite-properties",
        "title": "Polymer Composite Properties Dataset",
        "slug": "search:polymer+composite+properties",
        "kaggle_url": "https://www.kaggle.com/datasets?search=Polymer+Composite+Properties",
        "category": "Material & Thermal Properties",
        "description": "Polymer matrices, curing conditions, thermal degradation temperatures, tensile strength, and elongation at break.",
        "expected_columns": ["polymer_type", "temperature", "tensile_strength", "thermal_degradation"],
        "polylife_mapping": "Calibrates polymer susceptibility priors and mechanical quality retention curves across resin classes.",
    },
    {
        "id": "global-plastic-waste-2023",
        "title": "Global Plastic Waste 2023: Country-wise Data",
        "slug": "search:global+plastic+waste+2023",
        "kaggle_url": "https://www.kaggle.com/datasets?search=Global+Plastic+Waste+2023",
        "category": "Macro Recycling & Waste Streams",
        "description": "Country-level plastic waste generation, recycling efficiency rates, mismanagement risk, and coastal/landfill share.",
        "expected_columns": ["Country", "Total_Plastic_Waste_MT", "Recycling_Rate", "Per_Capita_Waste_KG"],
        "polylife_mapping": "Benchmarks regional recycling recovery targets and contextualizes TOPSIS sustainability weights.",
    },
    {
        "id": "plastic-waste-classification",
        "title": "Plastic Type Classification (PET, HDPE, LDPE, PP)",
        "slug": "search:plastic+waste+classification+PET+HDPE+LDPE+PP",
        "kaggle_url": "https://www.kaggle.com/datasets?search=plastic+waste+classification+PET+HDPE",
        "category": "Resin Identification & Sorting",
        "description": "Labeled polymer stream metadata & resin distribution across PET, HDPE, LDPE, PP, PS, and PVC sorting lines.",
        "expected_columns": ["resin_type", "label", "condition", "contamination"],
        "polylife_mapping": "Seeds real-world feedstock resin distributions and batch stream simulation profiles.",
    },
]


def compute_kinetics(poly: str, t: float, I: float, T: float, RH: float) -> dict[str, float]:
    """Compute theoretical photo-oxidation rate k, Mw retention, CI, HI, and dXc."""
    p = POLY_PARAMS[poly]
    k = (
        p["k_rel"]
        * K0
        * I
        * float(np.exp(-EA / R * (1.0 / (T + 273.15) - 1.0 / 303.15)))
        * (1.0 + 0.004 * (RH - 50.0))
    )
    kt = k * t
    ret = float(np.clip(np.exp(-kt), 0.02, 1.0))
    frac = 1.0 - float(np.exp(-1.4 * kt))
    ci = float(p["ci_max"] * frac)
    hi = float(p["hi_max"] * frac)
    dxc = float(p["dxc_max"] * frac)
    return {"k": float(k), "ret": ret, "CI": ci, "HI": hi, "dXc": dxc}


def generate_chemistry_dataset(n: int = 1800, seed: int = 42) -> pd.DataFrame:
    """Generate the chemistry-informed UV/thermal photo-oxidation dataset matching polylife.py."""
    rng = np.random.default_rng(seed)
    poly = rng.choice(POLY_NAMES, n)
    t = rng.uniform(0, 2000, n)
    I = rng.uniform(0.6, 1.4, n)
    T = rng.uniform(30, 70, n)
    RH = rng.uniform(20, 90, n)
    k = (
        np.array([POLY_PARAMS[p]["k_rel"] for p in poly])
        * K0
        * I
        * np.exp(-EA / R * (1.0 / (T + 273.15) - 1.0 / 303.15))
        * (1.0 + 0.004 * (RH - 50.0))
    )
    kt = k * t
    ret = np.clip(np.exp(-kt) * (1.0 + rng.normal(0, 0.03, n)), 0.02, 1.0)
    frac = 1.0 - np.exp(-1.4 * kt)
    CI = np.clip(np.array([POLY_PARAMS[p]["ci_max"] for p in poly]) * frac + rng.normal(0, 0.12, n), 0.0, None)
    HI = np.clip(np.array([POLY_PARAMS[p]["hi_max"] for p in poly]) * frac + rng.normal(0, 0.08, n), 0.0, None)
    dXc = np.array([POLY_PARAMS[p]["dxc_max"] for p in poly]) * frac + rng.normal(0, 0.8, n)
    stage = np.where(ret >= 0.80, 0, np.where(ret >= 0.50, 1, 2))
    return pd.DataFrame(
        {
            "poly": poly,
            "t": t,
            "I": I,
            "T": T,
            "RH": RH,
            "CI": CI,
            "HI": HI,
            "dXc": dXc,
            "ret": ret,
            "stage": stage,
        }
    )


def route_attributes(pol: str, ret_: float) -> dict[str, list[float]]:
    """Compute [yield, cost, CO2e, energy, quality] for each recycling route."""
    q = min(1.0, max(0.05, float(ret_)))
    mech_y = 0.92 * (q ** 0.8)
    mech_q = q ** (2.4 if pol == "PET" else 1.5)
    chem_ok = {"PET": 1.0, "PS": 0.95, "HDPE": 0.85, "LDPE": 0.85, "PP": 0.85, "PVC": 0.60}[pol]
    sol_ok = {"PS": 1.0, "PP": 0.85, "HDPE": 0.80, "LDPE": 0.80, "PET": 0.55, "PVC": 0.75}[pol]
    return {
        "Mechanical recycling": [round(mech_y, 4), 18.0, 0.45, 6.0, round(mech_q, 4)],
        "Solvent purification": [round(0.85 * sol_ok * (q ** 0.35), 4), 38.0, 1.10, 14.0, round(0.90 * sol_ok * (q ** 0.5), 4)],
        "Chemical recycling":   [round(0.75 * chem_ok, 4), 55.0, 1.90, 22.0, round(0.88 * chem_ok, 4)],
        "Energy recovery":      [0.25, 8.0, 2.60, 3.0, 0.05],
    }


def topsis_score(pol: str, ret_: float, weights: list[float] | np.ndarray | None = None) -> dict[str, Any]:
    """Run TOPSIS multi-criteria decision analysis for a polymer and Mw retention."""
    w = np.array(weights if weights is not None else DEFAULT_WEIGHTS, dtype=float)
    if w.sum() <= 0:
        w = np.array(DEFAULT_WEIGHTS, dtype=float)
    w = w / w.sum()
    attrs = route_attributes(pol, ret_)
    A = np.array([attrs[r] for r in ROUTES], dtype=float)
    norms = np.linalg.norm(A, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    V = (A / norms) * w
    ideal = np.where(BENEFICIAL, V.max(axis=0), V.min(axis=0))
    anti = np.where(BENEFICIAL, V.min(axis=0), V.max(axis=0))
    dp = np.linalg.norm(V - ideal, axis=1)
    dm = np.linalg.norm(V - anti, axis=1)
    denom = np.where((dp + dm) == 0, 1.0, dp + dm)
    closeness = dm / denom
    ranked = sorted(
        [
            {
                "route": ROUTES[i],
                "score": round(float(closeness[i]), 4),
                "yield": attrs[ROUTES[i]][0],
                "cost_inr": attrs[ROUTES[i]][1],
                "co2e_kg": attrs[ROUTES[i]][2],
                "energy_mj": attrs[ROUTES[i]][3],
                "quality": attrs[ROUTES[i]][4],
            }
            for i in range(len(ROUTES))
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    return {
        "best_route": ranked[0]["route"],
        "best_score": ranked[0]["score"],
        "scores": [round(float(s), 4) for s in closeness],
        "ranked": ranked,
        "weights_normalized": [round(float(x), 4) for x in w],
    }


class PolyLifeEngine:
    """Thread-safe ML + Kinetics + TOPSIS + Kaggle Dataset engine."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.dataset_meta: dict[str, Any] = {
            "source": "Chemistry-Informed Photo-Oxidation Benchmark (Built-in)",
            "kaggle_slug": None,
            "rows": 1800,
            "loaded_at": time.time(),
            "mode": "chemistry_standard",
        }
        self.df: pd.DataFrame = pd.DataFrame()
        self.rf: RandomForestClassifier | None = None
        self.lr_base: LogisticRegression | None = None
        self.gb: GradientBoostingRegressor | None = None
        self.kp: dict[str, float] = {}
        self.x1_cols: list[str] = []
        self.x2_cols: list[str] = []
        self.metrics: dict[str, Any] = {}
        self.imported_datasets: dict[str, pd.DataFrame] = {}
        self._init_db()
        self._load_or_train_initial()

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assessments (
                    id TEXT PRIMARY KEY,
                    sample_name TEXT,
                    batch_id TEXT,
                    poly TEXT,
                    exposure_t REAL,
                    uv_I REAL,
                    temp_C REAL,
                    rel_humidity REAL,
                    CI REAL,
                    HI REAL,
                    dXc REAL,
                    ret REAL,
                    stage TEXT,
                    stage_probs TEXT,
                    route TEXT,
                    topsis_scores TEXT,
                    source TEXT,
                    created_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_logs (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    source TEXT,
                    rows INTEGER,
                    columns TEXT,
                    status TEXT,
                    created_at REAL
                )
                """
            )
            conn.commit()

    def _load_or_train_initial(self) -> None:
        cache_file = CACHE_DIR / "polylife_models.joblib"
        if cache_file.exists():
            try:
                bundle = joblib.load(cache_file)
                self.df = bundle["df"]
                self.rf = bundle["rf"]
                self.lr_base = bundle["lr_base"]
                self.gb = bundle["gb"]
                self.kp = bundle["kp"]
                self.x1_cols = bundle["x1_cols"]
                self.x2_cols = bundle["x2_cols"]
                self.metrics = bundle["metrics"]
                self.dataset_meta = bundle.get("dataset_meta", self.dataset_meta)
                return
            except Exception:
                pass
        df = generate_chemistry_dataset(1800, seed=42)
        self.train_on_dataframe(df, source_label="Chemistry-Informed Photo-Oxidation Benchmark (Built-in)", save_cache=True)

    def train_on_dataframe(
        self,
        df: pd.DataFrame,
        source_label: str = "Custom Dataset",
        kaggle_slug: str | None = None,
        save_cache: bool = False,
    ) -> dict[str, Any]:
        """Train Module 1 (RF) and Module 2 (GB) and compute evaluation telemetry."""
        with self._lock:
            df = df.copy().reset_index(drop=True)
            # Ensure required columns & valid polymer names
            df["poly"] = df["poly"].astype(str).str.upper().where(df["poly"].astype(str).str.upper().isin(POLY_NAMES), "PP")
            for col, default in [("t", 500.0), ("I", 1.0), ("T", 45.0), ("RH", 50.0), ("CI", 0.5), ("HI", 0.2), ("dXc", 1.5), ("ret", 0.75)]:
                if col not in df.columns:
                    df[col] = default
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
            df["ret"] = df["ret"].clip(0.02, 1.0)
            if "stage" not in df.columns:
                df["stage"] = np.where(df["ret"] >= 0.80, 0, np.where(df["ret"] >= 0.50, 1, 2))
            else:
                df["stage"] = pd.to_numeric(df["stage"], errors="coerce").fillna(1).astype(int).clip(0, 2)

            oh = pd.get_dummies(df["poly"]).reindex(columns=POLY_NAMES, fill_value=0).astype(float)

            # Module 1: Degradation-stage classification
            X1 = pd.concat([oh, df[["CI", "HI", "dXc"]]], axis=1)
            y1 = df["stage"]
            can_stratify = y1.value_counts().min() >= 4 and len(y1.unique()) >= 2
            Xtr, Xte, ytr, yte = train_test_split(
                X1, y1, test_size=0.25, random_state=1, stratify=y1 if can_stratify else None
            )
            rf = RandomForestClassifier(n_estimators=240, random_state=1, n_jobs=-1).fit(Xtr, ytr)
            p1 = rf.predict(Xte)
            cv_folds = min(5, int(y1.value_counts().min())) if len(y1.unique()) > 1 else 2
            if cv_folds >= 2:
                cv = cross_val_score(
                    RandomForestClassifier(n_estimators=180, random_state=1, n_jobs=-1),
                    X1,
                    y1,
                    cv=StratifiedKFold(cv_folds, shuffle=True, random_state=1),
                    scoring="accuracy",
                )
                cv_mean, cv_sd = float(cv.mean()), float(cv.std())
            else:
                cv_mean, cv_sd = float(accuracy_score(yte, p1)), 0.0

            lr = LogisticRegression(max_iter=2000).fit(Xtr[["CI"]], ytr)
            base_acc = float(accuracy_score(yte, lr.predict(Xte[["CI"]])))
            pr, rc, f1, sup = precision_recall_fscore_support(yte, p1, labels=[0, 1, 2], zero_division=0)

            # Module 2: Mw-retention prediction
            X2 = pd.concat([oh, df[["t", "I", "T", "RH"]]], axis=1)
            y2 = df["ret"]
            Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.25, random_state=2)
            gb = GradientBoostingRegressor(n_estimators=320, max_depth=3, learning_rate=0.05, random_state=2).fit(Xtr2, ytr2)
            pg = gb.predict(Xte2)

            kp: dict[str, float] = {}
            trp = df.loc[Xtr2.index]
            for p in POLY_NAMES:
                s = trp[trp.poly == p]
                if len(s) > 0 and (s["I"] * s["t"]).pow(2).sum() > 0:
                    dose = (s["I"] * s["t"]).values
                    kp[p] = float(max(1e-6, -(dose * np.log(np.clip(s["ret"].values, 0.01, 1.0))).sum() / (dose ** 2).sum()))
                else:
                    kp[p] = float(POLY_PARAMS[p]["k_rel"] * K0)
            tep = df.loc[Xte2.index]
            pk = np.exp(-np.array([kp[p] for p in tep["poly"]]) * (tep["I"] * tep["t"]).values)

            # TOPSIS matrix & stability
            rep = {"Low": 0.90, "Moderate": 0.65, "Severe": 0.30}
            m3_decisions, m3_scores = {}, {}
            for p in POLY_NAMES:
                for s, r_ in rep.items():
                    res = topsis_score(p, r_)
                    m3_decisions[f"{p}-{s}"] = res["best_route"]
                    m3_scores[f"{p}-{s}"] = res["scores"]

            rng = np.random.default_rng(7)
            tot, same = 0, 0
            for key, base_route in m3_decisions.items():
                p, s = key.split("-")
                for _ in range(150):
                    w = np.array(DEFAULT_WEIGHTS) * rng.uniform(0.8, 1.2, 5)
                    w /= w.sum()
                    tot += 1
                    if topsis_score(p, rep[s], w)["best_route"] == base_route:
                        same += 1

            self.df = df
            self.rf = rf
            self.lr_base = lr
            self.gb = gb
            self.kp = kp
            self.x1_cols = list(X1.columns)
            self.x2_cols = list(X2.columns)
            self.dataset_meta = {
                "source": source_label,
                "kaggle_slug": kaggle_slug,
                "rows": int(len(df)),
                "loaded_at": time.time(),
                "class_counts": [int((df["stage"] == i).sum()) for i in range(3)],
                "polymer_counts": {p: int((df["poly"] == p).sum()) for p in POLY_NAMES},
            }
            self.metrics = {
                "n": int(len(df)),
                "class_counts": self.dataset_meta["class_counts"],
                "polymer_counts": self.dataset_meta["polymer_counts"],
                "m1": {
                    "acc": float(accuracy_score(yte, p1)),
                    "cv": cv_mean,
                    "cv_sd": cv_sd,
                    "macro_f1": float(f1_score(yte, p1, average="macro", zero_division=0)),
                    "base_acc": base_acc,
                    "per_class": {
                        STAGES[i]: {
                            "p": float(pr[i]),
                            "r": float(rc[i]),
                            "f1": float(f1[i]),
                            "n": int(sup[i]),
                        }
                        for i in range(3)
                    },
                    "cm": confusion_matrix(yte, p1, labels=[0, 1, 2]).tolist(),
                    "imp": {col: round(float(val), 4) for col, val in zip(X1.columns, rf.feature_importances_)},
                },
                "m2": {
                    "r2_ml": float(r2_score(yte2, pg)),
                    "rmse_ml": float(mean_squared_error(yte2, pg) ** 0.5),
                    "mae_ml": float(mean_absolute_error(yte2, pg)),
                    "r2_kin": float(r2_score(yte2, pk)),
                    "rmse_kin": float(mean_squared_error(yte2, pk) ** 0.5),
                    "mae_kin": float(mean_absolute_error(yte2, pk)),
                    "kp": kp,
                },
                "m3_decisions": m3_decisions,
                "m3_scores": m3_scores,
                "m3_stability": float(same / max(1, tot)),
                "m3_cases": len(m3_decisions),
                "dataset_meta": self.dataset_meta,
            }
            if save_cache:
                joblib.dump(
                    {
                        "df": self.df,
                        "rf": self.rf,
                        "lr_base": self.lr_base,
                        "gb": self.gb,
                        "kp": self.kp,
                        "x1_cols": self.x1_cols,
                        "x2_cols": self.x2_cols,
                        "metrics": self.metrics,
                        "dataset_meta": self.dataset_meta,
                    },
                    CACHE_DIR / "polylife_models.joblib",
                )
            return self.metrics

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run end-to-end real-time inference across Modules 1, 2, and 3."""
        with self._lock:
            poly = str(payload.get("poly", "PP")).upper()
            if poly not in POLY_PARAMS:
                poly = "PP"
            t = float(payload.get("t", 1500.0))
            I = float(payload.get("I", 1.0))
            T = float(payload.get("T", 50.0))
            RH = float(payload.get("RH", 60.0))
            spectral_mode = str(payload.get("spectral_mode", "auto"))
            weights = payload.get("weights", DEFAULT_WEIGHTS)

            kin = compute_kinetics(poly, t, I, T, RH)
            if spectral_mode == "manual":
                ci = float(payload.get("CI", kin["CI"]))
                hi = float(payload.get("HI", kin["HI"]))
                dxc = float(payload.get("dXc", kin["dXc"]))
            else:
                ci, hi, dxc = kin["CI"], kin["HI"], kin["dXc"]

            # Module 1: RF Stage Classification
            x1_row = {n: float(n == poly) for n in POLY_NAMES}
            x1_row.update({"CI": ci, "HI": hi, "dXc": dxc})
            x1_df = pd.DataFrame([x1_row])[self.x1_cols]
            stage_idx = int(self.rf.predict(x1_df)[0])
            probs = self.rf.predict_proba(x1_df)[0]
            classes = list(self.rf.classes_)
            stage_probs = {STAGES[i]: 0.0 for i in range(3)}
            for cls_idx, prob in zip(classes, probs):
                stage_probs[STAGES[int(cls_idx)]] = round(float(prob), 4)

            # Module 2: GB Mw Retention Prediction
            x2_row = {n: float(n == poly) for n in POLY_NAMES}
            x2_row.update({"t": t, "I": I, "T": T, "RH": RH})
            x2_df = pd.DataFrame([x2_row])[self.x2_cols]
            ret_ml = float(np.clip(self.gb.predict(x2_df)[0], 0.02, 1.0))
            ret_kin_dose = float(np.clip(np.exp(-self.kp[poly] * I * t), 0.02, 1.0))

            # Curve 0..2500h for live chart
            tg = np.arange(0, 2501, 125)
            curve_true = [float(np.clip(np.exp(-kin["k"] * ti), 0.02, 1.0)) for ti in tg]
            curve_kin = [float(np.clip(np.exp(-self.kp[poly] * I * ti), 0.02, 1.0)) for ti in tg]
            grid = pd.DataFrame(
                [{**{n: float(n == poly) for n in POLY_NAMES}, "t": float(ti), "I": I, "T": T, "RH": RH} for ti in tg]
            )[self.x2_cols]
            curve_ml = [float(np.clip(v, 0.02, 1.0)) for v in self.gb.predict(grid)]

            # Estimate t50
            tfine = np.arange(0, 3501, 25)
            gfine = pd.DataFrame(
                [{**{n: float(n == poly) for n in POLY_NAMES}, "t": float(ti), "I": I, "T": T, "RH": RH} for ti in tfine]
            )[self.x2_cols]
            ml_fine = self.gb.predict(gfine)
            below = np.where(ml_fine < 0.5)[0]
            t50_ml = float(tfine[below[0]]) if len(below) > 0 else None
            t50_true = float(np.log(2) / kin["k"]) if kin["k"] > 0 else None
            t50_kin = float(np.log(2) / (self.kp[poly] * max(I, 0.05)))

            # Module 3: TOPSIS Recycling Decision Support
            topsis = topsis_score(poly, ret_ml, weights=weights)

            return {
                "input": {
                    "poly": poly,
                    "poly_name": POLY_PARAMS[poly]["name"],
                    "resin_code": POLY_PARAMS[poly]["code"],
                    "color": POLY_PARAMS[poly]["color"],
                    "t": t,
                    "I": I,
                    "T": T,
                    "RH": RH,
                    "spectral_mode": spectral_mode,
                },
                "spectral": {
                    "CI": round(ci, 4),
                    "HI": round(hi, 4),
                    "dXc": round(dxc, 4),
                    "CI_max": POLY_PARAMS[poly]["ci_max"],
                    "HI_max": POLY_PARAMS[poly]["hi_max"],
                    "dXc_max": POLY_PARAMS[poly]["dxc_max"],
                },
                "module1": {
                    "stage": STAGES[stage_idx],
                    "stage_index": stage_idx,
                    "probabilities": stage_probs,
                },
                "module2": {
                    "ret_ml": round(ret_ml, 4),
                    "ret_true_kinetics": round(kin["ret"], 4),
                    "ret_dose_baseline": round(ret_kin_dose, 4),
                    "t50_ml": round(t50_ml, 1) if t50_ml is not None else None,
                    "t50_true": round(t50_true, 1) if t50_true is not None else None,
                    "t50_kin": round(t50_kin, 1),
                    "curve": {
                        "t": tg.tolist(),
                        "true": [round(v, 4) for v in curve_true],
                        "ml": [round(v, 4) for v in curve_ml],
                        "kin": [round(v, 4) for v in curve_kin],
                    },
                },
                "module3": topsis,
            }

    # ---------- Persistence (SQLite Real-Time Mirror) ----------
    def save_assessment(self, record: dict[str, Any]) -> dict[str, Any]:
        rec_id = str(record.get("id") or f"PL-{uuid.uuid4().hex[:8].upper()}")
        now = float(record.get("created_at") or time.time())
        row = {
            "id": rec_id,
            "sample_name": str(record.get("sample_name") or f"{record.get('poly', 'PP')} Specimen"),
            "batch_id": str(record.get("batch_id") or f"BATCH-{time.strftime('%H%M')}"),
            "poly": str(record.get("poly", "PP")),
            "t": float(record.get("t", 0)),
            "I": float(record.get("I", 1.0)),
            "T": float(record.get("T", 50.0)),
            "RH": float(record.get("RH", 60.0)),
            "CI": float(record.get("CI", 0.0)),
            "HI": float(record.get("HI", 0.0)),
            "dXc": float(record.get("dXc", 0.0)),
            "ret": float(record.get("ret", 1.0)),
            "stage": str(record.get("stage", "Low")),
            "stage_probs": record.get("stage_probs", {}),
            "route": str(record.get("route", "Mechanical recycling")),
            "topsis_scores": record.get("topsis_scores", []),
            "source": str(record.get("source", "Lab Console")),
            "created_at": now,
        }
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO assessments
                (id, sample_name, batch_id, poly, exposure_t, uv_I, temp_C, rel_humidity, CI, HI, dXc, ret, stage, stage_probs, route, topsis_scores, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["sample_name"],
                    row["batch_id"],
                    row["poly"],
                    row["t"],
                    row["I"],
                    row["T"],
                    row["RH"],
                    row["CI"],
                    row["HI"],
                    row["dXc"],
                    row["ret"],
                    row["stage"],
                    json.dumps(row["stage_probs"]),
                    row["route"],
                    json.dumps(row["topsis_scores"]),
                    row["source"],
                    row["created_at"],
                ),
            )
            conn.commit()
        return row

    def list_assessments(self, limit: int = 80) -> list[dict[str, Any]]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, sample_name, batch_id, poly,
                       exposure_t AS t, uv_I AS I, temp_C AS T, rel_humidity AS RH,
                       CI, HI, dXc, ret, stage, stage_probs, route, topsis_scores, source, created_at
                FROM assessments ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["stage_probs"] = json.loads(d["stage_probs"] or "{}")
            d["topsis_scores"] = json.loads(d["topsis_scores"] or "[]")
            out.append(d)
        return out

    def delete_assessment(self, rec_id: str) -> bool:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute("DELETE FROM assessments WHERE id = ?", (rec_id,))
            conn.commit()
            return cur.rowcount > 0

    # ---------- Kaggle Dataset Import & Mapping ----------
    def _read_kaggle_credentials(self, username: str | None, key: str | None) -> tuple[str, str] | None:
        if username and key:
            return username.strip(), key.strip()
        kj = Path.home() / ".kaggle" / "kaggle.json"
        if kj.exists():
            try:
                creds = json.loads(kj.read_text())
                if creds.get("username") and creds.get("key"):
                    return creds["username"], creds["key"]
            except Exception:
                pass
        env_u = os.environ.get("KAGGLE_USERNAME")
        env_k = os.environ.get("KAGGLE_KEY")
        if env_u and env_k:
            return env_u, env_k
        return None

    def fetch_kaggle_dataset(
        self,
        slug: str,
        username: str | None = None,
        key: str | None = None,
    ) -> dict[str, Any]:
        """Download a Kaggle dataset ZIP via Kaggle REST API v1 and profile its CSV files."""
        clean_slug = slug.strip().replace("https://www.kaggle.com/datasets/", "").strip("/")
        if "/" not in clean_slug:
            raise ValueError("Provide a Kaggle dataset slug in 'owner/dataset-name' format.")
        creds = self._read_kaggle_credentials(username, key)
        if not creds:
            raise PermissionError(
                "Kaggle API credentials not found. Enter your Kaggle Username & API Key (from kaggle.com/settings) or upload a downloaded Kaggle CSV directly."
            )
        url = f"https://www.kaggle.com/api/v1/datasets/download/{clean_slug}"
        token = base64.b64encode(f"{creds[0]}:{creds[1]}".encode()).decode()
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {token}", "User-Agent": "PolyLifeAI/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Kaggle API error HTTP {e.code}: {e.reason}. Check dataset slug and API key.") from e
        except Exception as e:
            raise RuntimeError(f"Failed to download from Kaggle: {e}") from e

        dest_dir = DATA_DIR / clean_slug.replace("/", "__")
        dest_dir.mkdir(parents=True, exist_ok=True)
        csv_files: list[Path] = []
        if raw_bytes[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".csv") and not member.startswith("__MACOSX"):
                        target = dest_dir / Path(member).name
                        target.write_bytes(zf.read(member))
                        csv_files.append(target)
        else:
            target = dest_dir / "dataset.csv"
            target.write_bytes(raw_bytes)
            csv_files.append(target)

        if not csv_files:
            raise ValueError("Downloaded Kaggle archive did not contain any CSV files.")

        primary_csv = max(csv_files, key=lambda p: p.stat().st_size)
        return self.register_csv_file(primary_csv, title=clean_slug, source=f"Kaggle ({clean_slug})")

    def register_csv_text(self, csv_text: str, filename: str = "kaggle_upload.csv") -> dict[str, Any]:
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        if not safe_name.endswith(".csv"):
            safe_name += ".csv"
        target = DATA_DIR / safe_name
        target.write_text(csv_text, encoding="utf-8")
        return self.register_csv_file(target, title=safe_name, source="Kaggle CSV Upload")

    def register_csv_file(self, path: Path, title: str, source: str) -> dict[str, Any]:
        df = pd.read_csv(path)
        ds_id = f"DS-{uuid.uuid4().hex[:8].upper()}"
        self.imported_datasets[ds_id] = df
        suggested = self._suggest_mapping(df.columns.tolist())
        preview = df.head(12).fillna("").to_dict(orient="records")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO dataset_logs (id, title, source, rows, columns, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ds_id, title, source, int(len(df)), json.dumps(df.columns.tolist()), "profiled", time.time()),
            )
            conn.commit()
        return {
            "dataset_id": ds_id,
            "title": title,
            "source": source,
            "rows": int(len(df)),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
            "suggested_mapping": suggested,
            "preview": preview,
        }

    def _suggest_mapping(self, cols: list[str]) -> dict[str, str]:
        import re
        lower = {c.lower(): c for c in cols}
        tokens = {c: set(re.split(r"[_\-\s./]+", c.lower())) for c in cols}
        patterns = {
            "poly": ["poly", "polymer", "polymer_type", "resin", "resin_type", "material", "plastic_type"],
            "t": ["t", "time", "hours", "exposure_time", "aging_hours", "uv_hours", "duration"],
            "I": ["i", "irradiance", "uv_intensity", "uv_irradiance", "uv"],
            "T": ["temp", "temperature", "temp_c", "deg_c"],
            "RH": ["rh", "humidity", "relative_humidity"],
            "CI": ["ci", "carbonyl", "carbonyl_index"],
            "HI": ["hi", "hydroxyl", "hydroxyl_index"],
            "dXc": ["dxc", "crystallinity", "crystallinity_gain"],
            "ret": ["ret", "retention", "mw_retention", "tensile_retention", "recycling_rate", "recovery_rate"],
        }
        mapping: dict[str, str] = {}
        used_cols: set[str] = set()
        for target, keys in patterns.items():
            for k in keys:
                matched = None
                if k in lower and lower[k] not in used_cols:
                    matched = lower[k]
                elif len(k) > 2:
                    for orig, toks in tokens.items():
                        if orig not in used_cols and (k in toks or k in orig.lower()):
                            matched = orig
                            break
                if matched:
                    mapping[target] = matched
                    used_cols.add(matched)
                    break
        return mapping

    def retrain_from_imported_dataset(
        self,
        dataset_id: str,
        mapping: dict[str, str],
        title: str | None = None,
    ) -> dict[str, Any]:
        """Adapt an imported Kaggle CSV into PolyLife features and retrain ML models."""
        if dataset_id not in self.imported_datasets:
            raise KeyError("Dataset ID not found in session. Upload or fetch the Kaggle CSV again.")
        raw = self.imported_datasets[dataset_id].copy()
        if len(raw) < 20:
            raise ValueError("Dataset needs at least 20 rows for train/test validation.")

        rng = np.random.default_rng(42)
        n = len(raw)
        mapped = pd.DataFrame(index=raw.index)

        # Polymer column normalization
        if mapping.get("poly") and mapping["poly"] in raw.columns:
            p_raw = raw[mapping["poly"]].astype(str).str.upper()
            def norm_poly(val: str) -> str:
                for p in POLY_NAMES:
                    if p in val:
                        return p
                return POLY_NAMES[abs(hash(val)) % len(POLY_NAMES)]
            mapped["poly"] = p_raw.apply(norm_poly)
        else:
            mapped["poly"] = rng.choice(POLY_NAMES, n)

        def col_num(key: str, low: float, high: float, default_arr: np.ndarray | None = None) -> np.ndarray:
            col_name = mapping.get(key)
            if col_name and col_name in raw.columns:
                vals = pd.to_numeric(raw[col_name], errors="coerce")
                if vals.notna().sum() >= 5:
                    vals = vals.fillna(vals.median())
                    return vals.to_numpy(dtype=float)
            if default_arr is not None:
                return default_arr
            return rng.uniform(low, high, n)

        t = np.clip(col_num("t", 0, 2000), 0, 5000)
        I = np.clip(col_num("I", 0.6, 1.4), 0.1, 3.0)
        T = np.clip(col_num("T", 30, 70), 5, 120)
        RH = np.clip(col_num("RH", 20, 90), 5, 100)

        k = (
            np.array([POLY_PARAMS[p]["k_rel"] for p in mapped["poly"]])
            * K0
            * I
            * np.exp(-EA / R * (1.0 / (T + 273.15) - 1.0 / 303.15))
            * (1.0 + 0.004 * (RH - 50.0))
        )
        kt = k * t
        frac = 1.0 - np.exp(-1.4 * kt)

        if mapping.get("ret") and mapping["ret"] in raw.columns:
            r_vals = pd.to_numeric(raw[mapping["ret"]], errors="coerce")
            r_vals = r_vals.fillna(r_vals.median())
            # Auto-scale percentage (0..100) to (0..1)
            if r_vals.max() > 1.5:
                r_vals = r_vals / 100.0
            ret = np.clip(r_vals.to_numpy(dtype=float), 0.02, 1.0)
        else:
            ret = np.clip(np.exp(-kt) * (1.0 + rng.normal(0, 0.03, n)), 0.02, 1.0)

        ci_def = np.clip(np.array([POLY_PARAMS[p]["ci_max"] for p in mapped["poly"]]) * frac + rng.normal(0, 0.1, n), 0.0, None)
        hi_def = np.clip(np.array([POLY_PARAMS[p]["hi_max"] for p in mapped["poly"]]) * frac + rng.normal(0, 0.07, n), 0.0, None)
        dxc_def = np.array([POLY_PARAMS[p]["dxc_max"] for p in mapped["poly"]]) * frac + rng.normal(0, 0.6, n)

        mapped["t"] = t
        mapped["I"] = I
        mapped["T"] = T
        mapped["RH"] = RH
        mapped["CI"] = np.clip(col_num("CI", 0, 3.5, ci_def), 0, None)
        mapped["HI"] = np.clip(col_num("HI", 0, 1.5, hi_def), 0, None)
        mapped["dXc"] = col_num("dXc", 0, 8.0, dxc_def)
        mapped["ret"] = ret
        mapped["stage"] = np.where(ret >= 0.80, 0, np.where(ret >= 0.50, 1, 2))

        label = title or f"Kaggle Dataset ({dataset_id})"
        metrics = self.train_on_dataframe(mapped, source_label=label, kaggle_slug=title, save_cache=False)
        return metrics

    def list_dataset_logs(self) -> list[dict[str, Any]]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM dataset_logs ORDER BY created_at DESC LIMIT 20").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["columns"] = json.loads(d["columns"] or "[]")
            out.append(d)
        return out
