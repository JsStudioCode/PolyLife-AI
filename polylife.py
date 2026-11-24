import numpy as np, pandas as pd, json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, r2_score, mean_squared_error, mean_absolute_error)

rng = np.random.default_rng(42)
R = 8.314

# ---------- 1. Chemistry-informed dataset (UV/thermal photo-oxidation, simulated) ----------
# relative photo-oxidation susceptibility, max carbonyl index, max hydroxyl index, crystallinity gain
POLY = {          #  k_rel   CI_max HI_max dXc
    "HDPE": (0.80, 2.2, 0.9, 6.0),
    "LDPE": (1.00, 2.6, 1.1, 5.0),
    "PP":   (1.60, 3.2, 1.3, 7.0),
    "PS":   (1.30, 1.8, 0.8, 0.0),
    "PET":  (0.50, 1.2, 0.6, 8.0),
    "PVC":  (0.70, 1.5, 0.7, 2.0),
}
names = list(POLY)
N = 1800
poly = rng.choice(names, N)
t = rng.uniform(0, 2000, N)                  # exposure time, h
I = rng.uniform(0.6, 1.4, N)                 # UV irradiance factor (relative)
T = rng.uniform(30, 70, N)                   # temperature, deg C
RH = rng.uniform(20, 90, N)                  # relative humidity, %
Ea = 30e3                                    # J/mol, apparent activation energy
k0 = 2.0e-4                                  # 1/h base rate constant
k = np.array([POLY[p][0] for p in poly]) * k0 * I * np.exp(-Ea / R * (1 / (T + 273.15) - 1 / 303.15)) * (1 + 0.004 * (RH - 50))
kt = k * t
ret = np.exp(-kt) * (1 + rng.normal(0, 0.03, N))          # Mw retention (GPC), first order + noise
ret = np.clip(ret, 0.02, 1.0)
frac = 1 - np.exp(-1.4 * kt)
CI = np.array([POLY[p][1] for p in poly]) * frac + rng.normal(0, 0.12, N)
HI = np.array([POLY[p][2] for p in poly]) * frac + rng.normal(0, 0.08, N)
dXc = np.array([POLY[p][3] for p in poly]) * frac + rng.normal(0, 0.8, N)
CI, HI = np.clip(CI, 0, None), np.clip(HI, 0, None)
stage = np.where(ret >= 0.80, 0, np.where(ret >= 0.50, 1, 2))   # 0 Low, 1 Moderate, 2 Severe
df = pd.DataFrame(dict(poly=poly, t=t, I=I, T=T, RH=RH, CI=CI, HI=HI, dXc=dXc, ret=ret, stage=stage))
STG = ["Low", "Moderate", "Severe"]
print("class balance:", np.bincount(stage))

oh = pd.get_dummies(df["poly"]).reindex(columns=names, fill_value=0).astype(float)
out = {"n": N, "class_counts": np.bincount(stage).tolist()}

# ---------- MODULE 1: degradation-stage classification from spectral/physical features ----------
X1 = pd.concat([oh, df[["CI", "HI", "dXc"]]], axis=1)
y1 = df["stage"]
Xtr, Xte, ytr, yte = train_test_split(X1, y1, test_size=0.25, random_state=1, stratify=y1)
rf = RandomForestClassifier(n_estimators=300, random_state=1).fit(Xtr, ytr)
p1 = rf.predict(Xte)
cv = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=1), X1, y1,
                     cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="accuracy")
# baseline: logistic regression on carbonyl index alone
lr = LogisticRegression(max_iter=2000).fit(Xtr[["CI"]], ytr)
base_acc = accuracy_score(yte, lr.predict(Xte[["CI"]]))
pr, rc, f1, sup = precision_recall_fscore_support(yte, p1, labels=[0, 1, 2], zero_division=0)
out["m1"] = dict(acc=accuracy_score(yte, p1), cv=float(cv.mean()), cv_sd=float(cv.std()),
                 macro_f1=f1_score(yte, p1, average="macro"), base_acc=base_acc,
                 per_class={STG[i]: dict(p=pr[i], r=rc[i], f1=f1[i], n=int(sup[i])) for i in range(3)},
                 weighted=dict(zip(["p", "r", "f1"], precision_recall_fscore_support(yte, p1, average="weighted")[:3])),
                 macro=dict(zip(["p", "r", "f1"], precision_recall_fscore_support(yte, p1, average="macro")[:3])),
                 cm=confusion_matrix(yte, p1).tolist(),
                 imp=dict(zip(X1.columns, rf.feature_importances_.round(3).tolist())))

# ---------- MODULE 2: Mw-retention prediction (ML vs first-order dose-only kinetics) ----------
X2 = pd.concat([oh, df[["t", "I", "T", "RH"]]], axis=1)
y2 = df["ret"]
Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.25, random_state=2)
gb = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=2).fit(Xtr2, ytr2)
pg = gb.predict(Xte2)
# baseline: per-polymer first-order fit  ln(R) = -k_p * (I*t)   (ignores T and RH)
kp = {}
trp = df.loc[Xtr2.index]
for p in names:
    s = trp[trp.poly == p]
    dose = (s.I * s.t).values
    kp[p] = float(-(dose * np.log(s.ret.values)).sum() / (dose ** 2).sum())
tep = df.loc[Xte2.index]
pk = np.exp(-np.array([kp[p] for p in tep.poly]) * (tep.I * tep.t).values)
out["m2"] = dict(r2_ml=r2_score(yte2, pg), rmse_ml=mean_squared_error(yte2, pg) ** 0.5, mae_ml=mean_absolute_error(yte2, pg),
                 r2_kin=r2_score(yte2, pk), rmse_kin=mean_squared_error(yte2, pk) ** 0.5, mae_kin=mean_absolute_error(yte2, pk),
                 kp={a: b for a, b in kp.items()})

# chart data: PP at I=1.0, T=50C, RH=60%, t = 0..2000 h  (true noise-free kinetics vs models)
tg = np.arange(0, 2001, 125)
kk = POLY["PP"][0] * k0 * 1.0 * np.exp(-Ea / R * (1 / 323.15 - 1 / 303.15)) * (1 + 0.004 * 10)
true = np.exp(-kk * tg)
grid = pd.DataFrame(dict(t=tg, I=1.0, T=50.0, RH=60.0))
grid = pd.concat([pd.DataFrame({n: (1.0 if n == "PP" else 0.0) for n in names}, index=grid.index), grid], axis=1)[X2.columns]
ml_curve = gb.predict(grid)
kin_curve = np.exp(-kp["PP"] * 1.0 * tg)
out["chart"] = dict(t=tg.tolist(), true=true.tolist(), ml=ml_curve.tolist(), kin=kin_curve.tolist())
# time to reach 50% Mw retention (severe threshold) for the case above
out["t50_true"] = float(np.log(2) / kk)
tfine = np.arange(0, 3001, 5)
gf = pd.concat([pd.DataFrame({n: (1.0 if n == "PP" else 0.0) for n in names}, index=range(len(tfine))),
                pd.DataFrame(dict(t=tfine, I=1.0, T=50.0, RH=60.0))], axis=1)[X2.columns]
mlf = gb.predict(gf)
out["t50_ml"] = float(tfine[np.argmax(mlf < 0.5)])
out["t50_kin"] = float(np.log(2) / kp["PP"])

# ---------- MODULE 3: TOPSIS recycling decision support ----------
ROUTES = ["Mechanical recycling", "Solvent purification", "Chemical recycling", "Energy recovery"]
# base attributes: yield (0-1), cost (INR/kg), CO2e (kg/kg), energy (MJ/kg), quality (0-1)
def attrs(pol, ret_):
    q = min(1.0, max(0.05, ret_))
    mech_y = 0.92 * q ** 0.8
    mech_q = q ** (2.4 if pol == 'PET' else 1.5)
    chem_ok = {"PET": 1.0, "PS": 0.95, "HDPE": 0.85, "LDPE": 0.85, "PP": 0.85, "PVC": 0.60}[pol]
    sol_ok = {"PS": 1.0, "PP": 0.85, "HDPE": 0.80, "LDPE": 0.80, "PET": 0.55, "PVC": 0.75}[pol]
    return {
        "Mechanical recycling": [mech_y, 18, 0.45, 6, mech_q],
        "Solvent purification": [0.85 * sol_ok * q ** 0.35, 38, 1.10, 14, 0.90 * sol_ok * q ** 0.5],
        "Chemical recycling":   [0.75 * chem_ok, 55, 1.90, 22, 0.88 * chem_ok],
        "Energy recovery":      [0.25, 8, 2.60, 3, 0.05],
    }
W = np.array([0.30, 0.15, 0.20, 0.10, 0.25])      # yield, cost, CO2e, energy, quality
BEN = np.array([1, 0, 0, 0, 1], bool)

def topsis(pol, ret_, w=W):
    A = np.array([attrs(pol, ret_)[r] for r in ROUTES], float)
    Nn = A / np.linalg.norm(A, axis=0)
    V = Nn * w
    ideal = np.where(BEN, V.max(0), V.min(0)); anti = np.where(BEN, V.min(0), V.max(0))
    dp = np.linalg.norm(V - ideal, axis=1); dm = np.linalg.norm(V - anti, axis=1)
    return dm / (dp + dm)

rep = {"Low": 0.90, "Moderate": 0.65, "Severe": 0.30}
dec, scores = {}, {}
for p in names:
    for s, r_ in rep.items():
        c = topsis(p, r_)
        dec[f"{p}-{s}"] = ROUTES[int(np.argmax(c))]
        scores[f"{p}-{s}"] = c.round(3).tolist()
out["m3_decisions"] = dec
out["m3_scores"] = scores
# sensitivity: perturb weights +-20 %, check rank-1 stability
stab = {}
tot, same = 0, 0
for key in dec:
    p, s = key.split("-"); base = int(np.argmax(topsis(p, rep[s])))
    for _ in range(300):
        w = W * rng.uniform(0.8, 1.2, 5); w /= w.sum()
        tot += 1; same += int(np.argmax(topsis(p, rep[s], w)) == base)
out["m3_stability"] = same / tot
out["m3_cases"] = len(dec)
# end-to-end demo: a weathered PP sample
demo = dict(poly="PP", t=1500, I=1.0, T=50, RH=60, CI=None)
dk = POLY["PP"][0] * k0 * np.exp(-Ea / R * (1 / 323.15 - 1 / 303.15)) * (1 + 0.004 * 10)
fr = 1 - np.exp(-1.4 * dk * 1500)
x = pd.DataFrame([{**{n: float(n == "PP") for n in names}, "CI": 3.2 * fr, "HI": 1.3 * fr, "dXc": 7.0 * fr}])[X1.columns]
st = int(rf.predict(x)[0])
xr = pd.DataFrame([{**{n: float(n == "PP") for n in names}, "t": 1500, "I": 1.0, "T": 50.0, "RH": 60.0}])[X2.columns]
rpred = float(gb.predict(xr)[0])
out["demo"] = dict(CI=float(3.2 * fr), stage=STG[st], ret=rpred,
                   route=ROUTES[int(np.argmax(topsis("PP", rpred)))], scores=topsis("PP", rpred).round(3).tolist())
json.dump(out, open("results.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))