import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

# --- 1. Load dataset ---
df = pd.read_csv("data/heart_cleveland_upload.csv")

# --- 2. Define target ---
target_col = "condition"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# --- 3. Ensure numeric types ---
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# --- 4. Split dataset ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# --- 5. Preprocessing pipeline ---
num_cols = X.columns.tolist()
num_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
preprocess = ColumnTransformer(transformers=[("num", num_pipe, num_cols)])

# --- 6. XGBoost model ---
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
model = XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    eval_metric="logloss",
)

pipe = Pipeline(steps=[("preprocess", preprocess), ("clf", model)])
pipe.fit(X_train, y_train)


# --- 7. Evaluate helper ---
def eval_split(name, X_, y_):
    proba = pipe.predict_proba(X_)[:, 1]
    roc = roc_auc_score(y_, proba)
    pr = average_precision_score(y_, proba)
    return {"split": name, "roc_auc": roc, "pr_auc": pr}


scores = [eval_split("valid", X_valid, y_valid), eval_split("test", X_test, y_test)]
print("\nValidation/Test AUC:")
for s in scores:
    print(f"{s['split']:>6}: ROC-AUC={s['roc_auc']:.3f}, PR-AUC={s['pr_auc']:.3f}")

# --- 8. Select thresholds using Precision / Specificity ---
proba_valid = pipe.predict_proba(X_valid)[:, 1]

# 8.1 chọn t2 theo Precision mục tiêu
target_precision = 0.80
prec, rec, thr_pr = precision_recall_curve(y_valid, proba_valid)
idx_prec = np.where(prec[:-1] >= target_precision)[0]
if len(idx_prec) > 0:
    t2 = float(thr_pr[idx_prec[0]])
else:
    t2 = float(thr_pr[np.argmax(prec[:-1])])

# 8.2 chọn t1 theo Specificity mục tiêu (1 - FPR)
target_specificity = 0.90
fpr, tpr, thr_roc = roc_curve(y_valid, proba_valid)
spec = 1 - fpr
idx_spec = np.where(spec >= target_specificity)[0]
if len(idx_spec) > 0:
    t1 = float(thr_roc[idx_spec[-1]])
else:
    t1 = 0.3

# 8.3 đảm bảo t1 < t2
if t1 >= t2:
    t1 = min(t1, 0.4)
    t2 = max(t2, 0.6)

thresholds = {"t1": round(t1, 3), "t2": round(t2, 3)}
print(f"\nChosen thresholds by policy: t1={thresholds['t1']}, t2={thresholds['t2']}")

# --- 9. Evaluate on test ---
proba_test = pipe.predict_proba(X_test)[:, 1]
y_pred = (proba_test >= t2).astype(int)

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec_test = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, proba_test)
cm = confusion_matrix(y_test, y_pred)

print(
    f"Accuracy={acc:.3f}, Recall={rec:.3f}, Precision={prec_test:.3f}, F1={f1:.3f}, ROC-AUC={roc:.3f}"
)
print("Confusion Matrix:\n", cm)

# --- 10. Save model and metadata ---
joblib.dump(pipe, ART_DIR / "heart_xgb_model.pkl")
json.dump(
    thresholds,
    open(ART_DIR / "heart_xgb_thresholds.json", "w"),
    indent=2,
)
json.dump(
    {"features": num_cols, "target": target_col},
    open(ART_DIR / "heart_xgb_schema.json", "w"),
    indent=2,
)

# --- 11. Visualization ---
fpr, tpr, _ = roc_curve(y_test, proba_test)
precision_test, recall_test, _ = precision_recall_curve(y_test, proba_test)

plt.figure(figsize=(12, 5))

# ROC curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# Precision-Recall curve
plt.subplot(1, 2, 2)
plt.plot(recall_test, precision_test, color="green", label="Precision-Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.show()
