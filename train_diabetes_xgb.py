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

# --- Paths ---
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

# --- 1. Load dataset ---
df = pd.read_csv("./data/diabetes.csv")

# --- 2. Define target ---
target_col = "Outcome"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# --- 3. Handle invalid 0-values ---
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[zero_cols] = X[zero_cols].replace(0, np.nan)

# --- 4. Split dataset ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# --- 5. Preprocessing ---
num_cols = X.columns.tolist()
num_pipe = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
preprocess = ColumnTransformer([("num", num_pipe, num_cols)])

# 6. XGBoost model (tuned)
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()  # balance positive class
model = XGBClassifier(
    n_estimators=600,
    max_depth=3,
    learning_rate=0.005,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    eval_metric="logloss",
)

pipe = Pipeline([("preprocess", preprocess), ("clf", model)])

pipe.fit(X_train, y_train)


# --- 7. Evaluation helper ---
def eval_split(name, X_, y_):
    proba = pipe.predict_proba(X_)[:, 1]
    roc = roc_auc_score(y_, proba)
    pr = average_precision_score(y_, proba)
    return {"split": name, "roc_auc": roc, "pr_auc": pr}


scores = [eval_split("valid", X_valid, y_valid), eval_split("test", X_test, y_test)]
print("Validation/Test AUC:")
for s in scores:
    print(f"{s['split']:>6}: ROC-AUC={s['roc_auc']:.3f}, PR-AUC={s['pr_auc']:.3f}")

# --- 8. Optimize threshold for best F1 ---
proba_valid = pipe.predict_proba(X_valid)[:, 1]
prec, rec, thresh = precision_recall_curve(y_valid, proba_valid)
best_f1, best_t = 0.0, 0.5
for p, r, t in zip(prec, rec, np.append(thresh, 1.0)):
    if (p + r) > 0:
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1, best_t = f1, t

print(f"Best threshold (F1): t*={best_t:.3f}, F1*={best_f1:.3f}")

# --- 9. Evaluate on test set ---
proba_test = pipe.predict_proba(X_test)[:, 1]
y_pred = (proba_test >= best_t).astype(int)

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, proba_test)
cm = confusion_matrix(y_test, y_pred)

print("Test Performance:")
print(
    f"Accuracy={acc:.3f}, Recall={rec:.3f}, Precision={prec:.3f}, F1={f1:.3f}, ROC-AUC={roc:.3f}"
)
print("Confusion Matrix:\n", cm)

# --- 10. Save model and metadata ---
joblib.dump(pipe, ART_DIR / "diabetes_xgb_model.pkl")
json.dump(
    {"t1": float(round(best_t * 0.8, 3)), "t2": float(round(best_t, 3))},
    open(ART_DIR / "diabetes_xgb_thresholds.json", "w"),
    indent=2,
)
json.dump(
    {"features": num_cols, "target": target_col},
    open(ART_DIR / "diabetes_xgb_schema.json", "w"),
    indent=2,
)

# --- 11. Visualization: ROC & Precision-Recall ---
# ROC
fpr, tpr, _ = roc_curve(y_test, proba_test)

# Precision–Recall (on test set)
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

# Precision–Recall curve
plt.subplot(1, 2, 2)
plt.plot(recall_test, precision_test, color="green", label="Precision-Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.show()
