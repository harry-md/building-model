import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

# --- 1. Load dataset ---
df = pd.read_csv("data/heart_cleveland_upload.csv")

# --- 2. Define target ---
target_col = "condition"

# --- 3. Prepare data ---
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# --- 4. Ensure numeric values ---
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# --- 5. Split ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# --- 6. Preprocessing ---
num_cols = X.columns.tolist()
num_pipe = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
preprocess = ColumnTransformer([("num", num_pipe, num_cols)])

# --- 7. Model ---
model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
pipe = Pipeline([("preprocess", preprocess), ("clf", model)])
pipe.fit(X_train, y_train)


# --- 8. Evaluate ---
def eval_split(name, X_, y_):
    proba = pipe.predict_proba(X_)[:, 1]
    roc = roc_auc_score(y_, proba)
    pr = average_precision_score(y_, proba)
    return {"split": name, "roc_auc": roc, "pr_auc": pr}


scores = [eval_split("valid", X_valid, y_valid), eval_split("test", X_test, y_test)]
print("Validation/Test AUC:")
for s in scores:
    print(f"{s['split']:>6}: ROC-AUC={s['roc_auc']:.3f}, PR-AUC={s['pr_auc']:.3f}")

# --- 9. Optimize threshold ---
proba_valid = pipe.predict_proba(X_valid)[:, 1]
prec, rec, thresh = precision_recall_curve(y_valid, proba_valid)
best_f1, best_t = 0.0, 0.5
for p, r, t in zip(prec, rec, np.append(thresh, 1.0)):
    if (p + r) > 0:
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1, best_t = f1, t

print(f"Best threshold (by F1 on valid): t* = {best_t:.3f}, F1* = {best_f1:.3f}")

t1 = max(0.30, min(best_t, 0.50))
t2 = max(0.60, best_t)
thresholds = {"t1": float(round(t1, 3)), "t2": float(round(t2, 3))}

# --- 10. Evaluate on test ---
proba_test = pipe.predict_proba(X_test)[:, 1]
y_pred = (proba_test >= thresholds["t2"]).astype(int)

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, proba_test)
cm = confusion_matrix(y_test, y_pred)

print(
    f"Accuracy={acc:.3f}, Recall={rec:.3f}, Precision={prec:.3f}, F1={f1:.3f}, ROC-AUC={roc:.3f}"
)
print("Confusion Matrix:\n", cm)

# --- 11. Export ---
joblib.dump(pipe, ART_DIR / "heart_model.pkl")
json.dump(thresholds, open(ART_DIR / "heart_thresholds.json", "w"), indent=2)
json.dump(
    {"features": num_cols, "target": target_col},
    open(ART_DIR / "heart_schema.json", "w"),
    indent=2,
)
