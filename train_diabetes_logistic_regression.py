import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# --- PATHS ---
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

# --- 1. Load dataset ---
df = pd.read_csv("data/diabetes.csv")

# --- 2. Select all columns ---
target_col = "Outcome"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# --- 3. Replace impossible 0s with NaN ---
zero_is_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for c in zero_is_missing:
    X.loc[X[c] == 0, c] = np.nan

# --- 4. Split data ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# --- 5. Preprocess ---
num_cols = X.columns.tolist()
num_pipe = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
preprocess = ColumnTransformer([("num", num_pipe, num_cols)])

# 6. Model
model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
pipe = Pipeline([("preprocess", preprocess), ("clf", model)])
pipe.fit(X_train, y_train)


# --- 7. Evaluate ---
def eval_split(name, X_, y_):
    proba = pipe.predict_proba(X_)[:, 1]
    return {
        "split": name,
        "roc_auc": roc_auc_score(y_, proba),
        "pr_auc": average_precision_score(y_, proba),
    }


scores = [eval_split("valid", X_valid, y_valid), eval_split("test", X_test, y_test)]
print("Validation/Test AUC:")
for s in scores:
    print(f"{s['split']:>6}: ROC-AUC={s['roc_auc']:.3f}, PR-AUC={s['pr_auc']:.3f}")

# --- 8. Optimize threshold ---
proba_valid = pipe.predict_proba(X_valid)[:, 1]
prec, rec, thresh = precision_recall_curve(y_valid, proba_valid)
best_f1, best_t = 0, 0
for p, r, t in zip(prec, rec, np.append(thresh, 1.0)):
    if (p + r) > 0:
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1, best_t = t, f1
print(f"Best threshold (F1): t*={best_t:.3f}, F1*={best_f1:.3f}")

t1 = max(0.3, min(best_t, 0.5))
t2 = max(0.6, best_t)
thresholds = {"t1": float(round(t1, 3)), "t2": float(round(t2, 3))}


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
print("Confusion Matrix: \n", cm)


joblib.dump(pipe, ART_DIR / "diabetes_model.pkl")
json.dump(thresholds, open(ART_DIR / "diabetes_thresholds.json", "w"), indent=2)
json.dump(
    {"features": num_cols, "target": target_col},
    open(ART_DIR / "diabetes_schema.json", "w"),
    indent=2,
)
