import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# =========================
# STEP 1: Load Dataset
# =========================
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# =========================
# STEP 2: Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# STEP 3: Scale Amount only
# =========================
scaler = StandardScaler()
X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_test["Amount"] = scaler.transform(X_test[["Amount"]])

# =========================
# STEP 4: Handle Imbalance (SMOTE)
# =========================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# =========================
# STEP 5: Train Model (XGBoost)
# =========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=10,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_resampled, y_resampled)

# =========================
# STEP 6: Get Probabilities
# =========================
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# STEP 7: AUTO THRESHOLD (F1 Optimization) 🔥
# =========================
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

best_threshold = 0.5
best_score = 0

for i in range(len(thresholds)):
    if recall[i] >= 0.90:  # ensure high recall
        score = precision[i]  # maximize precision among them
        if score > best_score:
            best_score = score
            best_threshold = thresholds[i]

print("\n✅ Best Threshold (Balanced Recall):", best_threshold)
print("Precision at this threshold:", best_score)

# =========================
# STEP 8: Final Prediction
# =========================
y_pred = (y_prob >= best_threshold).astype(int)

print("\n🔹 Final Classification Report:")
print(classification_report(y_test, y_pred))

# =========================
# STEP 9: Metrics
# =========================
roc_auc = roc_auc_score(y_test, y_prob)
print("\n🔹 ROC-AUC:", roc_auc)

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_curve, precision_curve)
print("🔹 PR-AUC:", pr_auc)

# =========================
# STEP 10: Save Model
# =========================
pickle.dump(model, open("app/model/model.pkl", "wb"))
pickle.dump(scaler, open("app/model/scaler.pkl", "wb"))

with open("app/model/threshold.txt", "w") as f:
    f.write(str(best_threshold))

print("\n✅ Model, Scaler, Threshold saved!")