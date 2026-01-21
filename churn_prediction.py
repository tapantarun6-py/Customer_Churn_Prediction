# ===============================
# Customer Churn Prediction
# (Full Kaggle Dataset)
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("data/churn.csv")

print("Dataset Shape:", data.shape)

# -------------------------------
# 2. Data Cleaning
# -------------------------------
data.drop("customerID", axis=1, inplace=True)

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

# -------------------------------
# 3. Encode Categorical Variables
# -------------------------------
label_encoders = {}

for col in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# -------------------------------
# 4. Feature & Target Split
# -------------------------------
X = data.drop("Churn", axis=1)
y = data["Churn"]

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 7. Train Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 8. Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 9. Feature Importance
# -------------------------------
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance[:10], y=feature_importance.index[:10])
plt.title("Top 10 Features Affecting Churn")
plt.tight_layout()
plt.show()

# -------------------------------
# 10. Save Model
# -------------------------------
joblib.dump(model, "model/churn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel saved successfully!")
