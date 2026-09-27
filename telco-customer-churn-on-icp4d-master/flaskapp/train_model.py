"""
Trains a local Customer Churn model from Telco-Customer-Churn.csv and saves it
to model.pkl. Run this ONCE before starting telcochurn.py.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

CSV_PATH = "Telco-Customer-Churn.csv"
MODEL_PATH = "model.pkl"

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def main():
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=["customerID"])

    # TotalCharges sometimes arrives as blank strings; coerce to numeric.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X = df[ALL_FEATURES]
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", SimpleImputer(strategy="mean"), NUMERIC_FEATURES),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=24, max_depth=10
        )),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, proba):.3f}")

    joblib.dump({"model": model, "features": ALL_FEATURES}, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
