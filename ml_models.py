from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from analytics import prepare_records, semester_summary

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

LOGISTIC_FEATURES = [
    "Attendance_Percentage",
    "Study_Hours",
    "Previous_Semester_Average",
    "Semester"
]
KMEANS_FEATURES = [
    "Attendance_Percentage",
    "Study_Hours",
    "Average_Marks"
]


def build_ml_table(raw_df, pass_mark=40.0):
    sem = semester_summary(raw_df, pass_mark)
    if sem.empty:
        return sem
    # A student's previous-semester average is known before the current
    # semester's final result. First semester uses the training median.
    return sem


def train_logistic(raw_df, pass_mark=40.0):
    data = build_ml_table(raw_df, pass_mark)
    if data.empty or data["Pass_Label"].nunique() < 2:
        raise ValueError("Training requires both Pass and Fail examples.")

    median_previous = data["Previous_Semester_Average"].median()
    data["Previous_Semester_Average"] = (
        data["Previous_Semester_Average"].fillna(median_previous)
    )

    X = data[LOGISTIC_FEATURES]
    y = data["Pass_Label"]

    if y.value_counts().min() < 2:
        raise ValueError("Not enough examples in one of the result classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_records": int(len(y_test)),
        "median_previous_average": float(median_previous)
    }

    artifact = {
        "model": model,
        "features": LOGISTIC_FEATURES,
        "median_previous_average": float(median_previous),
        "pass_mark": float(pass_mark)
    }
    joblib.dump(artifact, MODEL_DIR / "logistic_regression.joblib")
    return metrics, artifact


def train_kmeans(raw_df, clusters=3, pass_mark=40.0):
    data = build_ml_table(raw_df, pass_mark)
    if len(data) < clusters:
        raise ValueError("Not enough student-semester records for clustering.")

    X = data[KMEANS_FEATURES]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=clusters, random_state=42, n_init=10))
    ])
    labels = pipeline.fit_predict(X)

    artifact = {
        "pipeline": pipeline,
        "features": KMEANS_FEATURES,
        "pass_mark": float(pass_mark)
    }
    joblib.dump(artifact, MODEL_DIR / "kmeans.joblib")

    result = data[[
        "Student_ID", "Student_Name", "Department", "Semester",
        *KMEANS_FEATURES
    ]].copy()
    result["Cluster"] = labels
    return result, artifact


def load_logistic():
    path = MODEL_DIR / "logistic_regression.joblib"
    return joblib.load(path) if path.exists() else None


def load_kmeans():
    path = MODEL_DIR / "kmeans.joblib"
    return joblib.load(path) if path.exists() else None


def predict_pass_probability(artifact, attendance, study_hours,
                             previous_average, semester):
    previous = (
        artifact["median_previous_average"]
        if previous_average is None else float(previous_average)
    )
    row = pd.DataFrame([{
        "Attendance_Percentage": float(attendance),
        "Study_Hours": float(study_hours),
        "Previous_Semester_Average": previous,
        "Semester": int(semester)
    }])
    probability = artifact["model"].predict_proba(row)[0]
    classes = artifact["model"].named_steps["classifier"].classes_
    pass_index = list(classes).index(1)
    return float(probability[pass_index])
