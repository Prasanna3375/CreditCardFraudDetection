
# ============================================================
# CREDIT CARD FRAUD DETECTION - 11 ML MODELS
# ============================================================

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    IsolationForest
)
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
np.random.seed(42)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Class"].value_counts())

X = df.drop(columns=["Class"])
y = df["Class"]

# ------------------------------------------------------------
# 2. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Use class weights for models that support them.
weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# Fit the scaler on training data only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN and classic Gradient Boosting can be slow on
# the full training set. Use a reproducible stratified
# subset for these two models.
X_train_df, _, y_train_df, _ = train_test_split(
    X_train,
    y_train,
    train_size=min(50000, len(X_train)),
    random_state=42,
    stratify=y_train
)

# Recover the scaled rows corresponding to the subset.
# Indices are preserved by train_test_split.
X_train_knn_scaled = scaler.transform(X_train_df)

# ------------------------------------------------------------
# 3. DEFINE SUPERVISED MODELS
# ------------------------------------------------------------

models = {
    "Logistic Regression": (
        LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=42
        ),
        "scaled"
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=5,
            random_state=42
        ),
        "raw"
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        ),
        "raw"
    ),

    "SVM (Linear)": (
        LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=3000,
            random_state=42
        ),
        "scaled"
    ),

    "KNN": (
        KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1
        ),
        "knn_subset"
    ),

    "Naive Bayes": (
        GaussianNB(),
        "scaled"
    ),

    "Extra Trees": (
        ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        ),
        "raw"
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.1,
            random_state=42
        ),
        "gb_subset"
    ),

    "HistGradientBoosting": (
        HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.1,
            random_state=42
        ),
        "weighted"
    )
}

results = []
trained_models = {}

# ------------------------------------------------------------
# 4. TRAIN AND EVALUATE SUPERVISED MODELS
# ------------------------------------------------------------

for name, (model, mode) in models.items():

    print("\n" + "=" * 65)
    print("Training:", name)

    start = time.time()

    if mode == "scaled":
        model.fit(X_train_scaled, y_train)
        scores = model.decision_function(X_test_scaled) \
            if hasattr(model, "decision_function") \
            else model.predict_proba(X_test_scaled)[:, 1]

    elif mode == "knn_subset":
        model.fit(X_train_knn_scaled, y_train_df)
        scores = model.predict_proba(X_test_scaled)[:, 1]

    elif mode == "gb_subset":
        model.fit(X_train_df, y_train_df)
        scores = model.predict_proba(X_test)[:, 1]

    elif mode == "weighted":
        model.fit(X_train, y_train, sample_weight=weights)
        scores = model.predict_proba(X_test)[:, 1]

    else:
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]

    elapsed = time.time() - start

    # Scores from LinearSVC are decision scores rather
    # than calibrated probabilities. They still support
    # ranking metrics such as ROC-AUC and average precision.
    predictions = (scores >= 0.5).astype(int) \
        if mode != "scaled" or name != "SVM (Linear)" \
        else (scores >= 0).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    ap = average_precision_score(y_test, scores)
    auc = roc_auc_score(y_test, scores)

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "PR-AUC": ap,
        "ROC-AUC": auc,
        "Training Seconds": elapsed
    })

    trained_models[name] = model

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"PR-AUC:    {ap:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Time:      {elapsed:.2f} seconds")

# ------------------------------------------------------------
# 5. XGBOOST (OPTIONAL INSTALLATION)
# ------------------------------------------------------------

try:
    from xgboost import XGBClassifier

    print("\n" + "=" * 65)
    print("Training: XGBoost")

    start = time.time()

    # Give fraud examples greater training weight.
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    xgb.fit(X_train, y_train)
    scores = xgb.predict_proba(X_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)

    results.append({
        "Model": "XGBoost",
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "PR-AUC": average_precision_score(y_test, scores),
        "ROC-AUC": roc_auc_score(y_test, scores),
        "Training Seconds": time.time() - start
    })

    trained_models["XGBoost"] = xgb

except ImportError:
    print("\nXGBoost is not installed. Skipping it.")
    print("Install with: python -m pip install xgboost")

# ------------------------------------------------------------
# 6. ISOLATION FOREST - UNSUPERVISED ANOMALY DETECTION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("Training: Isolation Forest")

start = time.time()

iso = IsolationForest(
    n_estimators=150,
    contamination=float(y_train.mean()),
    random_state=42,
    n_jobs=-1
)

iso.fit(X_train)

# Lower decision_function scores indicate anomalies.
# Negating makes higher scores indicate greater anomaly.
iso_scores = -iso.decision_function(X_test)

# Isolation Forest labels anomalies as -1.
iso_predictions = (iso.predict(X_test) == -1).astype(int)

results.append({
    "Model": "Isolation Forest",
    "Precision": precision_score(
        y_test, iso_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, iso_predictions, zero_division=0
    ),
    "F1-Score": f1_score(
        y_test, iso_predictions, zero_division=0
    ),
    "PR-AUC": average_precision_score(y_test, iso_scores),
    "ROC-AUC": roc_auc_score(y_test, iso_scores),
    "Training Seconds": time.time() - start
})

trained_models["Isolation Forest"] = iso

# ------------------------------------------------------------
# 7. COMPARISON TABLE
# ------------------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "PR-AUC", ascending=False
).reset_index(drop=True)

print("\n\nFINAL MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

# ------------------------------------------------------------
# 8. PERFORMANCE GRAPHS
# ------------------------------------------------------------

for metric in ["PR-AUC", "Recall", "F1-Score"]:

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y=metric
    )

    plt.title(f"Model Comparison: {metric}")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(
        metric.lower().replace("-", "_") + "_comparison.png",
        dpi=150
    )
    plt.show()

# ------------------------------------------------------------
# 9. SAVE BEST SUPERVISED MODEL
# ------------------------------------------------------------

# Isolation Forest has a different training objective.
# Select the best supervised model for the initial
# transaction-classification application.
supervised_results = results_df[
    results_df["Model"] != "Isolation Forest"
]

best_name = supervised_results.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(best_model, "fraud_detection_model.pkl")

print("\nBest supervised model by test PR-AUC:", best_name)
print("Saved: fraud_detection_model.pkl")
print("Saved: model_comparison.csv")
print("Saved: pr_auc_comparison.png, recall_comparison.png,")
print("       f1_score_comparison.png")
print("\nAll experiments completed.")
