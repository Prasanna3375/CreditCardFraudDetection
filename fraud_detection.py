
# CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)

# 1. LOAD DATASET
df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:", df.isnull().sum().sum())
print("\nClass distribution:")
print(df["Class"].value_counts())


# 2. VISUALIZE CLASS IMBALANCE
counts = df["Class"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=["Legitimate", "Fraudulent"],
    y=[counts[0], counts[1]]
)

ax.set_yscale("log")

plt.title("Credit Card Transactions: Class Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions (Log Scale)")

for i, count in enumerate([counts[0], counts[1]]):
    plt.text(i, count * 1.15, f"{count:,}", ha="center")

plt.tight_layout()
plt.show()

# 3. SEPARATE FEATURES AND TARGET
X = df.drop(columns=["Class"])
y = df["Class"]

# 4. SPLIT DATA BEFORE MODEL TRAINING
# Stratification preserves the fraud proportion in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. DEFINE MACHINE LEARNING MODELS
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = {}
best_model = None
best_name = None
best_score = -1

# 6. TRAIN AND EVALUATE MODELS
for name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", name)

    model.fit(X_train, y_train)

    # Predict fraud probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Starting threshold; tune using validation data later.
    threshold = 0.50
    predictions = (probabilities >= threshold).astype(int)

    # Evaluation
    ap = average_precision_score(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)

    print("Average Precision (PR-AUC):", round(ap, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions)

    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    results[name] = {
        "Average Precision": ap,
        "ROC-AUC": auc
    }

    # Select by Average Precision, not accuracy.
    if ap > best_score:
        best_score = ap
        best_model = model
        best_name = name

# 7. COMPARE MODELS
comparison = pd.DataFrame(results).T
print("\nMODEL COMPARISON")
print(comparison.sort_values(
    by="Average Precision",
    ascending=False
))

# 8. SAVE THE BEST MODEL
joblib.dump(best_model, "fraud_detection_model.pkl")

print("\nBest model:", best_name)
print("Average Precision:", round(best_score, 4))
print("Model saved as fraud_detection_model.pkl")

# 9. PREDICT ON A NEW TRANSACTION
# This is only an example from the test set.
sample = X_test.iloc[[0]]
probability = best_model.predict_proba(sample)[0, 1]

print("\nExample transaction fraud probability:",
      round(probability * 100, 2), "%")

if probability >= 0.50:
    print("Prediction: POTENTIAL FRAUD")
else:
    print("Prediction: LIKELY LEGITIMATE")

# 5. DEFINE MACHINE LEARNING MODELS
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

# 5. TRAIN AND COMPARE MULTIPLE ML MODELS

import time

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=12,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=100,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ),

    "Histogram Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=15,
        random_state=42
    )
}

results = []
trained_models = {}

# Calculate weights for the rare fraud class.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)

    start_time = time.time()

    # Train the model.
    if name == "Histogram Gradient Boosting":
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - start_time

    # Predict probabilities on unseen test data.
    probabilities = model.predict_proba(X_test)[:, 1]

    # Use 0.50 as an initial threshold.
    predictions = (probabilities >= 0.50).astype(int)

    # Calculate evaluation metrics.
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
        roc_auc_score
    )

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    ap = average_precision_score(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "PR-AUC": ap,
        "ROC-AUC": auc,
        "Training Time (sec)": elapsed
    })

    trained_models[name] = model

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"PR-AUC:    {ap:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Training time: {elapsed:.2f} seconds")


# 6. COMPARE ALL MODELS

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="PR-AUC",
    ascending=False
).reset_index(drop=True)

print("\n\nMODEL COMPARISON TABLE")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

# 7. VISUALIZE MODEL PERFORMANCE

plt.figure(figsize=(11, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="PR-AUC"
)

plt.title("Credit Card Fraud Detection: Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Average Precision (PR-AUC)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# 8. EVALUATE AND SAVE THE BEST MODEL

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\nBest model by PR-AUC:", best_name)

best_probabilities = best_model.predict_proba(X_test)[:, 1]
best_predictions = (best_probabilities >= 0.50).astype(int)

print("\nClassification Report:")
print(classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

joblib.dump(best_model, "fraud_detection_model.pkl")

print("\nProject files saved:")
print("- model_comparison.csv")
print("- fraud_detection_model.pkl")
print("\nAll model training and evaluation completed.")

