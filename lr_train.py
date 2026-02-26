"""
Step 2: Load encodings from claim_encodings.npz and train the logistic regression
classifier with 5-fold cross-validation.
Run lr_encode.py first to generate the encodings file.
"""
from pathlib import Path

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
ENCODINGS_PATH = PROJECT_ROOT / "claim_encodings.npz"
WEIGHTS_JSON_PATH = PROJECT_ROOT / "src" / "public" / "claim_lr_weights.json"


def main():
    if not ENCODINGS_PATH.exists():
        raise FileNotFoundError(
            f"Encodings file not found: {ENCODINGS_PATH}. Run lr_encode.py first."
        )

    print(f"Loading encodings from {ENCODINGS_PATH}...")
    data = np.load(ENCODINGS_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    class_names = data["class_names"]
    if class_names.ndim == 0:
        class_names = class_names.item()
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    clf = LogisticRegression(
        C=1.0,
        l1_ratio=1.0,
        dual=False,
        tol=1e-4,
        max_iter=10000,
        solver="liblinear",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }
    print("\nRunning 5-fold cross-validation...")
    cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    print("\n5-fold cross-validation metrics:")
    for i in range(5):
        print(
            f"  Fold {i + 1}: accuracy={cv_results['test_accuracy'][i]:.4f}, "
            f"precision={cv_results['test_precision'][i]:.4f}, "
            f"recall={cv_results['test_recall'][i]:.4f}, "
            f"F1={cv_results['test_f1'][i]:.4f}"
        )
    print(f"  Mean accuracy:  {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std() * 2:.4f})")
    print(f"  Mean precision: {cv_results['test_precision'].mean():.4f} (+/- {cv_results['test_precision'].std() * 2:.4f})")
    print(f"  Mean recall:    {cv_results['test_recall'].mean():.4f} (+/- {cv_results['test_recall'].std() * 2:.4f})")
    print(f"  Mean F1:        {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std() * 2:.4f})")

    print("\nOverall metrics (out-of-fold predictions):")
    print(f"  Precision (weighted): {precision_score(y, y_pred, average='weighted'):.4f}")
    print(f"  Recall (weighted):   {recall_score(y, y_pred, average='weighted'):.4f}")
    print(f"  F1 score (weighted): {f1_score(y, y_pred, average='weighted'):.4f}")

    cm = confusion_matrix(y, y_pred)
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(f"              {class_names[0]:>12}  {class_names[1]:>12}")
    for i, name in enumerate(class_names):
        print(f"  {name:>12}  {cm[i, 0]:>12}  {cm[i, 1]:>12}")
    print(f"  (total)      {cm.sum(axis=0)[0]:>12}  {cm.sum(axis=0)[1]:>12}")

    print("\nClassification report:")
    print(classification_report(y, y_pred, target_names=class_names))

    clf.fit(X, y)

    # Export weights for the browser extension (no server required)
    weights_data = {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "classes": list(class_names),
    }
    WEIGHTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=2)
    print(f"Weights exported to {WEIGHTS_JSON_PATH} for the extension.")
    return clf


if __name__ == "__main__":
    main()
