"""
Step 2: Load encodings from claim_encodings_v2.npz and train the logistic regression
classifier with cross-validation.
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
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_predict,
    cross_validate,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
# Updated encodings format (v2).
# Prefer repo-local encodings, but also handle the case where the file
# is located one directory above the script (seen in some local setups).
ENCODINGS_PATH_CANDIDATES = [
    PROJECT_ROOT / "claim_encodings_v2.npz",
    PROJECT_ROOT.parent / "claim_encodings_v2.npz",
]
ENCODINGS_PATH = ENCODINGS_PATH_CANDIDATES[0]
WEIGHTS_JSON_PATH = PROJECT_ROOT / "src" / "public" / "claim_lr_weights.json"

RANDOM_STATE = 42
N_JOBS = 1


def main():
    encodings_path = next(
        (p for p in ENCODINGS_PATH_CANDIDATES if p.exists()), None
    )
    if encodings_path is None:
        tried = ", ".join(str(p) for p in ENCODINGS_PATH_CANDIDATES)
        raise FileNotFoundError(
            f"Encodings file not found (expected claim_encodings_v2.npz). "
            f"Tried: {tried}. Run lr_encode.py first."
        )

    print(f"Loading encodings from {encodings_path}...")
    data = np.load(encodings_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    class_names = data["class_names"]
    if class_names.ndim == 0:
        class_names = class_names.item()
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Inconsistent encodings: X and y have different sample counts. "
            "This usually means the NPZ was generated with an older/broken `lr_encode.py`. "
            "Please re-run `lr_encode.py` to regenerate `claim_encodings_v2.npz`."
        )

    # 10-fold CV for a more stable bias-variance tradeoff than 5-fold.
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    # Feature scaling + L2 (ridge) regularization with GridSearchCV.
    # Note: scikit-learn uses `C` as inverse regularization strength (larger C => less regularization).
    c_grid = np.logspace(-3, 3, num=13)

    base_clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        tol=1e-4,
        max_iter=10000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        [
            # StandardScaler lets the ridge term behave more consistently across embedding dimensions.
            ("scaler", StandardScaler(copy=False)),
            ("clf", base_clf),
        ]
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={"clf__C": c_grid},
        scoring="f1_weighted",
        cv=cv,
        # Avoid OOM: running CV folds in parallel duplicates data slices in memory.
        n_jobs=N_JOBS,
        pre_dispatch=1,
        refit=True,
    )

    print("\nRunning GridSearchCV for L2 regularization (C) ...")
    grid.fit(X, y)
    best_estimator = grid.best_estimator_
    best_c = grid.best_params_["clf__C"]
    print(f"Best C selected: {best_c:.5g} (Mean weighted F1={grid.best_score_:.4f})")

    cv_results = cross_validate(
        best_estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=N_JOBS,
    )
    y_pred = cross_val_predict(best_estimator, X, y, cv=cv, n_jobs=N_JOBS)

    print("\n10-fold cross-validation metrics (with best C):")
    for i in range(10):
        print(
            f"  Fold {i + 1}: accuracy={cv_results['test_accuracy'][i]:.4f}, "
            f"precision={cv_results['test_precision'][i]:.4f}, "
            f"recall={cv_results['test_recall'][i]:.4f}, "
            f"F1={cv_results['test_f1'][i]:.4f}"
        )
    print(
        f"  Mean accuracy:  {cv_results['test_accuracy'].mean():.4f} "
        f"(+/- {cv_results['test_accuracy'].std() * 2:.4f})"
    )
    print(
        f"  Mean precision: {cv_results['test_precision'].mean():.4f} "
        f"(+/- {cv_results['test_precision'].std() * 2:.4f})"
    )
    print(
        f"  Mean recall:    {cv_results['test_recall'].mean():.4f} "
        f"(+/- {cv_results['test_recall'].std() * 2:.4f})"
    )
    print(
        f"  Mean F1:        {cv_results['test_f1'].mean():.4f} "
        f"(+/- {cv_results['test_f1'].std() * 2:.4f})"
    )

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

    # Export weights for the browser extension (no server required)
    scaler = best_estimator.named_steps["scaler"]
    clf = best_estimator.named_steps["clf"]
    weights_data = {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classes": list(class_names),
    }
    WEIGHTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=2)
    print(f"Weights exported to {WEIGHTS_JSON_PATH} for the extension.")
    return best_estimator


if __name__ == "__main__":
    main()
