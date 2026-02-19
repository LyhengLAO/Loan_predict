import os
from pathlib import Path

import pandas as pd
from joblib import dump, Memory

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def gridsearch_compare_models(
    df_or_path,
    model_dir,
    test_size=0.2,
    random_state=42,
    scoring="f1",
    pos_label=1,
    n_splits=5,
    n_jobs=-1,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    df = pd.read_csv(df_or_path) if isinstance(df_or_path, (str, Path)) else df_or_path

    X = df.drop(columns=["Loan_ID", "Loan_Status"])
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Optional: cache pipeline steps (useful if preprocessing is heavy)
    memory = Memory(location=model_dir / "pipeline_cache", verbose=0)

    models = {
        "LogReg": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=5000)),
                ],
                memory=memory,
            ),
            {
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__solver": ["lbfgs", "liblinear"],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier()),
                ],
                memory=memory,
            ),
            {
                "clf__n_neighbors": [3, 5, 7, 11],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        "SVC": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", SVC()),  # keep probability=False for speed
                ],
                memory=memory,
            ),
            {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["linear", "rbf"],
                "clf__gamma": ["scale", "auto"],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=random_state),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.5],
            },
        ),
    }

    rows_scores = []
    rows_best_params = []

    best_overall = {"name": None, "estimator": None, "cv_score": -1.0}

    for name, (estimator, param_grid) in models.items():
        print(f"\n=== GridSearch: {name} ===")

        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
        )
        gs.fit(X_train, y_train)

        # Save best estimator for this algorithm (joblib is faster/more compact than pickle)
        algo_path = model_dir / f"model_{name}.joblib"
        dump(gs.best_estimator_, algo_path)

        # Track best overall (by CV score)
        if gs.best_score_ > best_overall["cv_score"]:
            best_overall.update(
                {"name": name, "estimator": gs.best_estimator_, "cv_score": gs.best_score_}
            )

        # Evaluate on test split
        y_pred = gs.predict(X_test)

        rows_scores.append(
            {
                "model": name,
                "best_cv_score": gs.best_score_,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
                "recall": recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
                "f1": f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
            }
        )

        rows_best_params.append({"model": name, "best_params": gs.best_params_})

        print(f"Best CV {scoring}: {gs.best_score_:.4f}")
        print(f"Saved: {algo_path.name}")

    df_scores = pd.DataFrame(rows_scores).sort_values("f1", ascending=False).reset_index(drop=True)
    df_best_params = pd.DataFrame(rows_best_params)

    # Save best overall model too
    if best_overall["estimator"] is not None:
        best_path = model_dir / "best_model_overall.joblib"
        dump(best_overall["estimator"], best_path)
        print(f"\n✅ Best overall by CV: {best_overall['name']} ({best_overall['cv_score']:.4f})")
        print(f"Saved best overall: {best_path.name}")

    return df_scores, df_best_params


if __name__ == "__main__":
    now = Path.cwd()
    root = now.parent.parent
    data_path = root / "data" / "processed" / "train_processed.csv"
    model_dir = root / "model"

    df_scores, df_best_params = gridsearch_compare_models(
        df_or_path=data_path,
        model_dir=model_dir,
        scoring="f1",
        pos_label=1,
        n_jobs=-1,
    )

    df_scores.to_csv(model_dir / "model_comparison.csv", index=False)
    df_best_params.to_csv(model_dir / "best_params.csv", index=False)
