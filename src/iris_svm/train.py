from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import FEATURE_COLS_DEFAULT, TARGET_COL_DEFAULT, load_dataset


def _maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required for --save-plots. Install it with: pip install matplotlib"
        ) from e


def save_confusion_matrix_png(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    plt = _maybe_import_matplotlib()

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_2d_decision_regions_png(
    X_2d: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, str],
    svc_params: dict[str, object],
    out_path: Path,
    title: str,
) -> None:
    """Train a 2D SVM on (X_2d, y) and plot decision regions.

    Notes:
    - This visualization trains on ALL samples to show overall separability.
    - It is for interpretation only, not the test-set evaluation.
    """
    plt = _maybe_import_matplotlib()

    # Encode labels to integers for coloring
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    class_names = le.classes_.tolist()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_2d)
    svc = SVC(**svc_params)
    svc.fit(Xs, y)

    x_min, x_max = Xs[:, 0].min() - 0.8, Xs[:, 0].max() + 0.8
    y_min, y_max = Xs[:, 1].min() - 0.8, Xs[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 420),
        np.linspace(y_min, y_max, 420),
    )
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_int = le.transform(Z).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=160)
    ax.contourf(xx, yy, Z_int, alpha=0.22, levels=np.arange(len(class_names) + 1) - 0.5, cmap="tab10")
    scatter = ax.scatter(
        Xs[:, 0],
        Xs[:, 1],
        c=y_int,
        s=20,
        cmap="tab10",
        edgecolors="k",
        linewidths=0.25,
    )
    ax.set_title(title)
    ax.set_xlabel(f"{feature_names[0]} (standardized)")
    ax.set_ylabel(f"{feature_names[1]} (standardized)")
    legend = ax.legend(
        handles=scatter.legend_elements()[0],
        labels=class_names,
        title="species",
        loc="best",
        frameon=True,
    )
    ax.add_artist(legend)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _resolve_data_path(raw: str) -> Path:
    """Resolve dataset path with a gentle fallback.

    If the user didn't change the default and data/iris.csv is missing but iris.csv exists,
    we fall back to ./iris.csv.
    """
    p = Path(raw)
    if p.exists():
        return p

    default = Path("data/iris.csv")
    if Path(raw) == default and Path("iris.csv").exists():
        return Path("iris.csv")

    return p


def build_pipeline(kernel: str, C: float, gamma: str | float, degree: int) -> Pipeline:
    svc_kwargs: dict[str, object] = {
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
    }
    if kernel == "poly":
        svc_kwargs["degree"] = degree

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(**svc_kwargs)),
        ]
    )


def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, np.ndarray, str]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    return acc, macro_f1, cm, report


def run_gridsearch(X_train: np.ndarray, y_train: np.ndarray, random_state: int, n_splits: int) -> GridSearchCV:
    pipe = Pipeline(steps=[("scaler", StandardScaler()), ("svc", SVC())])

    param_grid = [
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.01, 0.1, 1],
        },
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.1, 1, 10, 100],
        },
        {
            "svc__kernel": ["poly"],
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", 0.01, 0.1],
            "svc__degree": [2, 3, 4],
        },
    ]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    return gs


def print_top_grid_results(gs: GridSearchCV, top_k: int) -> None:
    if top_k <= 0:
        return

    results = gs.cv_results_
    ranks = results.get("rank_test_score")
    means = results.get("mean_test_score")
    stds = results.get("std_test_score")
    params = results.get("params")
    if ranks is None or means is None or stds is None or params is None:
        return

    order = np.argsort(ranks)
    print(f"\nTop {min(top_k, len(order))} CV configs:")
    for idx in order[:top_k]:
        print(f"  rank={int(ranks[idx])} mean={means[idx]:.4f} std={stds[idx]:.4f} params={params[idx]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SVM on iris.csv using sklearn.svm.SVC")
    p.add_argument("--data", type=str, default="data/iris.csv", help="Path to iris.csv")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument("--kernel", type=str, default="rbf", choices=["linear", "rbf", "poly"], help="SVC kernel")
    p.add_argument("--C", type=float, default=1.0, help="Regularization strength")
    p.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="Kernel coefficient: 'scale'/'auto' or a float",
    )
    p.add_argument("--degree", type=int, default=3, help="Degree for poly kernel")
    p.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a simple tuning demo: baseline -> GridSearchCV -> compare test metrics",
    )
    p.add_argument(
        "--gridsearch",
        action="store_true",
        help="(Compatibility) Run GridSearchCV (macro-F1 scoring) on train split",
    )
    p.add_argument("--cv-splits", type=int, default=5, help="CV folds for gridsearch")
    p.add_argument("--topk", type=int, default=5, help="When using --gridsearch, print top-k CV configs")
    p.add_argument("--save-plots", action="store_true", help="Save plots to report/outputs (confusion matrix + 2D decision regions)")
    p.add_argument("--outdir", type=str, default="report/outputs", help="Output directory for plots when --save-plots is set")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_path = _resolve_data_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = load_dataset(data_path, feature_cols=FEATURE_COLS_DEFAULT, target_col=TARGET_COL_DEFAULT)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=dataset.y,
    )

    try:
        gamma_value: str | float = float(args.gamma)
    except ValueError:
        gamma_value = args.gamma

    do_tune = bool(args.tune) or bool(args.gridsearch)

    if not do_tune:
        model = build_pipeline(kernel=args.kernel, C=args.C, gamma=gamma_value, degree=args.degree)
        model.fit(X_train, y_train)
        print("=== Model params ===")
        print(model.get_params()["svc"].get_params())

        acc, macro_f1, cm, report = evaluate(model, X_test, y_test)
        print("\n=== Hold-out test metrics ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(cm)
        print("\nClassification report:")
        print(report)
        return 0

    # Tuning demo: baseline -> grid search -> compare
    print("=== Tuning demo (baseline -> GridSearchCV) ===")
    print(
        f"Split: test_size={args.test_size}, random_state={args.random_state}, stratify=True; CV folds={args.cv_splits}"
    )

    baseline = build_pipeline(kernel=args.kernel, C=args.C, gamma=gamma_value, degree=args.degree)
    baseline.fit(X_train, y_train)
    base_acc, base_f1, base_cm, base_report = evaluate(baseline, X_test, y_test)
    print("\n--- Baseline model ---")
    print(baseline.get_params()["svc"].get_params())
    print(f"Test Accuracy: {base_acc:.4f}")
    print(f"Test Macro-F1: {base_f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(base_cm)

    gs = run_gridsearch(X_train, y_train, random_state=args.random_state, n_splits=args.cv_splits)
    tuned = gs.best_estimator_
    tuned_acc, tuned_f1, tuned_cm, tuned_report = evaluate(tuned, X_test, y_test)

    print("\n--- GridSearchCV (scoring=f1_macro) ---")
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV score (macro-F1): {gs.best_score_:.4f}")
    print_top_grid_results(gs, top_k=args.topk)

    print("\n--- Tuned model (best_estimator_) ---")
    print(f"Test Accuracy: {tuned_acc:.4f}")
    print(f"Test Macro-F1: {tuned_f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(tuned_cm)

    print("\n=== Comparison (tuned - baseline) ===")
    print(f"Δ Accuracy: {tuned_acc - base_acc:+.4f}")
    print(f"Δ Macro-F1: {tuned_f1 - base_f1:+.4f}")

    print("\n--- Classification report (baseline) ---")
    print(base_report)
    print("\n--- Classification report (tuned) ---")
    print(tuned_report)

    if args.save_plots:
        outdir = Path(args.outdir)
        class_names = sorted(np.unique(dataset.y).tolist())
        save_confusion_matrix_png(
            base_cm,
            class_names=class_names,
            out_path=outdir / "confusion_matrix_baseline.png",
            title="Iris SVM Confusion Matrix (baseline)",
        )
        save_confusion_matrix_png(
            tuned_cm,
            class_names=class_names,
            out_path=outdir / "confusion_matrix_tuned.png",
            title="Iris SVM Confusion Matrix (tuned)",
        )

        # 2D decision regions on petal features (interpretability visualization)
        petal_idx = [FEATURE_COLS_DEFAULT.index("petal_length"), FEATURE_COLS_DEFAULT.index("petal_width")]
        X_petal = dataset.X[:, petal_idx]
        best_svc_params = tuned.get_params()["svc"].get_params()
        svc_params_2d = {
            "kernel": best_svc_params.get("kernel"),
            "C": best_svc_params.get("C"),
            "gamma": best_svc_params.get("gamma"),
            "degree": best_svc_params.get("degree"),
        }
        save_2d_decision_regions_png(
            X_2d=X_petal,
            y=dataset.y,
            feature_names=("petal_length", "petal_width"),
            svc_params=svc_params_2d,
            out_path=outdir / "decision_regions_petal.png",
            title="SVM decision regions (petal_length vs petal_width)",
        )
        print(f"\n[plots] Saved to: {outdir.resolve()}")

    return 0
