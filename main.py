"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> clean -> deduplicate -> dev/test split
    -> cross-validate preprocessing + model on development data
    -> out-of-fold evaluation -> final development refit -> save results
"""
import yaml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target,
    split_dev_test, build_preprocessor,
)
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline, cv_report, oof_classification_report, fairness_report,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])

    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))
    prep_cfg = config["preprocessing"]
    X, y, extras = split_features_target(
        df_clean, config["data"], prep_cfg.get("mnar_indicator_sources", [])
    )
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    pipe = Pipeline([
        ("prep", build_preprocessor(prep_cfg)),
        ("model", build_model(config["model"])),
    ])
    cv_cfg = config["cv"]
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"] if cv_cfg["shuffle"] else None,
    )
    scoring = cv_cfg.get("scoring", "accuracy")
    fold_scores, y_oof = cross_validate_pipeline(
        pipe, X_dev, y_dev, cv, scoring, n_jobs=cv_cfg.get("n_jobs", 1)
    )
    report = cv_report(fold_scores, scoring)
    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"],
        context="development set, out-of-fold",
    )

    # CV evaluates fresh fold models; refit the recipe on all development rows for use.
    # The locked test set is neither scored nor used for model selection here.
    pipe.fit(X_dev, y_dev)
    final_summary = (
        f"Final model refit on {len(X_dev)} development rows. "
        f"Locked test set: {len(X_test)} rows (not evaluated)."
    )
    print(final_summary)
    report += "\n" + final_summary + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
