"""Run the Week 8 tuned XGBoost comparison: python compare_feature_selection.py."""
import copy
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from main import load_config, run_pipeline
from src.evaluate import corrected_paired_test


def main():
    config = load_config()
    if config["model"]["type"] != "xgboost":
        raise ValueError("This comparison requires model.type: xgboost")
    config["tuning"]["enabled"] = True
    output = Path(config.get("output", {}).get("results_dir", "results")) / "week8"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for enabled in (True, False):
        candidate = copy.deepcopy(config)
        candidate["feature_selection"]["enabled"] = enabled
        label = "with_selection" if enabled else "without_selection"
        print(f"Starting {label}: {config['tuning']['n_trials']} trials, nested CV", flush=True)
        with TemporaryDirectory(prefix="pipeline_cache_") as cache:
            with (output / f"{label}.log").open("w") as log, redirect_stdout(log):
                row = run_pipeline(candidate, cache)
        rows.append(row)
        print(f"Finished {label}: {row['mean']:.6f} +/- {row['std']:.6f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "scoreboard.csv", index=False)
    folds = [f"fold_{i}" for i in range(1, config["cv"]["n_splits"] + 1)]
    result = corrected_paired_test(
        [rows[0][f] for f in folds], [rows[1][f] for f in folds],
        test_train_ratio=1 / (config["cv"]["n_splits"] - 1),
    )
    result.update(run_with_selection=rows[0]["run"], run_without_selection=rows[1]["run"],
                  metric=config["cv"]["selection_metric"], difference="with - without selection")
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
