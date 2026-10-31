"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (accuracy, classification report, fairness table) to a timestamped file in `results/` instead, so youcan open it again later, or compare two runs side by side after changing something in config.yaml.
"""
import os
import pandas as pd
import yaml


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Append a run to the scoreboard CSV and return all recorded runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)

    path = os.path.join(results_dir, f"run_{run}.txt")

    header = (
        f"Run: {run}\n"
        "Configuration:\n"
        f"{yaml.safe_dump(config, sort_keys=False)}\n"
        f"Locked test size: {config['test_set']['size']}  "
        f"random_state: {config['test_set']['random_state']}\n"
        f"CV: {config['cv']['n_splits']} stratified folds; "
        f"shuffle: {config['cv']['shuffle']}; random_state: {config['cv']['random_state']}; "
        f"metrics: {config['cv']['metrics']}; "
        f"selection metric: {config['cv']['selection_metric']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path
