# Baseline Predictive Pipeline -- ETAI

**Name:** Ulaş Keskin  
**Student number:** 20260612

Week 2: 

Logistic regression achieved better test accuracy than the decision tree (67.9% vs 62.7%) in the saved runs. The tree trained faster: about 0.015 s versus 0.206 s for logistic regression. I think logistic regression generalised better because its simpler, regularised model reduced overfitting: its train/test accuracy was 67.9%/67.9%, compared with the tree's 82.9%/62.7%. Logistic regression therefore performed better on this split, while the tree was faster; but since tree was overfitting it makes much more sence to go with logistic regression.

Week 3:

Cleaning removed duplicates and redundant columns, standardised categories, and marked invalid values as missing. Test accuracy decreased for both models: from 63.1% to 60.4% for the decision tree and from 67.9% to 65.5% for logistic regression. I still think logistic regression generalised better: its cleaned train/test accuracy was 67.8%/65.5%, compared with the tree's 79.9%/60.4%. However, cleaning changed which rows were included in the test set, so these decreases do not necessarily mean cleaning made the models worse.

Week 4:

This week, I replaced dropping rows with missing values with median imputation for numeric features and most-frequent imputation for categorical features. I also added missingness flags for `priors_count` and `c_charge_degree`, target encoding, and robust scaling, with preprocessing and the model fitted together in one Pipeline using only development data. Compared with Week 3, logistic regression's train/test accuracy changed from 67.8%/65.5% to 67.6%/65.8%. Test accuracy increased by 0.3 percentage points, while the train/test gap decreased from 2.3 to 1.8 percentage points. However, retaining incomplete records changed the test set from 1,188 to 1,443 rows, so this is not a comparison on identical examples and does not prove that the new recipe improved performance. Recall for the recidivism class also decreased from 0.51 to 0.48. I think the main improvement is the preprocessing approach: it keeps more data and learns imputation, encoding, and scaling from training data only. The challenge below adds cross-validation within the development set to assess how stable the results are.

### Challenge

Before cross-validation, the Week 4 logistic regression pipeline reported 65.8% test accuracy from one split (Week 3: 65.5%). After adding stratified 5-fold CV around the whole Pipeline, development-set validation accuracy was **67.2% ± 1.3 percentage points** (mean ± sample standard deviation), with a mean train–validation gap of **0.3 percentage points**, compared with the previous 1.8-point train–test gap. I trust CV more for comparing recipes because every development row is evaluated out of fold and preprocessing is relearned inside each fold. This is a change in evaluation, not proof of better predictions: CV uses different evaluation rows and training subsets, while the 1,443-row test set stays excluded from CV and was not rescored.

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning, dev/test split, imputation/encoding/scaling
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | Data cleaning | Configurable placeholder handling, numeric conversion, validity rules, category standardisation, duplicate removal, and redundant-column removal before preprocessing; before/after results discussed above. |
| 4 | Preprocessing recipe | Row-preserving cleaning; training-only deduplication; locked dev/test split; MNAR flags; train-fitted imputation, encoding and scaling in one model Pipeline; dummy and random forest models; stratified 5-fold CV with out-of-fold classification and fairness reports. |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, cleans the data, locks away 20% as the test set, and
cross-validates the whole preprocessing/model Pipeline on the development set. It prints:

- Per-fold training and validation accuracy, their gaps, and mean ± sample standard deviation.
- A classification report using out-of-fold predictions for all development rows.
- A false-positive-rate-by-race comparison with COMPAS using those same predictions.

The recipe is then refitted on all development rows. The locked test set is not
scored during this workflow. Reports and configuration are saved to timestamped
files in `results/`; compare candidates using the same development folds.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.

### Cleaning configuration

`diagnostics` in `config.yaml` controls placeholder tokens, numeric conversion,
validity bounds, category mappings, ID deduplication, and redundant columns.
Cleaning copies the input and runs before preprocessing. Invalid values become
missing; cleaning preserves rows. Duplicate removal runs separately on training data
before the dev/test split. Numeric and categorical missing values are imputed inside
the model Pipeline, using only development rows.
Saved reports include the full configuration so the cleaning settings are recorded.

Run the pipeline with `python main.py` and review the saved report in `results/`, as before.

### Preprocessing configuration

`test_set` fixes the held-out fraction and seed. `preprocessing` selects numeric and
categorical features, MNAR indicator sources, imputation rules, encoder and scaler.
The default recipe uses target encoding with five-fold internal cross-fitting and
robust scaling. `TargetEncoder(cv=5, shuffle=True, random_state=42)` keeps the
notebook's stratified binary cross-fitting behavior with scikit-learn 1.3+.
Available encoders: `onehot`, `ordinal`, `count`, `target`; scalers: `none`,
`standard`, `minmax`, `robust`. Missing COMPAS scores are excluded from COMPAS FPR
comparisons rather than treated as high risk.

`cv` configures stratified cross-validation (`n_splits`, `shuffle`, `random_state`,
`scoring`, `n_jobs`). Each fold fits its own preprocessing and model; out-of-fold
predictions come from those same fits. `main.py` then refits on all development
rows, leaving the locked test set for a later final assessment.
Use `model.params: {strategy: "most_frequent"}` for the dummy baseline, or
`{random_state: 42}` as initial parameters for a tree/random forest.

Run regression checks with `python -m unittest discover -s tests`.
