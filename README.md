# COMPAS Fairness Diagnostics

Two complementary approaches to auditing and mitigating bias in a recidivism-risk classifier trained on the ProPublica COMPAS dataset:

1. **Accuracy-constrained fairness optimization** – tune the classifier's decision threshold per protected group (race, sex, age category) to trade a bounded amount of accuracy for improved disparate-impact (p-rule) scores.
2. **Information-theoretic fairness-aware feature selection** – use mutual information, conditional mutual information, and Shapley-value attribution to quantify how much each feature contributes to accuracy versus discriminatory effect, then use that to pick a feature set that keeps accuracy while reducing calibration gaps across groups.

The full analysis, including methodology notes, code, and commentary on the results, lives in [notebooks/fairness_diagnostics.ipynb](notebooks/fairness_diagnostics.ipynb).

## Repository structure

```
.
├── data/                    read-only input data
│   └── compas-scores-two-years.csv
├── figures/                 generated plots (safe to delete and regenerate)
│   └── accuracy_prule_vs_gamma.png
├── notebooks/
│   └── fairness_diagnostics.ipynb
├── requirements.txt
└── README.md
```

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the analysis

```
jupyter nbconvert --to notebook --execute --inplace notebooks/fairness_diagnostics.ipynb
```

The notebook reads `data/compas-scores-two-years.csv` and writes `figures/accuracy_prule_vs_gamma.png`; both paths are relative, so run it from the `notebooks/` directory or keep the repository layout intact.

## Key findings

- Sweeping the accuracy-budget parameter (gamma) from 0 to 0.26 shows the expected trade-off: accuracy declines roughly linearly while the p-rule (disparate-impact score) improves, with most of the fairness gain achieved by gamma ≈ 0.15.
- Among the demographic attributes tested, `race` carries the most discriminatory signal relative to its accuracy contribution – dropping it from the feature set ties for the best validation accuracy while measurably shrinking the calibration gap between groups.
- `age_cat` is the opposite case: removing it costs both accuracy and calibration, making it a feature worth keeping despite its correlation with the protected attributes.

## Data source

`data/compas-scores-two-years.csv` is the public COMPAS recidivism dataset released by ProPublica for their 2016 "Machine Bias" investigation.
