# COMPAS Fairness Diagnostics

## What this project does, in plain language

COMPAS is a commercial tool that gives defendants in US courts a "risk of re-offending" score. In 2016 the news organization ProPublica published the scores and the real outcomes for about 7,000 defendants in Broward County, Florida, and argued that the scores treated Black defendants less fairly than white defendants.

This project uses that public data to ask two practical questions about a simple prediction model trained on it:

1. **If we want the model to flag Black and white defendants as "high risk" at more similar rates, how much predictive accuracy do we have to give up?** The model's decision threshold (the probability above which someone is flagged) is moved to trade accuracy for fairness. The same is done for sex and for age groups.
2. **Which input features carry the information that is both useful for prediction and correlated with race?** An information-theoretic score (mutual information combined with Shapley values, following [Khodadadian et al., 2021](https://arxiv.org/abs/2106.00772)) is computed for each feature, and the effect of removing features is checked on held-out data.

Everything is reported over 20 different random train/test splits and compared with simple baselines, because a single split can be misleading. The full analysis is in [notebooks/fairness_diagnostics.ipynb](notebooks/fairness_diagnostics.ipynb); the code lives in a small package under `src/compas_fairness`.

## Key findings

- The logistic regression is **no better than the COMPAS score** it is compared with: accuracy 0.662 against 0.656 and AUC 0.709 against 0.707 on the race split (means over 20 splits), and always predicting the more common outcome already scores 0.520.
- Its p-rule between African-American and Caucasian defendants (about 60%) is lower than the p-rule of the real labels themselves (about 76%), so the model widens the disparity in the data instead of only reproducing it.
- Moving the threshold raises the p-rule, but at a steep price: reaching a p-rule of 80% costs 7.7 accuracy points for race, 9.4 for under-25s, 3.4 for ages 25 to 45 and 2.3 for sex, and at a p-rule near 90% the race model is only a little more accurate than always guessing the more common outcome.
- The features that help prediction most, prior-offense count and age category, are also the strongest proxies for race. **Removing race itself does not change accuracy (0.661 against 0.663) or the accuracy gap between races (0.0257 against 0.0243)**, because the other features carry much of the same information.

## Results

All numbers are on held-out test rows. "20 splits" means the data was split at random 20 times (seeds 0 to 19, 70% training, 30% test) and each cell shows the **mean and the range (minimum to maximum)**. The threshold is always chosen on the training rows only. Baselines are computed on the same test rows as the model. The p-rule is the ratio of the rates at which the two groups are flagged (100% is parity). AUC is the chance that a randomly chosen person who re-offended gets a higher score than one who did not (0.5 is guessing).

### Race (African-American against Caucasian, 5,915 defendants)

| Method | Accuracy | P-rule | AUC |
|---|---|---|---|
| Always predict the more common outcome | 0.520 (0.495–0.541) | not defined | 0.500 |
| COMPAS decile score ≥ 5 (same test rows) | 0.656 (0.639–0.671) | 60.0 (54.7–63.8) | 0.707 (0.689–0.726) |
| Logistic regression, no fairness threshold | 0.662 (0.650–0.679) | 60.1 (55.6–63.7) | 0.709 (0.699–0.724) |
| Logistic regression, `gamma` = 0.10 | 0.598 (0.570–0.637) | 79.1 (74.8–83.9) | |
| Logistic regression, `gamma` = 0.15 | 0.563 (0.529–0.593) | 85.1 (82.6–90.2) | |
| Logistic regression, `gamma` = 0.20 | 0.532 (0.501–0.559) | 90.4 (88.0–93.8) | |

### Sex (protected group: female, 7,214 defendants)

| Method | Accuracy | P-rule | AUC |
|---|---|---|---|
| Always predict the more common outcome | 0.550 (0.532–0.571) | not defined | 0.500 |
| COMPAS decile score ≥ 5 | 0.656 (0.644–0.669) | 91.1 (83.3–99.0) | 0.706 (0.696–0.720) |
| Logistic regression, no fairness threshold | 0.678 (0.663–0.692) | 65.3 (59.1–73.3) | 0.726 (0.705–0.737) |
| Logistic regression, `gamma` = 0.20 | 0.550 (0.515–0.576) | 94.8 (88.6–98.3) | |

### Age group (each group is the protected group in turn, 7,214 defendants)

| Group | P-rule without constraint | P-rule at `gamma` = 0.15 | Accuracy at `gamma` = 0.15 |
|---|---|---|---|
| Under 25 | 24.7 (16.4–30.8) | 79.1 (70.0–97.6) | 0.590 (0.497–0.618) |
| 25 to 45 | 58.3 (52.0–66.0) | 79.0 (47.7–96.9) | 0.564 (0.540–0.591) |
| Over 45 | 86.6 (75.9–99.8) | 96.8 (91.0–99.7) | 0.595 (0.510–0.632) |

The unconstrained age model has accuracy 0.636 (0.622–0.656). Each group gets its own threshold, so each row is a different classifier.

### Accuracy points lost to reach a p-rule of 80% (smallest `gamma` in steps of 0.01)

| Protected attribute | Mean | Range |
|---|---|---|
| Race | 7.7 | 3.2 to 11.0 |
| Age under 25 | 9.4 | 1.8 to 15.8 |
| Age 25 to 45 | 3.4 | −0.1 to 7.0 |
| Sex | 2.3 | 0.4 to 5.5 |
| Age over 45 | −0.2 | −2.5 to 0.0 |

### Feature analysis (race is the protected attribute, 5,915 defendants)

Marginal accuracy and discrimination coefficients (Shapley values) on the 4,225 training rows of the seed-42 split:

| Feature | Accuracy coefficient | Discrimination coefficient |
|---|---|---|
| `priors_count` | 0.0427 | 9.8e-6 |
| `age_cat` | 0.0217 | 1.2e-5 |
| `length_of_stay` | 0.0114 | 4.0e-6 |
| `sex` | 0.0086 | 2.9e-6 |
| `c_charge_degree` | 0.0052 | 3.1e-6 |

Effect of removing one feature from the classifier, test rows, 20 splits:

| Feature removed | Accuracy | Accuracy gap between races | Splits with a smaller gap than with all features |
|---|---|---|---|
| none | 0.663 | 0.0243 | not applicable |
| `race` | 0.661 | 0.0257 | 30% |
| `sex` | 0.660 | 0.0247 | 45% |
| `c_charge_degree` | 0.662 | 0.0270 | 40% |
| `length_of_stay` | 0.655 | 0.0305 | 40% |
| `age_cat` | 0.637 | 0.0297 | 50% |
| `priors_count` | 0.603 | 0.0303 | 55% |

The accuracy gap varies from split to split with a standard deviation of about 0.02, so differences of a few thousandths are not distinguishable from noise.

![Accuracy and p-rule versus gamma](figures/accuracy_prule_vs_gamma.png)

## Glossary

**Columns of the raw data that are used** (`data/compas-scores-two-years.csv`, 7,214 rows, one per defendant):

| Column | Meaning |
|---|---|
| `id` | Defendant identifier, used only to check that splits do not overlap |
| `sex` | Male or Female |
| `age` | Age in years as recorded in the file |
| `age_cat` | Age group: Less than 25, 25 - 45, Greater than 45 (age 45 is in the last group) |
| `race` | Race as recorded; only African-American and Caucasian are analyzed in the race experiments |
| `priors_count` | Number of prior offenses |
| `c_charge_degree` | Degree of the current charge: F (felony) or M (misdemeanor) |
| `c_jail_in`, `c_jail_out` | Dates the person entered and left jail for the current charge |
| `decile_score` | The COMPAS risk score from 1 (lowest) to 10 (highest) |
| `two_year_recid` | 1 if the person re-offended within two years according to the dataset, otherwise 0 |

**Variables derived in the code:**

| Variable | Meaning |
|---|---|
| `length_of_stay` | Jail stay for the current charge: 0 = up to 7 days, 1 = 8 to 90 days, 2 = more than 90 days |
| `priors_count` after binning | 0 = none, 1 = one to three, 2 = more than three |
| `race` after encoding | 0 = African-American, 1 = Caucasian |
| `sex` after encoding | 0 = Female, 1 = Male |
| `age_cat` after encoding | 0 = under 25, 1 = 25 to 45, 2 = over 45 |

**Terms:**

| Term | Meaning |
|---|---|
| Protected attribute | A characteristic, here race, sex or age, that should not lead to unfair treatment |
| Decision threshold | The predicted probability above which a person is flagged as likely to re-offend |
| `gamma` | The share of accuracy the analyst accepts to lose in exchange for fairness (0.10 means up to 10%) |
| P-rule | Ratio of the rates at which two groups are flagged, smaller over larger, in percent; 100% is parity and 80% is a common target |
| Accuracy gap between races | Absolute difference between the accuracy for Caucasian and for African-American defendants (accuracy parity; this is not calibration in the usual sense) |
| Out-of-fold prediction | A prediction for a training row made by a model that did not see that row (5-fold cross-validation); used to choose the threshold without touching test rows |
| Mutual information | How much knowing one variable reduces uncertainty about another, measured in nats |
| Accuracy coefficient | How much a feature subset adds to predicting the outcome beyond the other features and the protected attribute |
| Discrimination coefficient | How much a feature subset carries information about the protected attribute that is also about the outcome |
| Shapley value | A fair split of a joint contribution among features, averaging over all orders in which features could be added |

## Limitations

- **Two racial groups only.** The race analysis covers African-American and Caucasian defendants and says nothing about the other groups in the raw data.
- **Resampling is not independent sampling.** The 20 splits reuse the same defendants, so ranges understate the uncertainty about new defendants. No confidence intervals are computed.
- **One model family and two fairness measures.** Only logistic regression is evaluated. Fairness is measured with the p-rule and an accuracy gap; equalized odds, calibration and other definitions are not evaluated and can rank the same interventions differently.
- **One place and time.** Broward County, Florida, around 2013 to 2014. The results should not be assumed to carry over elsewhere.
- **Descriptive, not causal.** The analysis shows which features carry which information, not what causes the disparity. Prior-offense counts may themselves reflect policing intensity.
- **Models do not use the protected attribute as an input** in Part 1, yet still show disparities, because other features act as proxies.
- **Rows dropped.** 235 of the 6,150 African-American and Caucasian defendants have no jail dates and are removed from the race and feature-selection analyses; the sex and age experiments use all 7,214 rows and do not use stay length.
- **Data quirks.** The `age` column does not match "screening date minus birth date" for most rows, most likely because it is age at the time of data collection (this reason was not verified). Two columns appear twice in the raw file (`priors_count`, `decile_score`) with identical values. The raw file contains defendants' names and birth dates (public record); the code never uses them.
- **Unverified.**
  - The raw data file was not compared with the original ProPublica release; row, column and group counts were only checked against memory of that release.
  - The paper's COMPAS table could not be reproduced exactly: it uses 5,334 rows with a filtering it does not describe, and its discrimination coefficients are about half the size of ours, although the ordering of the features is the same except that sex and charge degree swap places.
  - The optimizer behind the shared information is checked against textbook known answers (for example the AND gate) and a brute-force grid search, but not against an independent implementation of the decomposition.
  - Information estimated from counts is biased upward when groups are small; a permutation null is shown for one case but not applied everywhere.
  - Only the library versions listed below were tested.

## Reproduce

Tested with **Python 3.14.7 and 3.12.12** on an Apple-silicon Mac (macOS), with the exact library versions pinned in `requirements.txt` (numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, jupyter 1.1.1, nbconvert 7.17.2, ipykernel 7.4.0, pytest 9.2.0). Other versions were not tested.

Run these from the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest
jupyter nbconvert --to notebook --execute --inplace notebooks/fairness_diagnostics.ipynb
```

Measured on a fresh clone with an empty pip cache: installing the dependencies about 80 seconds (depends on the network), the 76 tests about 28 seconds, and executing the notebook about 92 seconds, roughly three and a half minutes in total. The executed notebook reproduces the committed outputs and figure exactly, and Python 3.14.7 and 3.12.12 give identical numbers. The last command rewrites the notebook file in place; `git checkout notebooks/fairness_diagnostics.ipynb` restores the committed version.

## Repository structure

```
.
├── data/
│   ├── README.md
│   └── compas-scores-two-years.csv     ProPublica COMPAS data, read-only input
├── figures/
│   ├── README.md
│   └── accuracy_prule_vs_gamma.png     generated by the notebook
├── notebooks/
│   └── fairness_diagnostics.ipynb      walkthrough that calls the package
├── src/compas_fairness/
│   ├── __init__.py
│   ├── baselines.py                    majority-class and COMPAS baselines
│   ├── config.py                       seed, target column, paths
│   ├── data.py                         loading the raw file
│   ├── feature_selection.py            feature analysis, Shapley scores, feature removal
│   ├── information.py                  mutual information, coefficients, Shapley values
│   ├── metrics.py                      p-rule and group accuracy gap
│   ├── partial_information.py          shared and unique information (optimization)
│   ├── plots.py                        the gamma sweep figure
│   ├── preprocessing.py                encoding, binning, dropping rows without dates
│   ├── seed_study.py                   results over 20 splits, cost to reach a p-rule
│   ├── threshold_experiments.py        race, sex and age threshold experiments
│   └── thresholds.py                   threshold search
├── tests/                              known-answer tests
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Data source

`data/compas-scores-two-years.csv` is the public COMPAS recidivism dataset released by ProPublica for their 2016 ["Machine Bias"](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) investigation. The threshold method is adapted from [Zafar et al., *Fairness Constraints*](https://github.com/mbilalzafar/fair-classification/tree/master/disparate_impact), and the feature analysis follows [Information Theoretic Measures for Fairness-aware Feature Selection](https://arxiv.org/abs/2106.00772).
