### Data folder

Read-only input data. Code never writes into this directory.

- `compas-scores-two-years.csv` – the ProPublica COMPAS recidivism dataset used by both analyses in [notebooks/fairness_diagnostics.ipynb](../notebooks/fairness_diagnostics.ipynb). It has 7,214 rows (one per defendant) and 53 columns. The file contains defendants' names and birth dates as published by ProPublica; the code never reads them. The columns that are used are listed in the glossary of the [main README](../README.md).
