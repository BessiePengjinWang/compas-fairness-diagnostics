import pandas as pd
from scipy.stats import chi2_contingency

from compas_fairness.config import TARGET
from compas_fairness.preprocessing import encode_features

RACE_NAMES = {0: "African-American", 1: "Caucasian"}


def jail_dates_missing_mask(frame):
    return frame[["c_jail_in", "c_jail_out"]].isna().any(axis=1)


def missing_jail_dates_by_race(data):
    frame = encode_features(data)
    missing = jail_dates_missing_mask(frame)
    race_names = frame["race"].map(RACE_NAMES)
    counts = pd.crosstab(race_names, missing).rename(columns={False: "with_dates", True: "missing_dates"})
    counts["share_missing"] = counts["missing_dates"] / (counts["missing_dates"] + counts["with_dates"])
    p_value = chi2_contingency(pd.crosstab(race_names, missing), correction=False)[1]
    return counts, p_value


def reoffence_rate_by_missingness(data):
    frame = encode_features(data)
    missing = jail_dates_missing_mask(frame)
    rates = frame.groupby(missing)[TARGET].agg(["size", "mean"])
    rates.index = ["jail dates present", "jail dates missing"]
    return rates.rename(columns={"size": "defendants", "mean": "reoffence_rate"})
