import pandas as pd

RACE_MAPPING = {"African-American": 0, "Caucasian": 1}
SEX_MAPPING = {"Male": 1, "Female": 0}
AGE_CATEGORY_MAPPING = {"Less than 25": 0, "25 - 45": 1, "Greater than 45": 2}
CHARGE_DEGREE_MAPPING = {"F": 0, "M": 1}


def encode_features(df):
    df = df[df["race"].isin(RACE_MAPPING)].copy()
    df["race"] = df["race"].map(RACE_MAPPING)
    df["sex"] = df["sex"].map(SEX_MAPPING)
    df["age_cat"] = df["age_cat"].map(AGE_CATEGORY_MAPPING)
    df["c_charge_degree"] = df["c_charge_degree"].map(CHARGE_DEGREE_MAPPING)
    return df


def drop_rows_missing_jail_dates(df):
    return df.dropna(subset=["c_jail_in", "c_jail_out"])


def add_length_of_stay(df):
    if df[["c_jail_in", "c_jail_out"]].isna().any().any():
        raise ValueError("c_jail_in and c_jail_out must not be missing; drop those rows first")
    df = df.copy()
    jail_in = pd.to_datetime(df["c_jail_in"])
    jail_out = pd.to_datetime(df["c_jail_out"])
    stay_days = (jail_out - jail_in).dt.days
    df["length_of_stay"] = stay_days.apply(lambda d: 0 if d <= 7 else (2 if d > 90 else 1))
    return df


def bin_priors_count(df):
    df = df.copy()
    df["priors_count"] = df["priors_count"].apply(lambda c: 0 if c == 0 else (2 if c > 3 else 1))
    return df
