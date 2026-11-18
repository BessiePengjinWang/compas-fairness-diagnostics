import pandas as pd


def encode_features(df):
    race_mapping = {"African-American": 0, "Caucasian": 1}
    sex_mapping = {"Male": 1, "Female": 0}
    age_cat_mapping = {"Less than 25": 0, "25 - 45": 1, "Greater than 45": 2}
    charge_degree_mapping = {"F": 0, "M": 1}

    df = df[df["race"].isin(race_mapping)].copy()
    df["race"] = df["race"].map(race_mapping)
    df["sex"] = df["sex"].map(sex_mapping)
    df["age_cat"] = df["age_cat"].map(age_cat_mapping)
    df["c_charge_degree"] = df["c_charge_degree"].map(charge_degree_mapping)
    return df


def add_length_of_stay(df):
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
