import pandas as pd

from compas_fairness.preprocessing import AGE_CATEGORY_MAPPING
from compas_fairness.threshold_experiments import (
    build_age_experiment,
    build_race_experiment,
    build_sex_experiment,
    fair_threshold_report,
    unconstrained_report,
)

DEFAULT_SEEDS = tuple(range(20))
RACE_GAMMAS = (0.10, 0.15, 0.20)
SEX_GAMMA = 0.20
AGE_GAMMA = 0.15


def part1_seed_table(data, seeds=DEFAULT_SEEDS):
    records = []
    for seed in seeds:
        record = {"seed": seed}

        race_experiment = build_race_experiment(data, seed)
        race_report = unconstrained_report(race_experiment, 0)
        record["race_unconstrained_accuracy"] = race_report["accuracy"]
        record["race_unconstrained_p_rule"] = race_report["p_rule"]
        for gamma in RACE_GAMMAS:
            report = fair_threshold_report(race_experiment, 0, gamma)
            record[f"race_gamma_{gamma:.2f}_accuracy"] = report["accuracy"]
            record[f"race_gamma_{gamma:.2f}_p_rule"] = report["p_rule"]

        sex_experiment = build_sex_experiment(data, seed)
        sex_report = unconstrained_report(sex_experiment, "Female")
        record["sex_unconstrained_accuracy"] = sex_report["accuracy"]
        record["sex_unconstrained_p_rule"] = sex_report["p_rule"]
        report = fair_threshold_report(sex_experiment, "Female", SEX_GAMMA, 101)
        record[f"sex_gamma_{SEX_GAMMA:.2f}_accuracy"] = report["accuracy"]
        record[f"sex_gamma_{SEX_GAMMA:.2f}_p_rule"] = report["p_rule"]

        age_experiment = build_age_experiment(data, seed)
        record["age_unconstrained_accuracy"] = unconstrained_report(age_experiment, 0)["accuracy"]
        for age_group in AGE_CATEGORY_MAPPING.values():
            record[f"age_group_{age_group}_unconstrained_p_rule"] = unconstrained_report(age_experiment, age_group)["p_rule"]
            report = fair_threshold_report(age_experiment, age_group, AGE_GAMMA, 101)
            record[f"age_group_{age_group}_gamma_{AGE_GAMMA:.2f}_accuracy"] = report["accuracy"]
            record[f"age_group_{age_group}_gamma_{AGE_GAMMA:.2f}_p_rule"] = report["p_rule"]

        records.append(record)
    return pd.DataFrame(records)


def summarize_over_seeds(seed_table):
    return seed_table.drop(columns="seed").agg(["mean", "std", "min", "max"]).T
