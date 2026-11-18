import pandas as pd

from compas_fairness.baselines import baseline_report
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


def add_baselines(record, prefix, report):
    by_method = report.set_index("method")
    record[f"{prefix}_majority_class_accuracy"] = by_method.loc["always predict the majority class", "accuracy"]
    record[f"{prefix}_compas_accuracy"] = by_method.loc["COMPAS decile score >= 5", "accuracy"]
    record[f"{prefix}_compas_p_rule"] = by_method.loc["COMPAS decile score >= 5", "p_rule"]
    record[f"{prefix}_compas_auc"] = by_method.loc["COMPAS decile score >= 5", "auc"]
    record[f"{prefix}_model_auc"] = by_method.loc["logistic regression (project model)", "auc"]


def part1_seed_table(data, seeds=DEFAULT_SEEDS):
    records = []
    for seed in seeds:
        record = {"seed": seed}

        race_experiment = build_race_experiment(data, seed)
        race_report = unconstrained_report(race_experiment, 0)
        record["race_unconstrained_accuracy"] = race_report["accuracy"]
        record["race_unconstrained_p_rule"] = race_report["p_rule"]
        add_baselines(record, "race", baseline_report(race_experiment, data, 0))
        for gamma in RACE_GAMMAS:
            report = fair_threshold_report(race_experiment, 0, gamma)
            record[f"race_gamma_{gamma:.2f}_accuracy"] = report["accuracy"]
            record[f"race_gamma_{gamma:.2f}_p_rule"] = report["p_rule"]

        sex_experiment = build_sex_experiment(data, seed)
        sex_report = unconstrained_report(sex_experiment, "Female")
        record["sex_unconstrained_accuracy"] = sex_report["accuracy"]
        record["sex_unconstrained_p_rule"] = sex_report["p_rule"]
        add_baselines(record, "sex", baseline_report(sex_experiment, data, "Female"))
        report = fair_threshold_report(sex_experiment, "Female", SEX_GAMMA, 101)
        record[f"sex_gamma_{SEX_GAMMA:.2f}_accuracy"] = report["accuracy"]
        record[f"sex_gamma_{SEX_GAMMA:.2f}_p_rule"] = report["p_rule"]

        age_experiment = build_age_experiment(data, seed)
        record["age_unconstrained_accuracy"] = unconstrained_report(age_experiment, 0)["accuracy"]
        add_baselines(record, "age", baseline_report(age_experiment, data, 0))
        for age_group in AGE_CATEGORY_MAPPING.values():
            record[f"age_group_{age_group}_unconstrained_p_rule"] = unconstrained_report(age_experiment, age_group)["p_rule"]
            report = fair_threshold_report(age_experiment, age_group, AGE_GAMMA, 101)
            record[f"age_group_{age_group}_gamma_{AGE_GAMMA:.2f}_accuracy"] = report["accuracy"]
            record[f"age_group_{age_group}_gamma_{AGE_GAMMA:.2f}_p_rule"] = report["p_rule"]

        records.append(record)
    return pd.DataFrame(records)


def summarize_over_seeds(seed_table):
    return seed_table.drop(columns="seed").agg(["mean", "std", "min", "max"]).T


TARGET_P_RULE = 80.0
GAMMA_GRID = tuple(round(0.01 * step, 2) for step in range(31))


def accuracy_points_lost_to_reach(experiment, group_value, n_thresholds, target=TARGET_P_RULE):
    baseline = unconstrained_report(experiment, group_value)
    if baseline["p_rule"] >= target:
        return 0.0
    for gamma in GAMMA_GRID:
        report = fair_threshold_report(experiment, group_value, gamma, n_thresholds)
        if report["p_rule"] >= target:
            return (baseline["accuracy"] - report["accuracy"]) * 100
    return float("nan")


def part1_cost_table(data, seeds=DEFAULT_SEEDS, target=TARGET_P_RULE):
    records = []
    for seed in seeds:
        age_experiment = build_age_experiment(data, seed)
        record = {
            "seed": seed,
            "race": accuracy_points_lost_to_reach(build_race_experiment(data, seed), 0, 1001, target),
            "sex": accuracy_points_lost_to_reach(build_sex_experiment(data, seed), "Female", 101, target),
        }
        for age_group in AGE_CATEGORY_MAPPING.values():
            record[f"age_group_{age_group}"] = accuracy_points_lost_to_reach(age_experiment, age_group, 101, target)
        records.append(record)
    return pd.DataFrame(records)
