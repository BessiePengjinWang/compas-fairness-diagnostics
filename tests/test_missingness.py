import pandas as pd
import pytest

from compas_fairness.data import load_compas
from compas_fairness.missingness import missing_jail_dates_by_race, reoffence_rate_by_missingness


def build_small_frame():
    rows = []
    for race, sex, age_cat, charge, jail_in, recid in [
        ("African-American", "Male", "25 - 45", "F", None, 0),
        ("African-American", "Male", "25 - 45", "F", "2013-01-01 00:00:00", 1),
        ("Caucasian", "Female", "25 - 45", "M", "2013-01-01 00:00:00", 1),
        ("Caucasian", "Female", "25 - 45", "M", "2013-01-01 00:00:00", 0),
        ("Hispanic", "Male", "25 - 45", "F", None, 1),
    ]:
        rows.append({"race": race, "sex": sex, "age_cat": age_cat, "c_charge_degree": charge, "c_jail_in": jail_in, "c_jail_out": jail_in, "two_year_recid": recid})
    return pd.DataFrame(rows)


def test_missing_jail_dates_by_race_known_answer_and_other_races_are_excluded():
    counts, p_value = missing_jail_dates_by_race(build_small_frame())
    assert counts.loc["African-American", "missing_dates"] == 1
    assert counts.loc["African-American", "share_missing"] == pytest.approx(0.5)
    assert counts.loc["Caucasian", "missing_dates"] == 0
    assert "Hispanic" not in counts.index
    assert 0 <= p_value <= 1


def test_reoffence_rate_by_missingness_known_answer():
    rates = reoffence_rate_by_missingness(build_small_frame())
    assert rates.loc["jail dates missing", "defendants"] == 1
    assert rates.loc["jail dates missing", "reoffence_rate"] == pytest.approx(0.0)
    assert rates.loc["jail dates present", "reoffence_rate"] == pytest.approx(2 / 3)


def test_real_data_has_235_missing_rows_among_the_two_races():
    counts, _ = missing_jail_dates_by_race(load_compas())
    assert counts["missing_dates"].sum() == 235
    assert counts[["missing_dates", "with_dates"]].to_numpy().sum() == 6150
