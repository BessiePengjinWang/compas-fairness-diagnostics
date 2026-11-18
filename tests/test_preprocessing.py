import pandas as pd

from compas_fairness.preprocessing import add_length_of_stay, bin_priors_count, encode_features


def test_bin_priors_count_boundaries():
    frame = pd.DataFrame({"priors_count": [0, 1, 3, 4, 10]})
    assert bin_priors_count(frame)["priors_count"].tolist() == [0, 1, 1, 2, 2]


def test_length_of_stay_bin_boundaries():
    jail_in = pd.Timestamp("2013-01-01 00:00:00")
    stay_days = [0, 7, 8, 90, 91]
    frame = pd.DataFrame({
        "c_jail_in": [str(jail_in)] * len(stay_days),
        "c_jail_out": [str(jail_in + pd.Timedelta(days=days)) for days in stay_days],
    })
    assert add_length_of_stay(frame)["length_of_stay"].tolist() == [0, 0, 1, 1, 2]


def test_encode_features_keeps_two_races_and_maps_labels():
    frame = pd.DataFrame({
        "race": ["African-American", "Caucasian", "Hispanic"],
        "sex": ["Male", "Female", "Male"],
        "age_cat": ["Less than 25", "25 - 45", "Greater than 45"],
        "c_charge_degree": ["F", "M", "F"],
    })
    encoded = encode_features(frame)
    assert len(encoded) == 2
    assert encoded["race"].tolist() == [0, 1]
    assert encoded["sex"].tolist() == [1, 0]
    assert encoded["age_cat"].tolist() == [0, 1]
    assert encoded["c_charge_degree"].tolist() == [0, 1]
    assert len(frame) == 3
