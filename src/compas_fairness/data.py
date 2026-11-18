import pandas as pd

from compas_fairness.config import DEFAULT_DATA_PATH


def load_compas(path=DEFAULT_DATA_PATH):
    return pd.read_csv(path)
