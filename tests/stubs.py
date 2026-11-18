import numpy as np


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return np.column_stack([1 - self.probabilities, self.probabilities])

    def predict(self, features):
        return (self.probabilities > 0.5).astype(int)
