import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from compas_fairness.plots import plot_gamma_sweep


def test_gamma_sweep_figure_has_a_light_background_a_legend_and_is_saved(tmp_path):
    sweep = pd.DataFrame({"gamma": [0.0, 0.1, 0.2], "accuracy": [0.66, 0.60, 0.55], "p_rule": [58.0, 79.0, 90.0]})
    output_path = tmp_path / "sweep.png"
    figure = plot_gamma_sweep(sweep, 0.66, 58.0, 0.52, output_path)
    assert figure.get_facecolor() == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert all(axis.get_facecolor() == pytest.approx((1.0, 1.0, 1.0, 1.0)) for axis in figure.axes[:1])
    assert len(figure.axes[0].get_legend().get_texts()) == 6
    assert output_path.stat().st_size > 1000
