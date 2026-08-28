import numpy as np
import pandas as pd

from hyperspectral_cells import fitting


def test_gaussian_and_baseline_shapes():
    x = np.linspace(1200, 1700, 101)
    y = 2 + fitting.gaussian(x, 3, 1530, 20)

    assert fitting.gaussian(np.array([1530.0]), 3, 1530, 20).tolist() == [3.0]
    assert fitting.estimate_polynomial_baseline(x, y).shape == x.shape


def test_fit_single_spectrum_handles_empty_forced_peak_range():
    x = np.linspace(1000, 1100, 21)
    result = fitting.fit_single_spectrum(x, np.ones_like(x))

    assert result["components"].shape == (0, x.size)
    assert result["peaks"].empty


def test_fit_single_and_group_spectra_return_expected_shapes_and_columns():
    x = np.linspace(1200, 1700, 251)
    spectrum = 0.1 + fitting.gaussian(x, 1, 1530, 20) + fitting.gaussian(x, 0.8, 1612, 20)
    result = fitting.fit_single_spectrum(x, spectrum)

    assert {"forced_center_cm-1", "peak_height", "area"}.issubset(result["peaks"].columns)
    metadata = pd.DataFrame({"cell_uid": ["one", "two"], "group": ["Control / 1 hour"] * 2})
    grouped = fitting.fit_group_spectra(x, np.vstack([spectrum, spectrum]), metadata)
    assert grouped["fit_matrix"].shape == (2, x.size)
    assert grouped["baseline_matrix"].shape == (2, x.size)
    assert len(grouped["fit_summary"]) == 2
