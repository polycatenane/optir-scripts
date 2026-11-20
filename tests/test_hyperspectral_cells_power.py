import numpy as np
import pandas as pd
import pytest

from hyperspectral_cells import power


def write_scan(path, values):
    np.savetxt(path, np.asarray(values), delimiter=",")


def test_read_ac_power_skips_dc_and_sorts(tmp_path):
    write_scan(tmp_path / "20261111_120001_1612cm-1_scan_AC.csv", [[1, 3], [5, 7]])
    write_scan(tmp_path / "20261111_120000_1530cm-1_scan_AC.csv", [[2, 2], [2, 2]])
    write_scan(tmp_path / "20261111_120000_1530cm-1_scan_DC.csv", [[100, 100], [100, 100]])

    diagnostics = power.read_ac_power(tmp_path)

    assert diagnostics.attrs["date"] == "20261111"
    assert diagnostics["Wavenumber_cm-1"].tolist() == [1530.0, 1612.0]
    assert diagnostics["Averaged_Power_mV"].tolist() == [2.0, 4.0]
    assert diagnostics.loc[1, ["std", "min", "max", "pixel_count"]].tolist() == [
        pytest.approx(np.sqrt(5)),
        1.0,
        7.0,
        4,
    ]


def test_read_ac_power_rejects_multiple_dates(tmp_path):
    write_scan(tmp_path / "20261111_120000_1530cm-1_scan_AC.csv", [[1]])
    write_scan(tmp_path / "20261112_120000_1612cm-1_scan_AC.csv", [[1]])

    with pytest.raises(ValueError, match="one acquisition date"):
        power.read_ac_power(tmp_path)


def test_read_ac_power_rejects_duplicate_wavenumbers(tmp_path):
    write_scan(tmp_path / "20261111_120000_1530cm-1_scan_AC.csv", [[1, 1]])
    write_scan(tmp_path / "20261111_120001_1530cm-1_scan_AC.csv", [[2, 2]])

    with pytest.raises(ValueError, match="Duplicate AC power wavenumbers"):
        power.read_ac_power(tmp_path)


def test_write_power_uses_processed_power_format(tmp_path):
    diagnostics = pd.DataFrame(
        {"Wavenumber_cm-1": [1530.0], "Averaged_Power_mV": [2.5], "std": [0.1]}
    )
    diagnostics.attrs["date"] = "20261111"

    path = power.write_power(diagnostics, tmp_path / "Power data Processed")

    assert path.name == "power_20261111_processed.csv"
    assert path.read_text().splitlines() == ["Wavenumber_cm-1,Averaged_Power_mV", "1530.0,2.5"]


def test_low_pass_fft_removes_short_variations():
    x = np.arange(1000.0, 1100.0, 5.0)
    smooth = np.linspace(2.0, 4.0, x.size)
    diagnostics = pd.DataFrame(
        {"Wavenumber_cm-1": x, "Averaged_Power_mV": smooth + 0.5 * np.cos(np.pi * np.arange(x.size))}
    )
    diagnostics.attrs["date"] = "20261111"

    filtered = power.low_pass_fft(diagnostics, shortest_period_cm=20)

    assert filtered.attrs["date"] == "20261111"
    assert filtered["raw_mV"].tolist() == diagnostics["Averaged_Power_mV"].tolist()
    assert np.abs(filtered["Averaged_Power_mV"] - smooth).max() < np.abs(diagnostics["Averaged_Power_mV"] - smooth).max()


def test_low_pass_fft_requires_even_spacing():
    diagnostics = pd.DataFrame({"Wavenumber_cm-1": [1.0, 2.0, 4.0], "Averaged_Power_mV": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="evenly spaced"):
        power.low_pass_fft(diagnostics, 10)
