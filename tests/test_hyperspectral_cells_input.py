import numpy as np
import pandas as pd
import pytest

from hyperspectral_cells import input


def test_parse_metadata_and_group_details(tmp_path):
    path = tmp_path / "Control" / "2 hour" / "FOV3" / "20260825_1612cm-1_scan_AC.csv"
    path.parent.mkdir(parents=True)

    assert input.parse_wavenumber(path) == 1612.0
    assert input.parse_date(path) == "20260825"
    assert input.group_details(tmp_path, path.parent) == {
        "group": "Control / 2 hour",
        "condition": "Control",
        "time": "2 hour",
    }


def test_read_ac_cube_sorts_planes_and_rejects_inconsistent_shapes(tmp_path):
    high = tmp_path / "20260825_1612cm-1_scan_AC.csv"
    low = tmp_path / "20260825_1530cm-1_scan_AC.csv"
    np.savetxt(high, np.full((2, 2), 2), delimiter=",")
    np.savetxt(low, np.full((2, 2), 1), delimiter=",")

    cube, x = input.read_ac_cube([high, low])

    assert x.tolist() == [1530.0, 1612.0]
    assert cube[0, 0].tolist() == [1.0, 2.0]
    np.savetxt(tmp_path / "20260825_1550cm-1_scan_AC.csv", np.ones((3, 2)), delimiter=",")
    with pytest.raises(ValueError, match="Inconsistent AC plane shape"):
        input.read_ac_cube(list(tmp_path.glob("*_AC.csv")))


def test_scan_align_and_extract_label_spectra(tmp_path):
    fov = tmp_path / "Control" / "1 hour" / "FOV1"
    fov.mkdir(parents=True)
    for wavenumber in (1530, 1550, 1612):
        np.savetxt(fov / f"20260825_{wavenumber}cm-1_scan_AC.csv", np.ones((2, 2)), delimiter=",")
    records = input.scan_ac_fovs(tmp_path, ("Control",))
    assert records[0]["files"] == sorted(records[0]["files"], key=input.parse_wavenumber)

    x, corrected = input.align_to_ir_power(
        np.array([1500.0, 1530.0, 1550.0, 1612.0]),
        np.array([[2.0, 4.0, 6.0, 8.0]]),
        "20260825",
        {"20260825": (np.array([1530.0, 1550.0, 1612.0]), np.array([2.0, 3.0, 4.0]))},
    )
    assert x.tolist() == [1530.0, 1550.0, 1612.0]
    assert corrected.tolist() == [[2.0, 2.0, 2.0]]

    cube = np.arange(12, dtype=float).reshape(2, 2, 3)
    spectra, labels, counts = input.extract_label_spectra(
        cube, np.array([[0, 1], [1, 2]])
    )
    assert labels.tolist() == [1, 2]
    assert counts.tolist() == [2, 1]
    assert np.allclose(spectra, [[4.5, 5.5, 6.5], [9.0, 10.0, 11.0]])


def test_read_ir_profiles(tmp_path):
    profiles = tmp_path / "Power data Processed"
    profiles.mkdir()
    pd.DataFrame(
        {"Wavenumber_cm-1": [1612, 1530], "Averaged_Power_mV": [4, 2]}
    ).to_csv(profiles / "power_20260825_processed.csv", index=False)

    x, power = input.read_ir_profiles(tmp_path)["20260825"]
    assert x.tolist() == [1530.0, 1612.0]
    assert power.tolist() == [2.0, 4.0]
