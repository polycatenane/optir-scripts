from pathlib import Path
import os

import pytest

from cleanup_duplicate import ScanValidationError, build_cleanup_plan, cleanup


def acquisition(directory: Path, timestamp: str, wavenumber: str, extra: str = "") -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f"_{extra}" if extra else ""
    ac = directory / f"{timestamp}_{wavenumber}cm-1{suffix}_AC.csv"
    dc = directory / f"{timestamp}_{wavenumber}cm-1{suffix}_DC.csv"
    ac.write_text("ac")
    dc.write_text("dc")
    return ac, dc


def test_discovers_arbitrary_depth_and_retains_newest_timestamp(tmp_path):
    directory = tmp_path / "any root" / "a" / "deeply" / "nested" / "directory"
    old_ac, old_dc = acquisition(directory, "20240101_010101", "1600.0", "scan")
    new_ac, new_dc = acquisition(directory, "20250101_010101", "1600", "scan")

    plan = build_cleanup_plan(tmp_path / "any root")

    assert plan.deletions[0].paths == (old_ac, old_dc)
    assert plan.retained[0].paths == (new_ac, new_dc)


def test_same_wavenumber_in_separate_directories_is_independent(tmp_path):
    first_old = acquisition(tmp_path / "one", "20240101_010101", "1500")
    first_new = acquisition(tmp_path / "one", "20250101_010101", "1500")
    second_old = acquisition(tmp_path / "two", "20230101_010101", "1500")
    second_new = acquisition(tmp_path / "two", "20260101_010101", "1500")

    plan = build_cleanup_plan(tmp_path)

    assert {item.paths for item in plan.deletions} == {first_old, second_old}
    assert {item.paths for item in plan.retained} == {first_new, second_new}


def test_filename_timestamp_not_modification_time_controls_retention(tmp_path):
    old = acquisition(tmp_path, "20240101_010101", "1700")
    new = acquisition(tmp_path, "20250101_010101", "1700")
    for path in old:
        os.utime(path, (2_000_000_000, 2_000_000_000))

    plan = build_cleanup_plan(tmp_path)

    assert plan.retained[0].paths == new
    assert plan.deletions[0].paths == old


def test_dry_run_then_deletion_mode_and_unrelated_files(tmp_path):
    old = acquisition(tmp_path / "fov", "20240101_010101", "1600")
    new = acquisition(tmp_path / "fov", "20250101_010101", "1600")
    unrelated = tmp_path / "fov" / "notes.csv"
    unrelated.write_text("keep")
    unique = acquisition(tmp_path / "fov", "20250101_010101", "1650")

    cleanup(tmp_path, dry_run=True)
    assert all(path.exists() for path in (*old, *new, *unique, unrelated))

    cleanup(tmp_path, dry_run=False)
    assert not any(path.exists() for path in old)
    assert all(path.exists() for path in (*new, *unique, unrelated))


def test_incomplete_duplicate_aborts_before_any_deletion(tmp_path):
    old = acquisition(tmp_path, "20240101_010101", "1600")
    incomplete_ac = tmp_path / "20250101_010101_1600cm-1_AC.csv"
    incomplete_ac.write_text("ac")

    with pytest.raises(ScanValidationError, match="Incomplete"):
        cleanup(tmp_path, dry_run=False)

    assert all(path.exists() for path in (*old, incomplete_ac))


def test_ambiguous_duplicate_aborts_before_any_deletion(tmp_path):
    old = acquisition(tmp_path, "20240101_010101", "1600")
    new = acquisition(tmp_path, "20250101_010101", "1600", "second")
    duplicate_ac = tmp_path / "20250101_010101_1600cm-1_duplicate_AC.csv"
    duplicate_ac.write_text("ac")

    with pytest.raises(ScanValidationError, match="Repeated AC"):
        cleanup(tmp_path, dry_run=False)
    assert all(path.exists() for path in (*old, *new, duplicate_ac))


def test_unparseable_matching_file_aborts_before_any_deletion(tmp_path):
    old = acquisition(tmp_path, "20240101_010101", "1600")
    new = acquisition(tmp_path, "20250101_010101", "1600")

    (tmp_path / "not-a-timestamp_1600cm-1_AC.csv").write_text("bad")
    with pytest.raises(ScanValidationError, match="Cannot parse"):
        cleanup(tmp_path, dry_run=False)
    assert all(path.exists() for path in (*old, *new))
