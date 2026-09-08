#!/usr/bin/env python3

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import re


DATA_ROOT = Path("/var/home/praxis/Documents/Bai/Data/20260828 Artificial Sweetener/2026-08-25 and 2026-08-26/")
DRY_RUN = False


FILENAME_PATTERN = re.compile(
    r"^(?P<timestamp>\d{8}_\d{6})(?:_.*)?_(?P<wavenumber>\d+(?:\.\d+)?)cm-1(?:_.*)?_(?P<channel>AC|DC)\.csv$"
)


@dataclass(frozen=True)
class AcquisitionFile:
    path: Path
    timestamp: datetime
    wavenumber: Decimal
    channel: str


@dataclass(frozen=True)
class Acquisition:
    parent: Path
    wavenumber: Decimal
    timestamp: datetime
    ac_path: Path
    dc_path: Path

    @property
    def paths(self) -> tuple[Path, Path]:
        return self.ac_path, self.dc_path


@dataclass(frozen=True)
class CleanupPlan:
    retained: tuple[Acquisition, ...]
    deletions: tuple[Acquisition, ...]
    unique: tuple[Acquisition, ...]
    matching_files: int


class ScanValidationError(ValueError):
    pass


def parse_acquisition_file(path: Path) -> AcquisitionFile:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ScanValidationError(f"Cannot parse matching acquisition filename: {path}")
    try:
        timestamp = datetime.strptime(match["timestamp"], "%Y%m%d_%H%M%S")
    except ValueError as error:
        raise ScanValidationError(f"Invalid acquisition timestamp in {path}") from error
    return AcquisitionFile(
        path=path,
        timestamp=timestamp,
        wavenumber=Decimal(match["wavenumber"]),
        channel=match["channel"],
    )


def matching_files(root: Path) -> list[Path]:
    return sorted([*root.rglob("*_AC.csv"), *root.rglob("*_DC.csv")])


def build_cleanup_plan(root: Path) -> CleanupPlan:
    files = matching_files(root)
    grouped: dict[tuple[Path, Decimal], dict[datetime, dict[str, Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for path in files:
        parsed = parse_acquisition_file(path)
        channels = grouped[(path.parent, parsed.wavenumber)][parsed.timestamp]
        if parsed.channel in channels:
            raise ScanValidationError(
                f"Repeated {parsed.channel} file for {path.parent}, "
                f"{parsed.wavenumber} cm-1, {parsed.timestamp:%Y%m%d_%H%M%S}"
            )
        channels[parsed.channel] = path

    retained = []
    deletions = []
    unique = []
    for (parent, wavenumber), timestamps in sorted(
        grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        if len(timestamps) == 1:
            timestamp, channels = next(iter(timestamps.items()))
            if set(channels) == {"AC", "DC"}:
                unique.append(Acquisition(parent, wavenumber, timestamp, channels["AC"], channels["DC"]))
            continue
        incomplete = [
            timestamp
            for timestamp, channels in timestamps.items()
            if set(channels) != {"AC", "DC"}
        ]
        if incomplete:
            labels = ", ".join(timestamp.strftime("%Y%m%d_%H%M%S") for timestamp in sorted(incomplete))
            raise ScanValidationError(
                f"Incomplete acquisition(s) for {parent}, {wavenumber} cm-1: {labels}"
            )
        acquisitions = sorted(
            (
                Acquisition(parent, wavenumber, timestamp, channels["AC"], channels["DC"])
                for timestamp, channels in timestamps.items()
            ),
            key=lambda acquisition: acquisition.timestamp,
        )
        retained.append(acquisitions[-1])
        deletions.extend(acquisitions[:-1])
    return CleanupPlan(
        retained=tuple(retained),
        deletions=tuple(deletions),
        unique=tuple(unique),
        matching_files=len(files),
    )


def relative_path(root: Path, path: Path) -> Path:
    return path.relative_to(root)


def print_cleanup_plan(root: Path, plan: CleanupPlan, dry_run: bool) -> None:
    print(f"Data root: {root}")
    print(f"Mode: {'dry run' if dry_run else 'delete'}")
    for acquisition in plan.retained:
        for path in acquisition.paths:
            print(f"Retain: {relative_path(root, path)}")
    for acquisition in plan.deletions:
        for path in acquisition.paths:
            print(f"Delete: {relative_path(root, path)}")
    print(f"Matching files: {plan.matching_files}")
    print(f"Duplicate groups: {len(plan.retained)}")
    print(f"Retained acquisitions: {len(plan.retained)}")
    print(f"Deletion candidates: {len(plan.deletions)} acquisitions, {len(plan.deletions) * 2} files")
    print(f"Skipped unique acquisitions: {len(plan.unique)}")


def cleanup(root: Path, dry_run: bool = True) -> CleanupPlan:
    plan = build_cleanup_plan(root)
    print_cleanup_plan(root, plan, dry_run)
    if not dry_run:
        for acquisition in plan.deletions:
            for path in acquisition.paths:
                path.unlink()
    return plan


def main() -> None:
    cleanup(DATA_ROOT, DRY_RUN)


if __name__ == "__main__":
    main()
