#!/usr/bin/env python3
"""Package combined rc1/rc2 development cocopp and comparison evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc2-run-id", required=True)
    parser.add_argument("--cocopp-output", required=True)
    parser.add_argument(
        "--rc1-result-root",
        default="results_v2/formal_development/coco_rc1",
    )
    parser.add_argument(
        "--rc2-result-root",
        default="results_v2/formal_development/coco_rc2",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()

    rc1_root = ROOT / args.rc1_result_root
    rc1_run_id = (rc1_root / "LAST_RUN_ID.txt").read_text().strip()
    rc1_run = rc1_root / rc1_run_id
    rc2_run = ROOT / args.rc2_result_root / args.rc2_run_id
    cocopp = Path(args.cocopp_output).expanduser().resolve()

    package_root = (
        ROOT
        / "results_v2"
        / "rc1_rc2_review_package"
        / args.rc2_run_id
    )
    if package_root.exists():
        shutil.rmtree(package_root)

    (package_root / "cocopp").mkdir(parents=True)
    (package_root / "rc1_core").mkdir()
    (package_root / "rc2_core").mkdir()
    (package_root / "protocol").mkdir()

    shutil.copytree(
        cocopp,
        package_root / "cocopp",
        dirs_exist_ok=True,
    )

    for name in [
        "validation_report.json",
        "run_metadata.json",
        "coco_v2_development_algorithm_summary.csv",
        "coco_v2_development_dimension_summary.csv",
        "coco_v2_development_raw_results.csv",
    ]:
        source = rc1_run / name
        if source.exists():
            shutil.copy2(source, package_root / "rc1_core" / name)

    for name in [
        "validation_report.json",
        "run_metadata.json",
        "coco_rc2_development_raw_results.csv",
        "BasinGraph_rc2_detail_manifest.csv",
    ]:
        source = rc2_run / name
        if source.exists():
            shutil.copy2(source, package_root / "rc2_core" / name)

    comparison = rc2_run / "paired_rc1_rc2_comparison"
    shutil.copytree(
        comparison,
        package_root / "paired_comparison",
        dirs_exist_ok=True,
    )

    for relative in [
        "protocols/route_b/rc2_revision/"
        "RC1_DEVELOPMENT_EVIDENCE_SNAPSHOT.json",
        "protocols/route_b/rc2_revision/"
        "RC2_DEVELOPMENT_ACCEPTANCE_GATE.json",
        "protocols/route_b/rc2_revision/"
        "RC2_FROZEN_REVISION_SPEC.md",
        "protocols/route_b/rc2_revision/"
        "RC2_MACHINE_SPEC.json",
        "protocols/route_b/formal_v2_protocol/"
        "V2_FORMAL_EXPERIMENT_PROTOCOL.md",
    ]:
        source = ROOT / relative
        shutil.copy2(source, package_root / "protocol" / source.name)

    readme = f"""# rc1 versus rc2 COCO development review

rc1 run: `{rc1_run_id}`
rc2 run: `{args.rc2_run_id}`

This package contains COCO instances 1-3 only.

Review the combined official cocopp output in `cocopp/index.html` and apply
the frozen acceptance gate. Prospective COCO instances 4-15 and the
prospective CUTEst holdout remain unopened.
"""
    (package_root / "README_REVIEW.md").write_text(readme)

    manifest_path = package_root / "MANIFEST_SHA256.csv"
    rows = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(package_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    archive_path = (
        ROOT
        / "results_v2"
        / f"BasinGraph_rc1_vs_rc2_COCO_development_review_{args.rc2_run_id}.tar.gz"
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(
            package_root,
            arcname=args.rc2_run_id,
        )

    print("RC1_RC2_REVIEW_PACKAGE_OK")
    print("package root:", package_root)
    print("archive:", archive_path)
    print("files:", len(rows))


if __name__ == "__main__":
    main()
