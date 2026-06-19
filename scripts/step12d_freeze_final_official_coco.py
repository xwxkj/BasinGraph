from __future__ import annotations

import csv
import hashlib
import platform
import re
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cma
import cocoex
import cocopp
import matplotlib
import numpy as np
import pandas as pd
import scipy


ROOT = Path.cwd()
RUN_ID_FILE = ROOT / "official_results" / "OFFICIAL_FINAL_CORE_RUN_ID.txt"

if not RUN_ID_FILE.exists():
    raise FileNotFoundError(
        "Missing official_results/OFFICIAL_FINAL_CORE_RUN_ID.txt"
    )

RUN_ID = RUN_ID_FILE.read_text().strip()

CSV_PATH = ROOT / "processed_results" / f"{RUN_ID}.csv"
RUN_LOG = ROOT / "logs" / f"{RUN_ID}_run.log"
VALIDATION_LOG = ROOT / "logs" / f"{RUN_ID}_validation.log"
FILE_COUNT_LOG = ROOT / "logs" / f"{RUN_ID}_file_count.txt"
LABEL_LOG = ROOT / "logs" / f"{RUN_ID}_algorithm_labels.txt"
COCOPP_LOG = ROOT / "logs" / f"{RUN_ID}_cocopp.log"
ENV_FILE = ROOT / "protocols" / f"{RUN_ID}_environment.txt"
LOCK_FILE = ROOT / "protocols" / f"{RUN_ID}_requirements-lock.txt"

required_files = [
    CSV_PATH,
    RUN_LOG,
    VALIDATION_LOG,
    FILE_COUNT_LOG,
    LABEL_LOG,
    COCOPP_LOG,
    ENV_FILE,
    LOCK_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file is missing: {file_path}")


# ---------------------------------------------------------------------
# 1. Validate the result table
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

expected_algorithms = {
    "BasinGraph",
    "CMA-ES",
    "BIPOP-CMA-ES",
    "Differential Evolution",
    "Multi-start L-BFGS-B",
    "Latin Hypercube Sampling",
    "Random Search",
}

expected_dimensions = [2, 3, 5, 10, 20]
expected_rows = 12600
expected_problems = 1800

assert len(df) == expected_rows, (
    f"Expected {expected_rows} result rows, found {len(df)}."
)

assert set(df["algorithm"].unique()) == expected_algorithms, (
    "The algorithm set does not match the final protocol."
)

assert sorted(df["dimension"].unique().tolist()) == expected_dimensions, (
    "The dimension set does not match the final protocol."
)

assert df["problem_id"].nunique() == expected_problems, (
    f"Expected {expected_problems} unique COCO problems, "
    f"found {df['problem_id'].nunique()}."
)

if df["message"].astype(str).str.contains(
    "exception", case=False, na=False
).any():
    raise RuntimeError("At least one result row reports an exception.")


# ---------------------------------------------------------------------
# 2. Validate budget usage
# ---------------------------------------------------------------------
budget_rows = []

for algorithm in sorted(expected_algorithms):
    sub = df[df["algorithm"] == algorithm].copy()
    ratio = sub["nfe"] / sub["budget"]

    budget_rows.append({
        "algorithm": algorithm,
        "rows": len(sub),
        "min_nfe_budget_ratio": float(ratio.min()),
        "median_nfe_budget_ratio": float(ratio.median()),
        "mean_nfe_budget_ratio": float(ratio.mean()),
        "max_nfe_budget_ratio": float(ratio.max()),
        "rows_below_99pct_budget": int((ratio < 0.99).sum()),
    })

budget_df = pd.DataFrame(budget_rows)

for algorithm in ["BasinGraph", "BIPOP-CMA-ES"]:
    row = budget_df[budget_df["algorithm"] == algorithm].iloc[0]
    assert row["min_nfe_budget_ratio"] >= 0.99, (
        f"{algorithm} has runs using less than 99% of the budget."
    )


# ---------------------------------------------------------------------
# 3. Validate COCO observer output
# ---------------------------------------------------------------------
exdata_dirs = sorted((ROOT / "exdata").glob(f"{RUN_ID}_*"))

assert len(exdata_dirs) == 7, (
    f"Expected 7 COCO algorithm folders, found {len(exdata_dirs)}."
)

info_files = []
dat_files = []

for directory in exdata_dirs:
    info_files.extend(directory.rglob("*.info"))
    dat_files.extend(directory.rglob("*.dat"))

assert len(info_files) == 168, (
    f"Expected 168 .info files, found {len(info_files)}."
)

assert len(dat_files) == 840, (
    f"Expected 840 .dat files, found {len(dat_files)}."
)


# ---------------------------------------------------------------------
# 4. Validate formal COCO labels
# ---------------------------------------------------------------------
label_text = LABEL_LOG.read_text(errors="ignore")

found_labels = set(
    re.findall(r"algId\s*=\s*'([^']+)'", label_text)
)

expected_labels = {
    "BasinGraph",
    "CMA_ES",
    "BIPOP_CMA_ES",
    "DE",
    "MS_LBFGSB",
    "LHS",
    "Random",
}

assert found_labels == expected_labels, (
    f"Formal COCO labels do not match.\n"
    f"Expected: {sorted(expected_labels)}\n"
    f"Found: {sorted(found_labels)}"
)


# ---------------------------------------------------------------------
# 5. Locate and validate cocopp output
# ---------------------------------------------------------------------
cocopp_text = COCOPP_LOG.read_text(errors="ignore")

assert "ALL done" in cocopp_text, (
    "The cocopp log does not contain 'ALL done'."
)

output_matches = re.findall(
    r"Output data written to folder\s+(.+)",
    cocopp_text
)

if not output_matches:
    raise RuntimeError("Could not locate cocopp output directory in log.")

COCOPP_DIR = Path(output_matches[-1].strip())

if not COCOPP_DIR.exists():
    candidate = ROOT / output_matches[-1].strip()
    if candidate.exists():
        COCOPP_DIR = candidate
    else:
        raise FileNotFoundError(
            f"cocopp output directory does not exist: {COCOPP_DIR}"
        )


# ---------------------------------------------------------------------
# 6. Create final archive directory
# ---------------------------------------------------------------------
ARCHIVE_DIR = ROOT / "official_results" / f"{RUN_ID}_FINAL"

if ARCHIVE_DIR.exists():
    shutil.rmtree(ARCHIVE_DIR)

ARCHIVE_DIR.mkdir(parents=True)
(ARCHIVE_DIR / "logs").mkdir()
(ARCHIVE_DIR / "protocols").mkdir()
(ARCHIVE_DIR / "source_snapshot").mkdir()
(ARCHIVE_DIR / "compressed_raw_data").mkdir()
(ARCHIVE_DIR / "tables").mkdir()


# ---------------------------------------------------------------------
# 7. Copy core result files
# ---------------------------------------------------------------------
shutil.copy2(CSV_PATH, ARCHIVE_DIR / "tables" / CSV_PATH.name)
budget_df.to_csv(
    ARCHIVE_DIR / "tables" / "final_budget_usage_summary.csv",
    index=False,
)

for src in [
    RUN_LOG,
    VALIDATION_LOG,
    FILE_COUNT_LOG,
    LABEL_LOG,
    COCOPP_LOG,
]:
    shutil.copy2(src, ARCHIVE_DIR / "logs" / src.name)

for src in [ENV_FILE, LOCK_FILE]:
    shutil.copy2(src, ARCHIVE_DIR / "protocols" / src.name)


# ---------------------------------------------------------------------
# 8. Freeze actual source files used in the final run
# ---------------------------------------------------------------------
source_files = [
    ROOT / "basingraph" / "__init__.py",
    ROOT / "basingraph" / "optimizer.py",
    ROOT / "baselines" / "__init__.py",
    ROOT / "baselines" / "reference_optimizers.py",
    ROOT / "experiments" / "__init__.py",
    ROOT / "experiments" / "run_coco.py",
    ROOT / "scripts" / "run_step12c_corrected_official_coco.sh",
    ROOT / "scripts" / "step12b_patch_names_and_true_bipop.py",
]

for src in source_files:
    if not src.exists():
        continue

    relative = src.relative_to(ROOT)
    destination = ARCHIVE_DIR / "source_snapshot" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, destination)


# ---------------------------------------------------------------------
# 9. Write formal baseline provenance table
# ---------------------------------------------------------------------
provenance_rows = [
    {
        "algorithm": "BasinGraph",
        "implementation_source": "This work",
        "software": "BasinGraph Python implementation",
        "version": "Final COCO core snapshot",
        "modification": "Not applicable",
        "parameter_policy": "Fixed manuscript parameters",
        "formal_coco_label": "BasinGraph",
    },
    {
        "algorithm": "CMA-ES",
        "implementation_source": "pycma",
        "software": "cma",
        "version": getattr(cma, "__version__", "unknown"),
        "modification": "Standardized objective and budget wrapper only",
        "parameter_policy": "pycma defaults with bounds and fixed seed",
        "formal_coco_label": "CMA_ES",
    },
    {
        "algorithm": "BIPOP-CMA-ES",
        "implementation_source": "pycma fmin2 interface",
        "software": "cma",
        "version": getattr(cma, "__version__", "unknown"),
        "modification": "Standardized objective and budget wrapper only",
        "parameter_policy": "restarts=9, bipop=True",
        "formal_coco_label": "BIPOP_CMA_ES",
    },
    {
        "algorithm": "Differential Evolution",
        "implementation_source": "Self-contained reference implementation",
        "software": "BasinGraph benchmark repository",
        "version": "Final COCO core snapshot",
        "modification": "None",
        "parameter_policy": "F=0.5, CR=0.9, fixed population policy",
        "formal_coco_label": "DE",
    },
    {
        "algorithm": "Multi-start L-BFGS-B",
        "implementation_source": "SciPy wrapper",
        "software": "scipy.optimize.minimize",
        "version": scipy.__version__,
        "modification": "Standardized multistart and budget wrapper",
        "parameter_policy": "Fixed multistart budget allocation",
        "formal_coco_label": "MS_LBFGSB",
    },
    {
        "algorithm": "Latin Hypercube Sampling",
        "implementation_source": "Self-contained implementation",
        "software": "BasinGraph benchmark repository",
        "version": "Final COCO core snapshot",
        "modification": "None",
        "parameter_policy": "Uniform LHS under identical FE budget",
        "formal_coco_label": "LHS",
    },
    {
        "algorithm": "Random Search",
        "implementation_source": "Self-contained implementation",
        "software": "BasinGraph benchmark repository",
        "version": "Final COCO core snapshot",
        "modification": "None",
        "parameter_policy": "Uniform random sampling under identical FE budget",
        "formal_coco_label": "Random",
    },
]

with open(
    ARCHIVE_DIR / "protocols" / "final_baseline_provenance.csv",
    "w",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(provenance_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(provenance_rows)


# ---------------------------------------------------------------------
# 10. Write machine-readable run metadata
# ---------------------------------------------------------------------
metadata = {
    "run_id": RUN_ID,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "suite": "official COCO/BBOB noiseless",
    "functions": "1-24",
    "dimensions": "2,3,5,10,20",
    "instances": "1-15",
    "budget": "1000d",
    "result_rows": len(df),
    "unique_problem_ids": df["problem_id"].nunique(),
    "algorithm_count": len(expected_algorithms),
    "info_files": len(info_files),
    "dat_files": len(dat_files),
    "cocopp_all_done": True,
    "python": sys.version.replace("\n", " "),
    "platform": platform.platform(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__,
    "pycma": getattr(cma, "__version__", "unknown"),
    "cocoex": getattr(cocoex, "__version__", "unknown"),
    "cocopp": getattr(cocopp, "__version__", "unknown"),
    "formal_alg_ids": ";".join(sorted(expected_labels)),
}

with open(
    ARCHIVE_DIR / "protocols" / "final_run_metadata.csv",
    "w",
    newline="",
) as file:
    writer = csv.DictWriter(file, fieldnames=metadata.keys())
    writer.writeheader()
    writer.writerow(metadata)


# ---------------------------------------------------------------------
# 11. Compress raw COCO observer data
# ---------------------------------------------------------------------
observer_archive = (
    ARCHIVE_DIR
    / "compressed_raw_data"
    / f"coco_observer_logs_{RUN_ID}.tar.gz"
)

with tarfile.open(observer_archive, "w:gz", compresslevel=6) as archive:
    for directory in exdata_dirs:
        archive.add(
            directory,
            arcname=Path("exdata") / directory.name,
        )


# ---------------------------------------------------------------------
# 12. Compress cocopp output
# ---------------------------------------------------------------------
cocopp_archive = (
    ARCHIVE_DIR
    / "compressed_raw_data"
    / f"cocopp_output_{RUN_ID}.tar.gz"
)

with tarfile.open(cocopp_archive, "w:gz", compresslevel=6) as archive:
    archive.add(
        COCOPP_DIR,
        arcname=Path("ppdata") / COCOPP_DIR.name,
    )


# ---------------------------------------------------------------------
# 13. Write README
# ---------------------------------------------------------------------
readme = f"""# BasinGraph final official COCO/BBOB core validation

## Run identity

- RUN_ID: `{RUN_ID}`
- Status: frozen final core COCO/BBOB validation
- Suite: official COCO/BBOB noiseless
- Functions: 1-24
- Dimensions: 2, 3, 5, 10 and 20
- Instances: 1-15
- Budget: 1000d function evaluations
- Algorithms: seven

## Integrity

- Result rows: {len(df)}
- Unique COCO problems: {df['problem_id'].nunique()}
- `.info` files: {len(info_files)}
- `.dat` files: {len(dat_files)}
- Formal COCO labels: {", ".join(sorted(expected_labels))}
- cocopp status: ALL done

## Corrected baseline status

- CMA-ES uses pycma {getattr(cma, "__version__", "unknown")}.
- BIPOP-CMA-ES uses `cma.fmin2` with `restarts=9` and `bipop=True`.
- Official COCO short labels are stored in the `.info` logs.
- BasinGraph and BIPOP-CMA-ES use at least 99% of the prescribed budget in every run.

## Contents

- `tables/`: final CSV and budget-usage summary.
- `logs/`: run, validation, COCO-label and cocopp logs.
- `protocols/`: environment, dependency lock, provenance and run metadata.
- `source_snapshot/`: source files used for this run.
- `compressed_raw_data/`: compressed COCO observer logs and cocopp outputs.

This archive supersedes the earlier development Step 9 COCO run.
"""

(ARCHIVE_DIR / "README.md").write_text(readme)


# ---------------------------------------------------------------------
# 14. Generate SHA-256 manifest
# ---------------------------------------------------------------------
manifest_path = ARCHIVE_DIR / "MANIFEST_SHA256.csv"
manifest_rows = []

for file_path in sorted(ARCHIVE_DIR.rglob("*")):
    if not file_path.is_file():
        continue
    if file_path == manifest_path:
        continue

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    manifest_rows.append({
        "relative_path": str(file_path.relative_to(ARCHIVE_DIR)),
        "sha256": digest.hexdigest(),
        "size_bytes": file_path.stat().st_size,
    })

with open(manifest_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["relative_path", "sha256", "size_bytes"],
    )
    writer.writeheader()
    writer.writerows(manifest_rows)


# ---------------------------------------------------------------------
# 15. Create final ZIP package
# ---------------------------------------------------------------------
zip_path = (
    ROOT
    / "official_results"
    / f"BasinGraph_{RUN_ID}_FINAL_official_COCO_core.zip"
)

if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
) as archive:
    for file_path in sorted(ARCHIVE_DIR.rglob("*")):
        if file_path.is_file():
            archive.write(
                file_path,
                file_path.relative_to(ARCHIVE_DIR.parent),
            )


print("STEP_12D_OK")
print("RUN_ID:", RUN_ID)
print("Archive directory:", ARCHIVE_DIR)
print("Final ZIP:", zip_path)
print("Result rows:", len(df))
print("Unique problem IDs:", df["problem_id"].nunique())
print(".info files:", len(info_files))
print(".dat files:", len(dat_files))
print("Formal COCO labels:", sorted(found_labels))
print("cocopp directory:", COCOPP_DIR)
print("Observer archive size:", observer_archive.stat().st_size)
print("cocopp archive size:", cocopp_archive.stat().st_size)
