from pathlib import Path
import shutil
import numpy as np
import cocoex

# ------------------------------------------------------------
# COCO official observer smoke test.
# COCO writes relative result folders under ./exdata/.
# ------------------------------------------------------------

RESULT_FOLDER = "step2_coco_smoke_results"
COCO_OUTPUT_DIR = Path("exdata") / RESULT_FOLDER

# Clean previous smoke-test output.
if COCO_OUTPUT_DIR.exists():
    shutil.rmtree(COCO_OUTPUT_DIR)

suite = cocoex.Suite(
    "bbob",
    "",
    "dimensions: 2 function_indices: 1 instance_indices: 1"
)

observer = cocoex.Observer(
    "bbob",
    f"result_folder: {RESULT_FOLDER} algorithm_name: SmokeRandom"
)

rng = np.random.default_rng(20260613)

for problem in suite:
    problem.observe_with(observer)

    dim = problem.dimension

    try:
        lb = np.asarray(problem.lower_bounds, dtype=float)
        ub = np.asarray(problem.upper_bounds, dtype=float)
    except Exception:
        lb = -5.0 * np.ones(dim)
        ub = 5.0 * np.ones(dim)

    max_evals = 50 * dim

    x = np.zeros(dim)
    best = problem(x)

    for _ in range(max_evals - 1):
        x = lb + rng.random(dim) * (ub - lb)
        y = problem(x)
        if y < best:
            best = y

# Force object cleanup before the shell checks files.
del observer
del suite

print("COCO observer smoke test finished.")
print("COCO output directory:", COCO_OUTPUT_DIR)
print("STEP_2_RUN_FINISHED")
