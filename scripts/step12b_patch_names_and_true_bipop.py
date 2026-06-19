from pathlib import Path
from datetime import datetime
import shutil


ROOT = Path.cwd()
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP = ROOT / "backups" / f"step12b_{STAMP}"
BACKUP.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = ROOT / "baselines" / "reference_optimizers.py"
RUNNER_FILE = ROOT / "experiments" / "run_coco.py"

for src in [BASELINE_FILE, RUNNER_FILE]:
    shutil.copy2(src, BACKUP / src.name)

# ---------------------------------------------------------------------
# Patch 1: replace the development restart wrapper with true pycma BIPOP
# ---------------------------------------------------------------------
text = BASELINE_FILE.read_text()

marker = "def optimize_bipop_cmaes("
start = text.find(marker)

if start < 0:
    raise RuntimeError("Cannot locate optimize_bipop_cmaes in reference_optimizers.py")

prefix = text[:start]

true_bipop = r'''def optimize_bipop_cmaes(objective, lb, ub, max_evals, seed=0):
    """
    BIPOP-CMA-ES using the official pycma restart interface.

    The BIPOP restart strategy is invoked through:
        cma.fmin2(..., restarts=9, bipop=True)

    Function evaluations are counted externally through BudgetedObjective.
    """
    import cma

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    dim = lb.size

    bo = BudgetedObjective(objective, lb, ub, int(max_evals))

    # A callable initial point generates a new random start at every restart,
    # as recommended for the pycma BIPOP interface.
    def x0_factory():
        return lb + rng.random(dim) * (ub - lb)

    sigma0 = max(1e-12, 0.25 * float(np.mean(ub - lb)))

    options = {
        "bounds": [lb.tolist(), ub.tolist()],
        "seed": int(seed),
        "maxfevals": int(max_evals),
        "verbose": -9,
        "verb_log": 0,
        "verb_disp": 0,
        "verb_time": 0,
    }

    try:
        cma.fmin2(
            bo,
            x0_factory,
            sigma0,
            options,
            restarts=9,
            bipop=True,
        )
    except BudgetExhausted:
        # The external objective wrapper prevents evaluations beyond budget.
        pass
    except Exception as exc:
        return _finish(
            bo,
            f"terminated_with_exception:{type(exc).__name__}"
        )

    message = "budget_exhausted" if bo.nfe >= int(max_evals) else "completed"
    return _finish(bo, message)
'''

BASELINE_FILE.write_text(prefix + true_bipop + "\n")

# ---------------------------------------------------------------------
# Patch 2: use formal short names and detailed COCO algorithm_info
# ---------------------------------------------------------------------
runner = RUNNER_FILE.read_text()

insertion_marker = '''ALL_CORE = [
    "basingraph",
    "cmaes",
    "bipop-cmaes",
    "de",
    "ms-lbfgsb",
    "random",
    "lhs",
]
'''

replacement = insertion_marker + r'''

# COCO requires short algorithm_name labels without spaces.
COCO_SHORT_NAMES = {
    "basingraph": "BasinGraph",
    "cmaes": "CMA_ES",
    "bipop-cmaes": "BIPOP_CMA_ES",
    "de": "DE",
    "ms-lbfgsb": "MS_LBFGSB",
    "random": "Random",
    "lhs": "LHS",
}

# Detailed provenance-oriented descriptions stored in official .info logs.
COCO_ALGORITHM_INFO = {
    "basingraph":
        "BasinGraph geometry-controlled basin-graph optimizer implemented in this work",
    "cmaes":
        "CMA-ES implemented with the pycma package under a strict COCO evaluation budget",
    "bipop-cmaes":
        "BIPOP-CMA-ES implemented with pycma fmin2 restarts=9 and bipop=True",
    "de":
        "Self-contained differential evolution reference implementation with fixed F and CR",
    "ms-lbfgsb":
        "Multi-start L-BFGS-B using scipy.optimize.minimize with fixed budget allocation",
    "random":
        "Uniform random search under the same function-evaluation budget",
    "lhs":
        "Latin hypercube sampling under the same function-evaluation budget",
}
'''

if "COCO_SHORT_NAMES = {" not in runner:
    if insertion_marker not in runner:
        raise RuntimeError("Cannot locate ALL_CORE block in experiments/run_coco.py")
    runner = runner.replace(insertion_marker, replacement)

old_observer = '''    observer = cocoex.Observer(
        "bbob",
        f"result_folder: {result_folder} algorithm_name: {display_name}"
    )
'''

new_observer = '''    coco_short_name = COCO_SHORT_NAMES[algorithm_id]
    algorithm_info = COCO_ALGORITHM_INFO[algorithm_id]

    observer_options = (
        f"result_folder: {result_folder} "
        f"algorithm_name: {coco_short_name} "
        f'algorithm_info: "{algorithm_info}"'
    )

    observer = cocoex.Observer("bbob", observer_options)
'''

if old_observer not in runner:
    raise RuntimeError("Cannot locate COCO observer block in experiments/run_coco.py")

runner = runner.replace(old_observer, new_observer)
RUNNER_FILE.write_text(runner)

print("STEP_12B_PATCH_OK")
print("Backup directory:", BACKUP)
print("Patched:", BASELINE_FILE)
print("Patched:", RUNNER_FILE)
