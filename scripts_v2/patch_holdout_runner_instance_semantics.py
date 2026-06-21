#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runner = ROOT / "experiments_v2" / "run_coco_rc1_holdout_algorithm.py"
protocol = (
    ROOT
    / "protocols"
    / "route_b"
    / "final_rc1"
    / "COCO_HOLDOUT_EXECUTION_PROTOCOL.md"
)

text = runner.read_text()

old_tag = (
    'RUNNER_FREEZE_TAG = '
    '"route-b-v2.0.0-rc1-holdout-runner-freeze"'
)
new_tag = (
    'RUNNER_FREEZE_TAG = '
    '"route-b-v2.0.0-rc1-holdout-runner-freeze-v2"'
)
if old_tag not in text:
    raise RuntimeError("Old runner tag constant was not found.")
text = text.replace(old_tag, new_tag, 1)

old_suite = (
    '    suite = cocoex.Suite(\n'
    '        "bbob",\n'
    '        "",\n'
    '        (\n'
    '            "dimensions: 2,3,5,10,20 "\n'
    '            "function_indices: 1-24 "\n'
    '            "instance_indices: 4-15"\n'
    '        ),\n'
    '    )\n'
)

new_suite = (
    '    suite = cocoex.Suite(\n'
    '        "bbob",\n'
    '        "instances: 4-15",\n'
    '        "dimensions: 2,3,5,10,20 function_indices: 1-24",\n'
    '    )\n'
    '\n'
    '    # Validate actual IDs before attaching an observer or evaluating.\n'
    '    suite_ids = list(suite.ids())\n'
    '    parsed_ids = [parse_problem_id(pid) for pid in suite_ids]\n'
    '    observed_functions = sorted({f for f, _, _ in parsed_ids})\n'
    '    observed_instances = sorted({i for _, i, _ in parsed_ids})\n'
    '    observed_dimensions = sorted({d for _, _, d in parsed_ids})\n'
    '\n'
    '    suite_preflight = {\n'
    '        "status": "COCO_HOLDOUT_SUITE_PREFLIGHT_OK",\n'
    '        "suite_instance": "instances: 4-15",\n'
    '        "problem_count": len(suite_ids),\n'
    '        "functions": observed_functions,\n'
    '        "instances": observed_instances,\n'
    '        "dimensions": observed_dimensions,\n'
    '        "objective_evaluations_performed": 0,\n'
    '    }\n'
    '\n'
    '    if len(suite_ids) != 24 * 5 * 12:\n'
    '        raise RuntimeError(\n'
    '            f"Unexpected holdout problem count: {len(suite_ids)}"\n'
    '        )\n'
    '    if observed_functions != list(range(1, 25)):\n'
    '        raise RuntimeError(\n'
    '            f"Unexpected holdout functions: {observed_functions}"\n'
    '        )\n'
    '    if observed_instances != list(range(4, 16)):\n'
    '        raise RuntimeError(\n'
    '            f"Unexpected actual holdout instances: {observed_instances}"\n'
    '        )\n'
    '    if observed_dimensions != [2, 3, 5, 10, 20]:\n'
    '        raise RuntimeError(\n'
    '            f"Unexpected holdout dimensions: {observed_dimensions}"\n'
    '        )\n'
    '\n'
    '    (algorithm_root / "suite_preflight.json").write_text(\n'
    '        json.dumps(suite_preflight, indent=2)\n'
    '    )\n'
)

if old_suite not in text:
    raise RuntimeError("Old COCO Suite construction block was not found.")
text = text.replace(old_suite, new_suite, 1)
runner.write_text(text)

protocol_text = protocol.read_text()
protocol_text = protocol_text.replace(
    "`route-b-v2.0.0-rc1-holdout-runner-freeze`.",
    "`route-b-v2.0.0-rc1-holdout-runner-freeze-v2`.",
)
protocol_text += (
    "\n## COCO instance-construction semantics\n\n"
    "Actual BBOB instance identifiers 4-15 are supplied through the "
    "suite-instance argument:\n\n"
    '`cocoex.Suite("bbob", "instances: 4-15", suite_options)`.\n\n'
    "The `instance_indices` suite option is not used because it filters "
    "ordinal positions in an instantiated suite rather than actual instance "
    "numbers. A zero-evaluation preflight verifies 1,440 IDs with instances "
    "exactly 4-15 before an observer is attached.\n"
)
protocol.write_text(protocol_text)

print("HOLDOUT_RUNNER_INSTANCE_SEMANTICS_PATCH_OK")
print("runner:", runner)
print("protocol:", protocol)
