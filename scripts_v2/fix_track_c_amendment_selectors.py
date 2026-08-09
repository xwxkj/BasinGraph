#!/usr/bin/env python3
"""Fix exact selectors in the one-time Track C amendment script."""

from pathlib import Path


path = Path(__file__).resolve().parent / "amend_track_c_preconfirmatory.py"
text = path.read_text(encoding="utf-8")

old_nist = '''    replace_once(\n        "evidence_extension_v1/track_c/nist.py",\n        "    start1 = np.asarray(spec.start1, dtype=float)\\n",\n        "    start1 = np.asarray(spec.start1, dtype=float)\\n"\n        "    start2 = np.asarray(spec.start2, dtype=float)\\n",\n    )\n'''
new_nist = '''    replace_once(\n        "evidence_extension_v1/track_c/nist.py",\n        "    certified = np.asarray(spec.certified, dtype=float)\\n"\n        "    start1 = np.asarray(spec.start1, dtype=float)\\n",\n        "    certified = np.asarray(spec.certified, dtype=float)\\n"\n        "    start1 = np.asarray(spec.start1, dtype=float)\\n"\n        "    start2 = np.asarray(spec.start2, dtype=float)\\n",\n    )\n'''

old_smoke = '''        '''  python evidence_extension_v1/track_c/finalize_track_c.py \\\\\n    --mode smoke \\\\\n    --run-id "${RUN_ID}"\\n''',\n        '''  python evidence_extension_v1/track_c/task_specific_references.py \\\\\n    --mode smoke \\\\\n    --output "results_b21/track_c/${RUN_ID}/task_specific_reference_results.csv"\\n  python evidence_extension_v1/track_c/finalize_track_c.py \\\\\n    --mode smoke \\\\\n    --run-id "${RUN_ID}"\\n''',\n'''
new_smoke = '''        '''  python evidence_extension_v1/track_c/finalize_track_c.py \\\\\n    --mode smoke --run-id "${RUN_ID}"\\n''',\n        '''  python evidence_extension_v1/track_c/task_specific_references.py \\\\\n    --mode smoke \\\\\n    --output "results_b21/track_c/${RUN_ID}/task_specific_reference_results.csv"\\n  python evidence_extension_v1/track_c/finalize_track_c.py \\\\\n    --mode smoke --run-id "${RUN_ID}"\\n''',\n'''

old_confirmatory = '''        '''  python evidence_extension_v1/track_c/finalize_track_c.py \\\\\n    --mode confirmatory \\\\\n    --run-id "${RUN_ID}" \\\\\n    --authorize-confirmatory\\n''',\n        '''  python evidence_extension_v1/track_c/task_specific_references.py \\\\\n    --mode confirmatory \\\\\n    --authorize-confirmatory \\\\\n    --output "results_b21/track_c/${RUN_ID}/task_specific_reference_results.csv"\\n  python evidence_extension_v1/track_c/finalize_track_c.py \\\\\n    --mode confirmatory \\\\\n    --run-id "${RUN_ID}" \\\\\n    --authorize-confirmatory\\n''',\n'''
new_confirmatory = '''        '''  python evidence_extension_v1/track_c/finalize_track_c.py \\\\\n    --mode confirmatory --run-id "${RUN_ID}" --authorize-confirmatory\\n''',\n        '''  python evidence_extension_v1/track_c/task_specific_references.py \\\\\n    --mode confirmatory \\\\\n    --authorize-confirmatory \\\\\n    --output "results_b21/track_c/${RUN_ID}/task_specific_reference_results.csv"\\n  python evidence_extension_v1/track_c/finalize_track_c.py \\\\\n    --mode confirmatory --run-id "${RUN_ID}" --authorize-confirmatory\\n''',\n'''

for name, old, new in [
    ("NIST selector", old_nist, new_nist),
    ("smoke execution selector", old_smoke, new_smoke),
    ("confirmatory execution selector", old_confirmatory, new_confirmatory),
]:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected one {name}, found {count}.")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("TRACK_C_AMENDMENT_SELECTORS_FIXED")
