#!/usr/bin/env python3
"""Apply registered Track C amendments before confirmatory access."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{relative}: expected exactly one replacement, found {count}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def amend_tasks() -> None:
    replace_once(
        "evidence_extension_v1/track_c/tasks.py",
        "    metadata: dict[str, Any]\n\n    @property\n",
        "    metadata: dict[str, Any]\n"
        "    specialized_reference: Callable[[], dict[str, Any]] | None = None\n\n"
        "    @property\n",
    )

    phase_old = '''    base = np.zeros(d)\n    f_ref, f_base = _safe_reference(objective, truth, base)\n    family = "noisy_phase_retrieval" if noisy else "phase_retrieval"\n    return ScientificTask(\n        "c1", family, instance, d, 200,\n        -2.0 * np.ones(d), 2.0 * np.ones(d), objective, metrics,\n        f_ref, f_base, truth, {"measurements": m, "noise_sigma": sigma},\n    )\n'''
    phase_new = '''    base = np.zeros(d)\n    f_ref, f_base = _safe_reference(objective, truth, base)\n    family = "noisy_phase_retrieval" if noisy else "phase_retrieval"\n\n    def specialized_reference() -> dict[str, Any]:\n        covariance = matrix.T @ (measurements[:, None] * matrix) / m\n        _, eigenvectors = np.linalg.eigh(covariance)\n        norm_estimate = math.sqrt(\n            max(float(np.mean(measurements)) * d, 1e-12)\n        )\n        vector = np.clip(\n            norm_estimate * eigenvectors[:, -1],\n            -2.0,\n            2.0,\n        )\n        current = objective(vector)\n        best = vector.copy()\n        best_value = current\n        evaluations = 1\n        step = 0.25\n        iterations = 0\n        for iterations in range(1, 501):\n            projected = matrix @ vector\n            gradient = (4.0 / (m * scale)) * matrix.T @ (\n                (projected * projected - measurements) * projected\n            )\n            norm_sq = float(np.dot(gradient, gradient))\n            if not np.isfinite(norm_sq) or norm_sq <= 1e-24:\n                break\n            local_step = step\n            accepted = False\n            candidate = vector\n            candidate_value = current\n            for _ in range(24):\n                candidate = np.clip(\n                    vector - local_step * gradient,\n                    -2.0,\n                    2.0,\n                )\n                candidate_value = objective(candidate)\n                evaluations += 1\n                if (\n                    candidate_value\n                    <= current - 1e-4 * local_step * norm_sq\n                    or local_step <= 1e-12\n                ):\n                    accepted = True\n                    break\n                local_step *= 0.5\n            if not accepted:\n                break\n            if candidate_value < best_value:\n                best_value = candidate_value\n                best = candidate.copy()\n            change = float(np.linalg.norm(candidate - vector))\n            vector = candidate\n            current = candidate_value\n            step = min(1.0, local_step * 1.25)\n            if change <= 1e-10 * max(\n                1.0,\n                float(np.linalg.norm(vector)),\n            ):\n                break\n        return {\n            "method": "spectral_wirtinger_flow",\n            "xbest": best,\n            "fbest": best_value,\n            "iterations": iterations,\n            "objective_evaluations": evaluations,\n            "metadata": {\n                "spectral_initialization": True,\n                "backtracking": True,\n            },\n        }\n\n    return ScientificTask(\n        "c1", family, instance, d, 200,\n        -2.0 * np.ones(d), 2.0 * np.ones(d), objective, metrics,\n        f_ref, f_base, truth, {"measurements": m, "noise_sigma": sigma},\n        specialized_reference,\n    )\n'''
    replace_once(
        "evidence_extension_v1/track_c/tasks.py",
        phase_old,
        phase_new,
    )

    factor_old = '''    base = np.zeros(d)\n    f_ref, f_base = _safe_reference(objective, truth, base)\n    family = "large_matrix_factorization" if large else "matrix_factorization"\n    budget_multiplier = 75 if large else 150\n    return ScientificTask(\n        "c1", family, instance, d, budget_multiplier,\n        -3.0 * np.ones(d), 3.0 * np.ones(d), objective, metrics,\n        f_ref, f_base, truth,\n        {"matrix_shape": [m, n], "rank": rank, "sample_count": sample_count},\n    )\n'''
    factor_new = '''    base = np.zeros(d)\n    f_ref, f_base = _safe_reference(objective, truth, base)\n    family = "large_matrix_factorization" if large else "matrix_factorization"\n    budget_multiplier = 75 if large else 150\n\n    def specialized_reference() -> dict[str, Any]:\n        reference_rng = np.random.default_rng(\n            260_000 + instance + (10_000 if large else 0)\n        )\n        best_vector = base.copy()\n        best_value = objective(best_vector)\n        evaluations = 1\n        total_iterations = 0\n        ridge = 1e-8\n        row_observations = [\n            np.flatnonzero(rows == row) for row in range(m)\n        ]\n        col_observations = [\n            np.flatnonzero(cols == col) for col in range(n)\n        ]\n        for _restart in range(3):\n            u = reference_rng.normal(0.0, 0.25, (m, rank))\n            v = reference_rng.normal(0.0, 0.25, (n, rank))\n            for _ in range(80):\n                total_iterations += 1\n                for row_index, locations in enumerate(row_observations):\n                    if len(locations):\n                        design = v[cols[locations]]\n                        u[row_index] = np.linalg.solve(\n                            design.T @ design + ridge * np.eye(rank),\n                            design.T @ observations[locations],\n                        )\n                for col_index, locations in enumerate(col_observations):\n                    if len(locations):\n                        design = u[rows[locations]]\n                        v[col_index] = np.linalg.solve(\n                            design.T @ design + ridge * np.eye(rank),\n                            design.T @ observations[locations],\n                        )\n                u_norm = max(float(np.linalg.norm(u)), 1e-12)\n                v_norm = max(float(np.linalg.norm(v)), 1e-12)\n                balance_scale = math.sqrt(v_norm / u_norm)\n                u *= balance_scale\n                v /= balance_scale\n                vector = np.clip(\n                    np.concatenate([u.ravel(), v.ravel()]),\n                    -3.0,\n                    3.0,\n                )\n                value = objective(vector)\n                evaluations += 1\n                if value < best_value:\n                    best_value = value\n                    best_vector = vector.copy()\n            if best_value <= 1e-14:\n                break\n        return {\n            "method": "alternating_ridge_least_squares",\n            "xbest": best_vector,\n            "fbest": best_value,\n            "iterations": total_iterations,\n            "objective_evaluations": evaluations,\n            "metadata": {"restarts": 3, "ridge": ridge},\n        }\n\n    return ScientificTask(\n        "c1", family, instance, d, budget_multiplier,\n        -3.0 * np.ones(d), 3.0 * np.ones(d), objective, metrics,\n        f_ref, f_base, truth,\n        {"matrix_shape": [m, n], "rank": rank, "sample_count": sample_count},\n        specialized_reference,\n    )\n'''
    replace_once(
        "evidence_extension_v1/track_c/tasks.py",
        factor_old,
        factor_new,
    )


def amend_nist() -> None:
    replace_once(
        "evidence_extension_v1/track_c/nist.py",
        "import numpy as np\n\nfrom .common",
        "import numpy as np\nfrom scipy.optimize import least_squares\n\nfrom .common",
    )
    replace_once(
        "evidence_extension_v1/track_c/nist.py",
        "    start1 = np.asarray(spec.start1, dtype=float)\n",
        "    start1 = np.asarray(spec.start1, dtype=float)\n"
        "    start2 = np.asarray(spec.start2, dtype=float)\n",
    )
    replace_once(
        "evidence_extension_v1/track_c/nist.py",
        "    z_start1 = (start1 - center) / radius\n",
        "    z_start1 = (start1 - center) / radius\n"
        "    z_start2 = (start2 - center) / radius\n",
    )

    marker = '''    def metrics(z: np.ndarray) -> dict[str, float]:\n        beta = physical(z)\n        return {\n            "certified_rss_ratio": float(objective(z)),\n            "certified_parameter_scaled_error": float(np.linalg.norm((beta - certified) / radius)),\n            "physical_parameter_norm": float(np.linalg.norm(beta)),\n        }\n\n    return ScientificTask(\n'''
    replacement = '''    def metrics(z: np.ndarray) -> dict[str, float]:\n        beta = physical(z)\n        return {\n            "certified_rss_ratio": float(objective(z)),\n            "certified_parameter_scaled_error": float(np.linalg.norm((beta - certified) / radius)),\n            "physical_parameter_norm": float(np.linalg.norm(beta)),\n        }\n\n    def specialized_reference() -> dict[str, object]:\n        residual_scale = math.sqrt(spec.certified_rss)\n\n        def residual(z: np.ndarray) -> np.ndarray:\n            predicted = spec.model(physical(z), x)\n            if not np.all(np.isfinite(predicted)):\n                return np.full_like(y, 1e100)\n            return (predicted - y) / residual_scale\n\n        best_z = z_start1.copy()\n        best_value = objective(best_z)\n        evaluations = 0\n        statuses: list[int] = []\n        for start in [z_start1, z_start2]:\n            fit = least_squares(\n                residual,\n                np.clip(start, -1.0, 1.0),\n                bounds=(-np.ones_like(start), np.ones_like(start)),\n                max_nfev=20_000,\n                xtol=1e-13,\n                ftol=1e-13,\n                gtol=1e-13,\n                x_scale="jac",\n            )\n            value = objective(fit.x)\n            evaluations += int(fit.nfev) + 1\n            statuses.append(int(fit.status))\n            if value < best_value:\n                best_value = value\n                best_z = fit.x.copy()\n        return {\n            "method": "scipy_least_squares_official_far_near_starts",\n            "xbest": best_z,\n            "fbest": best_value,\n            "iterations": evaluations,\n            "objective_evaluations": evaluations,\n            "metadata": {\n                "starts": 2,\n                "statuses": statuses,\n                "max_nfev_per_start": 20_000,\n            },\n        }\n\n    return ScientificTask(\n'''
    replace_once(
        "evidence_extension_v1/track_c/nist.py",
        marker,
        replacement,
    )
    replace_once(
        "evidence_extension_v1/track_c/nist.py",
        '''            "certified_z": z_certified.tolist(),\n            "description": spec.description,\n        },\n    )\n''',
        '''            "certified_z": z_certified.tolist(),\n            "description": spec.description,\n        },\n        specialized_reference=specialized_reference,\n    )\n''',
    )


def amend_checkpoint_semantics() -> None:
    replace_once(
        "evidence_extension_v1/track_c/common.py",
        "def verify_source_identity(*, require_clean: bool = True) -> dict[str, Any]:\n",
        '''def checkpoint_multipliers(budget_multiplier: int) -> list[int]:\n    available = [\n        value\n        for value in CHECKPOINTS_PER_DIMENSION\n        if value <= int(budget_multiplier)\n    ]\n    return sorted(set([*available, int(budget_multiplier)]))\n\n\ndef verify_source_identity(*, require_clean: bool = True) -> dict[str, Any]:\n''',
    )
    for relative in [
        "evidence_extension_v1/track_c/run_track_c_shard.py",
        "evidence_extension_v1/track_c/finalize_track_c.py",
    ]:
        replace_once(
            relative,
            "    TARGET_RATIOS,\n    normalize_gap,\n",
            "    TARGET_RATIOS,\n    checkpoint_multipliers,\n"
            "    normalize_gap,\n",
        )
    replace_once(
        "evidence_extension_v1/track_c/run_track_c_shard.py",
        "    points = sorted(set([1, 3, 10, 30, 100, 300, task.budget_multiplier]))\n",
        "    points = checkpoint_multipliers(task.budget_multiplier)\n",
    )
    replace_once(
        "evidence_extension_v1/track_c/finalize_track_c.py",
        "        checkpoints = sorted(set([1, 3, 10, 30, 100, 300, int(row.budget_multiplier)]))\n",
        "        checkpoints = checkpoint_multipliers(int(row.budget_multiplier))\n",
    )
    replace_once(
        "evidence_extension_v1/track_c/finalize_track_c.py",
        '''                        "checkpoint_evaluations_per_dimension": checkpoint,\n                        "reached": reached,\n''',
        '''                        "checkpoint_evaluations_per_dimension": checkpoint,\n                        "is_final_checkpoint": checkpoint == int(row.budget_multiplier),\n                        "reached": reached,\n''',
    )


def amend_primary_analysis() -> None:
    path = ROOT / "evidence_extension_v1/track_c/finalize_track_c.py"
    text = path.read_text(encoding="utf-8")
    marker = "\n\ndef write_manifest(root: Path) -> None:\n"
    if text.count(marker) != 1:
        raise RuntimeError("Could not locate finalizer insertion point.")
    function = '''\n\ndef paired_target_statistics(primary: pd.DataFrame) -> pd.DataFrame:\n    rows: list[dict[str, object]] = []\n    endpoint_specs: list[tuple[str, pd.DataFrame]] = [\n        ("final_registered_budget", primary[primary.is_final_checkpoint])\n    ]\n    for checkpoint in sorted(\n        primary.checkpoint_evaluations_per_dimension.unique()\n    ):\n        endpoint_specs.append(\n            (\n                f"checkpoint_{int(checkpoint)}d",\n                primary[\n                    primary.checkpoint_evaluations_per_dimension == checkpoint\n                ],\n            )\n        )\n\n    rng = np.random.default_rng(20260808)\n    for endpoint, subset in endpoint_specs:\n        block = (\n            subset.groupby(\n                ["domain", "task_id", "paired_seed", "algorithm"],\n                as_index=False,\n            )\n            .agg(target_fraction=("reached", "mean"))\n        )\n        basin = block[block.algorithm == "BasinGraph"][\n            ["domain", "task_id", "paired_seed", "target_fraction"]\n        ].rename(columns={"target_fraction": "basingraph_fraction"})\n        endpoint_rows: list[dict[str, object]] = []\n        raw_p: list[float] = []\n        for algorithm in ALGORITHMS[1:]:\n            paired = block[block.algorithm == algorithm][\n                ["domain", "task_id", "paired_seed", "target_fraction"]\n            ].merge(\n                basin,\n                on=["domain", "task_id", "paired_seed"],\n                validate="one_to_one",\n            )\n            difference = (\n                paired.basingraph_fraction.to_numpy(float)\n                - paired.target_fraction.to_numpy(float)\n            )\n            nonzero = difference[\n                ~np.isclose(difference, 0.0, rtol=1e-12, atol=1e-14)\n            ]\n            if len(nonzero):\n                test = wilcoxon(\n                    nonzero,\n                    alternative="two-sided",\n                    zero_method="wilcox",\n                    method="auto",\n                )\n                statistic = float(test.statistic)\n                p_value = float(test.pvalue)\n            else:\n                statistic = 0.0\n                p_value = 1.0\n\n            if len(difference):\n                samples = rng.integers(\n                    0,\n                    len(difference),\n                    size=(2000, len(difference)),\n                )\n                bootstrap = difference[samples].mean(axis=1)\n                low, high = np.quantile(bootstrap, [0.025, 0.975])\n            else:\n                low = high = 0.0\n\n            bg_better = int(np.sum(difference > 1e-14))\n            baseline_better = int(np.sum(difference < -1e-14))\n            ties = int(len(difference) - bg_better - baseline_better)\n            endpoint_rows.append(\n                {\n                    "endpoint": endpoint,\n                    "baseline": algorithm,\n                    "baseline_display": DISPLAY_NAMES[algorithm],\n                    "paired_blocks": len(difference),\n                    "basingraph_better_blocks": bg_better,\n                    "baseline_better_blocks": baseline_better,\n                    "ties": ties,\n                    "mean_fraction_difference_basingraph_minus_baseline": (\n                        float(np.mean(difference)) if len(difference) else 0.0\n                    ),\n                    "bootstrap_95_low": float(low),\n                    "bootstrap_95_high": float(high),\n                    "wilcoxon_statistic": statistic,\n                    "raw_p": p_value,\n                    "rank_biserial_positive_means_basingraph_better": (\n                        (bg_better - baseline_better)\n                        / max(bg_better + baseline_better, 1)\n                    ),\n                }\n            )\n            raw_p.append(p_value)\n        for row, adjusted in zip(\n            endpoint_rows,\n            holm_adjust(raw_p),\n        ):\n            row["holm_p"] = adjusted\n        rows.extend(endpoint_rows)\n    return pd.DataFrame(rows)\n'''
    path.write_text(text.replace(marker, function + marker, 1), encoding="utf-8")

    replace_once(
        "evidence_extension_v1/track_c/finalize_track_c.py",
        "    checks: dict[str, bool] = {\n",
        '''    reference_path = run_root / "task_specific_reference_results.csv"\n    expected_reference_rows = 3 if args.mode == "smoke" else 18\n    reference_rows = (\n        len(pd.read_csv(reference_path)) if reference_path.is_file() else 0\n    )\n    checks: dict[str, bool] = {\n        "task_specific_reference_rows": (\n            reference_rows == expected_reference_rows\n        ),\n''',
    )
    replace_once(
        "evidence_extension_v1/track_c/finalize_track_c.py",
        '''    checkpoint_summary.to_csv(run_root / "target_fraction_summary.csv", index=False)\n\n    family_target = (\n''',
        '''    checkpoint_summary.to_csv(\n        run_root / "target_fraction_summary.csv",\n        index=False,\n    )\n\n    final_target_summary = (\n        primary[primary.is_final_checkpoint]\n        .groupby(["algorithm", "domain"])\n        .agg(\n            target_fraction=("reached", "mean"),\n            records=("reached", "size"),\n        )\n        .reset_index()\n    )\n    final_target_summary.to_csv(\n        run_root / "final_registered_budget_target_fraction.csv",\n        index=False,\n    )\n\n    primary_pairwise = paired_target_statistics(primary)\n    primary_pairwise.to_csv(\n        run_root / "pairwise_target_fraction.csv",\n        index=False,\n    )\n\n    family_target = (\n''',
    )


def amend_execution_order() -> None:
    replace_once(
        "evidence_extension_v1/track_c/run_track_c_mac.sh",
        '''  python evidence_extension_v1/track_c/finalize_track_c.py \\\n    --mode smoke \\\n    --run-id "${RUN_ID}"\n''',
        '''  python evidence_extension_v1/track_c/task_specific_references.py \\\n    --mode smoke \\\n    --output "results_b21/track_c/${RUN_ID}/task_specific_reference_results.csv"\n  python evidence_extension_v1/track_c/finalize_track_c.py \\\n    --mode smoke \\\n    --run-id "${RUN_ID}"\n''',
    )
    replace_once(
        "evidence_extension_v1/track_c/run_track_c_mac.sh",
        '''  python evidence_extension_v1/track_c/finalize_track_c.py \\\n    --mode confirmatory \\\n    --run-id "${RUN_ID}" \\\n    --authorize-confirmatory\n''',
        '''  python evidence_extension_v1/track_c/task_specific_references.py \\\n    --mode confirmatory \\\n    --authorize-confirmatory \\\n    --output "results_b21/track_c/${RUN_ID}/task_specific_reference_results.csv"\n  python evidence_extension_v1/track_c/finalize_track_c.py \\\n    --mode confirmatory \\\n    --run-id "${RUN_ID}" \\\n    --authorize-confirmatory\n''',
    )


def amend_tests() -> None:
    replace_once(
        "tests_v2/test_track_c_contract.py",
        "    TARGET_RATIOS,\n    seed_for_c1,\n",
        "    TARGET_RATIOS,\n    checkpoint_multipliers,\n"
        "    seed_for_c1,\n",
    )
    path = ROOT / "tests_v2/test_track_c_contract.py"
    text = path.read_text(encoding="utf-8")
    block = '''\n\ndef test_checkpoint_availability_and_final_budget() -> None:\n    assert checkpoint_multipliers(75) == [1, 3, 10, 30, 75]\n    assert checkpoint_multipliers(100) == [1, 3, 10, 30, 100]\n    assert checkpoint_multipliers(150) == [1, 3, 10, 30, 100, 150]\n    assert checkpoint_multipliers(300) == [1, 3, 10, 30, 100, 300]\n    assert checkpoint_multipliers(1000) == [1, 3, 10, 30, 100, 300, 1000]\n\n\ndef test_registered_task_specific_reference_hooks() -> None:\n    assert callable(\n        make_c1_task("phase_retrieval", 1).specialized_reference\n    )\n    assert callable(\n        make_c1_task("matrix_factorization", 1).specialized_reference\n    )\n    assert callable(make_nist_task("BoxBOD").specialized_reference)\n'''
    if "test_checkpoint_availability_and_final_budget" not in text:
        path.write_text(text + block, encoding="utf-8")


def amend_protocols_and_identity() -> None:
    note = (
        ROOT
        / "protocols/evidence_extension_v1/track_c/"
        "TRACK_C_IMPLEMENTATION_NOTE_003.md"
    )
    note.write_text(
        "# Track C implementation note 003: checkpoint semantics and "
        "descriptive references\n\n"
        "Status: **registered before any Track C confirmatory objective "
        "evaluation**  \n"
        "Date: 2026-08-08  \n"
        "Track C confirmatory objective evaluations before amendment: **0**\n\n"
        "Fixed-budget summaries now include only checkpoints that exist for "
        "a task and tag each family-specific final budget explicitly. This "
        "removes the diagnostic-only truncation of shorter budgets to a "
        "nominal `300d` label. Registered descriptive task-specific anchors "
        "are implemented: spectral Wirtinger flow for phase retrieval, "
        "alternating ridge least squares for matrix factorization, and SciPy "
        "least squares from the official NIST far and near starts. These "
        "anchors are not pooled with the seven general-purpose optimizers. "
        "Primary target-fraction comparisons now include paired Wilcoxon "
        "tests, Holm correction, rank-biserial effects and bootstrap "
        "confidence intervals.\n",
        encoding="utf-8",
    )

    for relative in [
        "protocols/evidence_extension_v1/track_c/"
        "TRACK_C_CONFIRMATORY_PROTOCOL.md",
        "protocols/evidence_extension_v1/track_c/"
        "TRACK_C_MASTER_PROTOCOL_DRAFT.md",
        "protocols/evidence_extension_v1/track_c/"
        "TRACK_C_ANALYSIS_PLAN.md",
    ]:
        path = ROOT / relative
        content = path.read_text(encoding="utf-8")
        if "Implementation note 003" not in content:
            content += (
                "\n\n## Implementation note 003\n\n"
                "Only checkpoints available under a task budget are analyzed, "
                "and the family-specific final budget is explicitly "
                "identified. Registered task-specific references comprise "
                "spectral Wirtinger flow, alternating ridge least squares and "
                "NIST far/near-start least squares. They are descriptive and "
                "excluded from the general-purpose aggregate ranking. Paired "
                "primary target-fraction inference is frozen before "
                "confirmatory evaluation.\n"
            )
            path.write_text(content, encoding="utf-8")

    lock_path = (
        ROOT
        / "protocols/evidence_extension_v1/track_c/"
        "TRACK_C_CONFIRMATORY_LOCK.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["protocol_revision"] = "1.1.0"
    lock["implementation_notes"] = [1, 2, 3]
    lock["task_specific_reference_rows"] = {
        "smoke": 3,
        "confirmatory": 18,
    }
    lock["checkpoint_semantics"] = (
        "available checkpoints plus explicit registered final budget"
    )
    lock_path.write_text(
        json.dumps(lock, indent=2) + "\n",
        encoding="utf-8",
    )

    materialize = ROOT / "evidence_extension_v1/track_c/materialize_track_c.py"
    text = materialize.read_text(encoding="utf-8")
    text = text.replace(
        '"identity_version": "1.0.1",',
        '"identity_version": "1.1.0",',
        1,
    )
    text = text.replace(
        '"identity_revision": "exclude generated Python bytecode from portable source identity",',
        '"identity_revision": "portable identity, available-checkpoint semantics, paired primary inference and registered descriptive task-specific references",',
        1,
    )
    marker = '        "reference_construction_tasks": len(references),\n'
    addition = (
        '        "reference_construction_tasks": len(references),\n'
        '        "task_specific_reference_tasks": 18,\n'
        '        "checkpoint_semantics": "available checkpoints plus explicit registered final budget",\n'
    )
    if text.count(marker) != 1:
        raise RuntimeError("Could not locate identity reference count.")
    text = text.replace(marker, addition, 1)
    provenance_marker = (
        '                "official_url": NIST_URL.format(name=name),\n'
    )
    provenance_addition = (
        '                "official_url": NIST_URL.format(name=name),\n'
        '                "retrieval_mirror": "https://raw.githubusercontent.com/lmfit/lmfit-py/fe389bbbd1fe936cd73742bd81fc6fce7ac92858/NIST_STRD/{name}.dat".format(name=name),\n'
        '                "mirror_commit": "fe389bbbd1fe936cd73742bd81fc6fce7ac92858",\n'
    )
    if text.count(provenance_marker) != 1:
        raise RuntimeError("Could not locate NIST provenance URL field.")
    materialize.write_text(
        text.replace(provenance_marker, provenance_addition, 1),
        encoding="utf-8",
    )


def main() -> None:
    amend_tasks()
    amend_nist()
    amend_checkpoint_semantics()
    amend_primary_analysis()
    amend_execution_order()
    amend_tests()
    amend_protocols_and_identity()
    print("TRACK_C_PRECONFIRMATORY_AMENDMENT_APPLIED")


if __name__ == "__main__":
    main()
