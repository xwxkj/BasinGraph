"""Principal search-direction construction for BasinGraph v2.0.0-rc2."""

from __future__ import annotations

import numpy as np

from .archive import BasinArchive
from .graph import BasinTransitionGraph
from .types import DirectionDiagnostics


def _append_unique(
    retained: list[np.ndarray],
    candidate: np.ndarray,
    cosine_threshold: float,
) -> bool:
    candidate = np.asarray(
        candidate,
        dtype=float,
    ).reshape(-1)
    norm = np.linalg.norm(candidate)
    if norm <= 1e-14:
        return False
    candidate = candidate / norm

    for existing in retained:
        if (
            abs(np.dot(existing, candidate))
            >= cosine_threshold
        ):
            return False

    retained.append(candidate)
    return True


def build_principal_directions(
    archive: BasinArchive,
    graph: BasinTransitionGraph,
    *,
    dimension: int,
    elite_count: int,
    covariance_direction_count: int,
    transition_direction_count: int,
    max_directions: int,
    cosine_threshold: float,
) -> tuple[list[np.ndarray], DirectionDiagnostics]:
    retained: list[np.ndarray] = []
    covariance_added = 0
    transition_added = 0
    coordinate_added = 0

    elite = archive.sorted_nodes()[
        : max(2, elite_count)
    ]
    if len(elite) >= 2:
        centers = np.vstack(
            [node.center for node in elite]
        )
        centered = centers - centers.mean(
            axis=0,
            keepdims=True,
        )
        covariance = (
            centered.T @ centered
        ) / max(len(elite) - 1, 1)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(
                covariance
            )
            order = np.argsort(eigenvalues)[::-1]
            for index in order:
                if (
                    covariance_added
                    >= covariance_direction_count
                    or len(retained)
                    >= max_directions
                ):
                    break
                if eigenvalues[index] <= 1e-18:
                    continue
                if _append_unique(
                    retained,
                    eigenvectors[:, index],
                    cosine_threshold,
                ):
                    covariance_added += 1
        except np.linalg.LinAlgError:
            pass

    nodes_by_id = {
        node.node_id: node
        for node in archive.nodes
    }
    transition_candidates = (
        graph.successful_transition_directions(
            nodes_by_id
        )
    )
    for _, direction in transition_candidates:
        if (
            transition_added
            >= transition_direction_count
            or len(retained)
            >= max_directions
        ):
            break
        if _append_unique(
            retained,
            direction,
            cosine_threshold,
        ):
            transition_added += 1

    for coordinate in range(dimension):
        if len(retained) >= max_directions:
            break
        direction = np.zeros(
            dimension,
            dtype=float,
        )
        direction[coordinate] = 1.0
        if _append_unique(
            retained,
            direction,
            cosine_threshold,
        ):
            coordinate_added += 1

    diagnostics = DirectionDiagnostics(
        covariance_directions=covariance_added,
        transition_directions=transition_added,
        coordinate_fallback_directions=coordinate_added,
        retained_directions=len(retained),
        direction_dimension=dimension,
    )
    return retained, diagnostics
