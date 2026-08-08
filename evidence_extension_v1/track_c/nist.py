"""NIST StRD observed-data nonlinear-regression tasks for Track C."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable

import numpy as np

from .common import ROOT
from .tasks import ScientificTask


@dataclass(frozen=True)
class NISTSpec:
    name: str
    observations: int
    start1: tuple[float, ...]
    start2: tuple[float, ...]
    certified: tuple[float, ...]
    certified_rss: float
    model: Callable[[np.ndarray, np.ndarray], np.ndarray]
    difficulty: str
    description: str


def _chwirut(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    b1, b2, b3 = beta
    denominator = b2 + b3 * x
    if np.any(np.abs(denominator) < 1e-14):
        return np.full_like(x, np.nan)
    return np.exp(np.clip(-b1 * x, -700, 700)) / denominator


def _roszman(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    b1, b2, b3, b4 = beta
    denominator = x - b4
    if np.any(np.abs(denominator) < 1e-14):
        return np.full_like(x, np.nan)
    return b1 - b2 * x - np.arctan(b3 / denominator) / math.pi


def _enso(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    b1, b2, b3, b4, b5, b6, b7, b8, b9 = beta
    if abs(b4) < 1e-12 or abs(b7) < 1e-12:
        return np.full_like(x, np.nan)
    return (
        b1
        + b2 * np.cos(2 * math.pi * x / 12.0)
        + b3 * np.sin(2 * math.pi * x / 12.0)
        + b5 * np.cos(2 * math.pi * x / b4)
        + b6 * np.sin(2 * math.pi * x / b4)
        + b8 * np.cos(2 * math.pi * x / b7)
        + b9 * np.sin(2 * math.pi * x / b7)
    )


def _eckerle(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    b1, b2, b3 = beta
    if abs(b2) < 1e-14:
        return np.full_like(x, np.nan)
    return (b1 / b2) * np.exp(-0.5 * ((x - b3) / b2) ** 2)


def _bennett(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    b1, b2, b3 = beta
    base = b2 + x
    if abs(b3) < 1e-14 or np.any(base <= 0):
        return np.full_like(x, np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return b1 * np.power(base, -1.0 / b3)


def _boxbod(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    b1, b2 = beta
    return b1 * (1.0 - np.exp(np.clip(-b2 * x, -700, 700)))


SPECS: dict[str, NISTSpec] = {
    "Chwirut1": NISTSpec(
        "Chwirut1", 214,
        (0.1, 0.01, 0.02),
        (0.15, 0.008, 0.010),
        (1.9027818370e-1, 6.1314004477e-3, 1.0530908399e-2),
        2.3844771393e3, _chwirut, "lower",
        "NIST ultrasonic reference-block response versus metal distance.",
    ),
    "Roszman1": NISTSpec(
        "Roszman1", 25,
        (0.1, -1.0e-5, 1000.0, -100.0),
        (0.2, -5.0e-6, 1200.0, -150.0),
        (2.0196866396e-1, -6.1953516256e-6, 1.2044556708e3, -1.8134269537e2),
        4.9484847331e-4, _roszman, "average",
        "NIST sulfur-I quantum-defect observations.",
    ),
    "ENSO": NISTSpec(
        "ENSO", 168,
        (11.0, 3.0, 0.5, 40.0, -0.7, -1.3, 25.0, -0.3, 1.4),
        (10.0, 3.0, 0.5, 44.0, -1.5, 0.5, 26.0, -0.1, 1.5),
        (10.510749193, 3.0762128085, 0.53280138227, 44.311088700,
         -1.6231428586, 0.52554493756, 26.887614440, 0.21232288488,
         1.4966870418),
        7.8853978668e2, _enso, "average",
        "Monthly pressure-difference observations associated with ENSO cycles.",
    ),
    "Eckerle4": NISTSpec(
        "Eckerle4", 35,
        (1.0, 10.0, 500.0),
        (1.5, 5.0, 450.0),
        (1.5543827178, 4.0888321754, 451.54121844),
        1.4635887487e-3, _eckerle, "higher",
        "NIST circular-interference transmittance observations.",
    ),
    "Bennett5": NISTSpec(
        "Bennett5", 154,
        (-2000.0, 50.0, 0.8),
        (-1500.0, 45.0, 0.85),
        (-2523.5058043, 46.736564644, 0.93218483193),
        5.2404744073e-4, _bennett, "higher",
        "NIST superconductivity magnetization observations.",
    ),
    "BoxBOD": NISTSpec(
        "BoxBOD", 6,
        (1.0, 1.0),
        (100.0, 0.75),
        (213.80940889, 0.54723748542),
        1.1680088766e3, _boxbod, "higher",
        "Biochemical oxygen-demand observations from Box, Hunter and Hunter.",
    ),
}


_FLOAT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def load_nist_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = SPECS[name]
    path = ROOT / "data" / "nist" / f"{name}.dat"
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("data:") and "x" in line.lower() and "y" in line.lower():
            start = index + 1
    if start is None:
        # The official files put the numerical block after the final Data: line.
        candidates = [i for i, line in enumerate(lines) if line.strip().lower().startswith("data:")]
        if not candidates:
            raise RuntimeError(f"No data block found in {path}")
        start = candidates[-1] + 1
    pairs: list[tuple[float, float]] = []
    for line in lines[start:]:
        values = _FLOAT.findall(line)
        if len(values) == 2:
            pairs.append((float(values[0]), float(values[1])))
    if len(pairs) != spec.observations:
        raise RuntimeError(
            f"{name} observation count mismatch: {len(pairs)} != {spec.observations}"
        )
    data = np.asarray(pairs, dtype=float)
    return data[:, 0], data[:, 1]


def _coordinate_system(spec: NISTSpec) -> tuple[np.ndarray, np.ndarray]:
    start1 = np.asarray(spec.start1, dtype=float)
    start2 = np.asarray(spec.start2, dtype=float)
    certified = np.asarray(spec.certified, dtype=float)
    center = 0.5 * (start1 + start2)
    radius = np.maximum.reduce(
        [
            np.abs(start1 - center),
            np.abs(start2 - center),
            np.abs(certified - center),
            0.10 * np.maximum(np.abs(certified), 1.0),
        ]
    )
    radius = np.maximum(1.5 * radius, 1e-12)
    return center, radius


def make_nist_task(name: str) -> ScientificTask:
    spec = SPECS[name]
    y, x = load_nist_data(name)
    center, radius = _coordinate_system(spec)
    certified = np.asarray(spec.certified, dtype=float)
    start1 = np.asarray(spec.start1, dtype=float)

    def physical(z: np.ndarray) -> np.ndarray:
        return center + radius * np.asarray(z, dtype=float)

    def rss_from_beta(beta: np.ndarray) -> float:
        predicted = spec.model(np.asarray(beta, dtype=float), x)
        if not np.all(np.isfinite(predicted)):
            return 1e300
        residual = predicted - y
        value = float(np.dot(residual, residual))
        return value if np.isfinite(value) else 1e300

    def objective(z: np.ndarray) -> float:
        return float(rss_from_beta(physical(z)) / spec.certified_rss)

    z_certified = (certified - center) / radius
    z_start1 = (start1 - center) / radius
    f_ref = float(objective(z_certified))
    f_base = float(objective(z_start1))
    if f_base <= f_ref + 1e-12:
        f_base = f_ref + 1.0

    def metrics(z: np.ndarray) -> dict[str, float]:
        beta = physical(z)
        return {
            "certified_rss_ratio": float(objective(z)),
            "certified_parameter_scaled_error": float(np.linalg.norm((beta - certified) / radius)),
            "physical_parameter_norm": float(np.linalg.norm(beta)),
        }

    return ScientificTask(
        domain="c2",
        family=name,
        instance=0,
        dimension=len(spec.certified),
        budget_multiplier=1000,
        lb=-np.ones(len(spec.certified)),
        ub=np.ones(len(spec.certified)),
        objective=objective,
        metrics=metrics,
        f_ref=f_ref,
        f_base=f_base,
        reference_x=z_certified,
        metadata={
            "observations": spec.observations,
            "difficulty": spec.difficulty,
            "certified_rss": spec.certified_rss,
            "center": center.tolist(),
            "radius": radius.tolist(),
            "start1_z": z_start1.tolist(),
            "certified_z": z_certified.tolist(),
            "description": spec.description,
        },
    )
