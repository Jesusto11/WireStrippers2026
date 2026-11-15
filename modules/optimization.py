"""Utilidades de optimización multiobjetivo y análisis de incertidumbre."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

LUNAR_RADIUS_M = 1_737_400.0


def pareto_mask(
    frame: pd.DataFrame,
    objective_columns: Sequence[str],
    maximize: Sequence[str] = (),
) -> np.ndarray:
    """Devuelve máscara de soluciones no dominadas.

    Cada fila es una alternativa. Por defecto los objetivos se minimizan;
    las columnas incluidas en ``maximize`` se invierten internamente.
    Los valores NaN se consideran no comparables y quedan fuera de la frontera.
    """
    if not objective_columns:
        raise ValueError("Indica al menos una columna objetivo.")
    values = frame[list(objective_columns)].astype(float).to_numpy(copy=True)
    maximize_set = set(maximize)
    for index, column in enumerate(objective_columns):
        if column in maximize_set:
            values[:, index] *= -1

    finite = np.isfinite(values).all(axis=1)
    mask = np.zeros(len(frame), dtype=bool)
    valid_indices = np.flatnonzero(finite)
    for i in valid_indices:
        others = values[valid_indices]
        no_worse = np.all(others <= values[i], axis=1)
        strictly_better = np.any(others < values[i], axis=1)
        # Excluir la propia fila por la condición estrictamente mejor.
        if not np.any(no_worse & strictly_better):
            mask[i] = True
    return mask


def filtrar_frontera_pareto(
    frame: pd.DataFrame,
    objective_columns: Sequence[str],
    maximize: Sequence[str] = (),
) -> pd.DataFrame:
    result = frame.copy()
    result["Pareto"] = pareto_mask(result, objective_columns, maximize=maximize)
    return result


def generar_dispersion_lunar(
    lat: float,
    lon: float,
    sigma_east_m: float,
    sigma_north_m: float,
    n_samples: int = 5000,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera posiciones aleatorias alrededor de un punto nominal.

    Las desviaciones estándar representan hipótesis de dispersión local en
    metros; deben sustituirse por un modelo de dispersión del vehículo real.
    """
    if n_samples < 1:
        raise ValueError("El número de muestras debe ser mayor que cero.")
    if sigma_east_m < 0 or sigma_north_m < 0:
        raise ValueError("Las desviaciones estándar no pueden ser negativas.")
    rng = np.random.default_rng(seed)
    east = rng.normal(0.0, sigma_east_m, int(n_samples))
    north = rng.normal(0.0, sigma_north_m, int(n_samples))
    latitudes = float(lat) + np.degrees(north / LUNAR_RADIUS_M)
    cos_lat = max(abs(np.cos(np.radians(float(lat)))), 1e-6)
    longitudes = float(lon) + np.degrees(east / (LUNAR_RADIUS_M * cos_lat))
    latitudes = np.clip(latitudes, -90.0, 90.0)
    longitudes = (longitudes + 180.0) % 360.0 - 180.0
    return latitudes, longitudes


def estimar_probabilidad_factibilidad(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, float | int | np.ndarray]:
    """Aplica un evaluador booleano y estima la proporción de escenarios válidos.

    ``evaluator`` recibe arreglos (latitudes, longitudes) y debe devolver una
    máscara booleana del mismo tamaño. La función no define por sí sola qué
    significa 'factible': eso depende de las restricciones de la misión.
    """
    feasible = np.asarray(evaluator(latitudes, longitudes), dtype=bool)
    if feasible.shape != np.asarray(latitudes).shape:
        raise ValueError("La función evaluadora debe devolver una máscara del mismo tamaño que las muestras.")
    total = int(feasible.size)
    count = int(feasible.sum())
    return {
        "n_samples": total,
        "n_feasible": count,
        "probability": (count / total) if total else float("nan"),
        "mask": feasible,
    }


def ejecutar_nsga2(
    objective_function: Callable[[np.ndarray], Sequence[float]],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    population_size: int = 64,
    generations: int = 100,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Ejecuta NSGA-II sobre una función objetivo vectorial de minimización.

    La función objetivo recibe una solución ``x`` y devuelve un vector de
    objetivos, todos expresados como minimización. Para maximizar una métrica,
    devuelve su negativo. Se requiere instalar pymoo.
    """
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
    except ImportError as exc:
        raise ImportError("NSGA-II requiere la biblioteca pymoo. Instálala con: pip install pymoo") from exc

    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape or np.any(upper <= lower):
        raise ValueError("Los límites deben ser vectores unidimensionales del mismo tamaño y cada límite superior debe ser mayor que el inferior.")
    probe = np.asarray(objective_function((lower + upper) / 2), dtype=float).reshape(-1)
    if probe.size < 2:
        raise ValueError("NSGA-II requiere al menos dos funciones objetivo.")

    class LandingProblem(Problem):
        def __init__(self):
            super().__init__(n_var=lower.size, n_obj=probe.size, n_ieq_constr=0, xl=lower, xu=upper)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = np.vstack([
                np.asarray(objective_function(row), dtype=float).reshape(-1)
                for row in X
            ])

    result = minimize(
        LandingProblem(),
        NSGA2(pop_size=int(population_size)),
        termination=("n_gen", int(generations)),
        seed=int(seed),
        verbose=False,
    )
    return {
        "X": np.asarray(result.X),
        "F": np.asarray(result.F),
    }
