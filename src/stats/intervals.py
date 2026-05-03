"""ETI/HDI interval computation and overlap diagnostics.

Unified from the duplicate implementations in ``src-orig/dm.py`` and
``src-orig/rc.py``.

Functions that depend on ArviZ (``calc_hdi_from_sample_matrix``) are kept
optional — they will raise ``ModuleNotFoundError`` only if called without
ArviZ installed.
"""

from __future__ import annotations

import numpy as np


def calc_eti_from_sample_matrix(
    samples_2d: np.ndarray, prob: float = 0.68
) -> np.ndarray:
    """Compute equal-tailed intervals (ETI) for each row of *samples_2d*.

    Parameters
    ----------
    samples_2d : ndarray, shape (n_param, n_samples)
        Posterior samples for each parameter.
    prob : float
        Interval probability [0, 1].

    Returns
    -------
    ndarray, shape (n_param, 2)
        Lower and upper bounds for each parameter.
    """
    samples_2d = np.asarray(samples_2d, dtype=float)
    if samples_2d.ndim != 2:
        raise ValueError("samples_2d must be a 2D array")

    tail = (1.0 - prob) / 2.0
    lower = np.nanquantile(samples_2d, tail, axis=1)
    upper = np.nanquantile(samples_2d, 1.0 - tail, axis=1)
    return np.stack([lower, upper], axis=1)


def calc_hdi_from_sample_matrix(
    samples_2d: np.ndarray, prob: float = 0.95
) -> np.ndarray:
    """Compute highest-density intervals (HDI) using ArviZ.

    Parameters
    ----------
    samples_2d : ndarray, shape (n_param, n_samples)
    prob : float

    Returns
    -------
    ndarray, shape (n_param, 2)
    """
    from src.stats.arviz_compat import get_az

    az = get_az()
    sample_cube = np.expand_dims(
        np.asarray(samples_2d, dtype=float).T, axis=0
    )
    return np.asarray(az.hdi(sample_cube, hdi_prob=prob), dtype=float)


def calc_interval_overlap_mask(
    values: np.ndarray,
    sigma: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sigma_scale: float = 1.0,
) -> np.ndarray:
    """Return a boolean mask where the interval ``[values - sigma_scale*sigma, values + sigma_scale*sigma]``
    overlaps with ``[lower, upper]``.

    This is the unified version of ``_calc_interval_overlap_mask`` from
    ``dm.py`` and ``rc.py``.
    """
    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    value_lower = values - sigma_scale * sigma
    value_upper = values + sigma_scale * sigma
    valid = (
        np.isfinite(value_lower)
        & np.isfinite(value_upper)
        & np.isfinite(lower)
        & np.isfinite(upper)
    )
    return valid & (
        np.maximum(value_lower, lower) <= np.minimum(value_upper, upper)
    )


def get_interval_value_formatter(values: np.ndarray):
    """Return a callable ``format(value: float) -> str`` tuned to the scale of *values*.

    Used to auto-format posterior point estimates and error bars.
    """
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return lambda value: f"{float(value):.1f}"

    def _format_value(value: float) -> str:
        value = float(value)
        if abs(value) >= 1000:
            return f"{value:.1e}"
        return f"{value:.1f}"

    return _format_value


def format_pair_interval_title(
    median: float,
    lower_err: float,
    upper_err: float,
    unit_label: str = "",
) -> str:
    """Format a LaTeX-style ``$val_{-err}^{+err}$`` string for a parameter estimate.

    Handles large numbers with scientific-notation scaling.
    """
    median = float(median)
    lower_err = float(lower_err)
    upper_err = float(upper_err)

    if np.isfinite(median) and abs(median) >= 1000:
        exponent = int(np.floor(np.log10(abs(median))))
        scale = 10**exponent
        median_text = f"{median / scale:.1f}"
        lower_text = f"{lower_err / scale:.1f}"
        upper_text = f"{upper_err / scale:.1f}"
        title_text = (
            rf"{median_text}_{{-{lower_text}}}^{{+{upper_text}}}"
            rf"\times 10^{{{exponent}}}"
        )
    else:
        median_text = f"{median:.1f}"
        lower_text = f"{lower_err:.1f}"
        upper_text = f"{upper_err:.1f}"
        title_text = (
            rf"{median_text}_{{-{lower_text}}}^{{+{upper_text}}}"
        )

    if unit_label:
        title_text += f" {unit_label}"
    return f"${title_text}$"
