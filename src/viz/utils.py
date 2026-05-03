"""Shared visualization utilities (colors, formatters, helpers).

Thin wrapper — delegates to the original ``src-orig/util/plot_util.py``
until PlotUtil and plot_posterior_1d_hdi are fully migrated.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_posterior_1d_hdi(*args, **kwargs):
    """Generate a 1-D posterior density histogram with ETI shading.

    Thin wrapper around the legacy implementation.
    """
    import sys
    from pathlib import Path as _Path
    _old_util = _Path(__file__).resolve().parent.parent.parent / "src-orig" / "util"
    if str(_old_util) not in sys.path:
        sys.path.insert(0, str(_old_util))
    from plot_util import plot_posterior_1d_hdi as _impl
    return _impl(*args, **kwargs)
