"""ArviZ compatibility functions.

Thin wrapper for backward-compatibility: real implementation lives in
``src.stats.arviz_compat``.
"""

from src.stats.arviz_compat import (  # noqa: F401
    ensure_arviz_compat,
    get_arviz_api,
    get_az,
    get_posterior_dataset,
    get_prior_dataset,
    get_summary_interval_columns,
    require_pymc_stack,
    set_arviz_ci_defaults,
    summary_with_compat,
)
