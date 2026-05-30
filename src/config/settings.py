"""Singleton configuration loader for manga-dm.

Replaces 7 separate ``tomllib.load("config.toml")`` calls scattered across
the old scripts.  All modules should access configuration through::

    from src.config import settings
    print(settings.data_dir)

The configuration file (``config.toml``) is searched in the following order:

1. A path passed explicitly to :func:`init_settings`.
2. The current working directory.
3. The project root (one directory above ``src/``).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

__all__ = ["Settings", "init_settings", "settings"]


def _find_project_root() -> Path:
    """Return the project root directory.

    Walks upward from this source file up to the directory that contains both
    ``src/`` and ``pyproject.toml`` (or ``config.toml``).
    """
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "src" / "config" / "settings.py").exists():
        return candidate
    # Fallback: walk up from CWD
    cwd = Path.cwd().resolve()
    for start in (cwd, Path(__file__).resolve()):
        for p in [start] + list(start.parents):
            if (p / "pyproject.toml").exists() or (p / "config.toml").exists():
                return p
    return cwd


class Settings:
    """Application-wide settings loaded from ``config.toml``.

    Kept intentionally as a plain dict-backed object rather than a
    typed-dataclass so that adding new keys to the TOML file works
    without code changes.

    Usage::

        from src.config import settings
        settings.data_dir / "analysis"
    """

    # ── Project root ────────────────────────────────────────────────────
    _root_dir: Path | None = None
    _result_dir_override: str | Path | None = None

    @property
    def root_dir(self) -> Path:
        if self._root_dir is None:
            self._root_dir = _find_project_root()
        return self._root_dir

    @root_dir.setter
    def root_dir(self, value: str | Path) -> None:
        self._root_dir = Path(value).resolve()

    # ── Constructor / init ──────────────────────────────────────────────

    # ── Configuration ───────────────────────────────────────────────────
    # file cfg
    data_directory: str
    result_directory: str
    rc_param_filename: str
    nfw_param_cm200_filename: str
    nfw_param_cm200_sample_filename: str

    # thresholds
    SNR_THRESHOLD: float
    PHI_DEG_THRESHOLD: float
    IVAR_RATIO_THRESHOLD: float
    GSIGMA_MAX: float
    USE_GSIGMA_INST_CORR: bool
    INC_MIN: float
    INC_MAX: float
    VEL_OBS_COUNT_THRESHOLD: int
    RMAX_RT_FACTOR: int
    INFER_RHAT_THRESHOLD: float
    INFER_ESS_THRESHOLD: int
    HDI_PROB1: float
    HDI_PROB2: float
    PPC_HDI_VALUE_COVERAGE_THRESHOLD: float
    PPC_HDI_OVERLAP_THRESHOLD: float
    PPC_MEAS_SIGMA_SCALE: float

    # rc
    RADIUS_MIN_KPC: float
    BA_0: float
    VEL_SYSTEM_ERROR: float

    def __new__(cls) -> "Settings":
        # True singleton: always return the same instance
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
            cls._instance._init_from_toml()
        return cls._instance

    def configure(
        self,
        config_path: str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
        result_dir: str | Path | None = None,
        root_dir: str | Path | None = None,
    ) -> "Settings":
        """Reload settings and apply CLI-style directory overrides."""
        self._initialised = False
        if root_dir is not None:
            self.root_dir = root_dir
        self._init_from_toml(
            config_path=config_path,
            data_dir=data_dir,
            result_dir=result_dir,
        )
        return self

    def _init_from_toml(
        self,
        config_path: str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
        result_dir: str | Path | None = None,
    ) -> None:
        if self._initialised:
            return

        # Resolve config file
        if config_path is not None:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"config file not found: {path}")
        else:
            path = Path("config.toml")
            if not path.exists():
                path = self.root_dir / "config.toml"

        if path.exists():
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        else:
            raw = {}

        file_cfg = raw.get("file", {})
        thresholds = raw.get("thresholds", {})
        rc_cfg = raw.get("rc", {})
        plateifus_cfg = raw.get("plateifus", {})

        # ── file section ────────────────────────────────────────────
        self.data_directory = file_cfg.get("data_directory", "data")
        self.result_directory = file_cfg.get("result_directory", "results")
        self._result_dir_override = None
        self.rc_param_filename = file_cfg.get(
            "rc_param_filename", "rc_param.csv"
        )
        self.nfw_param_cm200_filename = file_cfg.get(
            "nfw_param_cm200_filename", "nfw_param_cm200.csv"
        )
        self.nfw_param_cm200_sample_filename = file_cfg.get(
            "nfw_param_cm200_sample_filename", "nfw_param_cm200_samples.nc"
        )

        # ── thresholds section ──────────────────────────────────────
        self.SNR_THRESHOLD = float(thresholds.get("SNR_THRESHOLD", 10.0))
        self.PHI_DEG_THRESHOLD = float(thresholds.get("PHI_DEG_THRESHOLD", 45.0))
        self.IVAR_RATIO_THRESHOLD = float(
            thresholds.get("IVAR_RATIO_THRESHOLD", 0.10)
        )
        self.GSIGMA_MAX = float(thresholds.get("GSIGMA_MAX", 0.0))
        self.USE_GSIGMA_INST_CORR = thresholds.get(
            "USE_GSIGMA_INST_CORR", True
        )
        self.INC_MIN = float(
            thresholds.get(
                "INC_MIN",
                plateifus_cfg.get("INC_MIN", 25.0),
            )
        )
        self.INC_MAX = float(
            thresholds.get(
                "INC_MAX",
                plateifus_cfg.get("INC_MAX", 70.0),
            )
        )
        self.VEL_OBS_COUNT_THRESHOLD = int(
            thresholds.get("VEL_OBS_COUNT_THRESHOLD", 150)
        )
        self.RMAX_RT_FACTOR = int(thresholds.get("RMAX_RT_FACTOR", 2))
        self.INFER_RHAT_THRESHOLD = float(
            thresholds.get("INFER_RHAT_THRESHOLD", 1.05)
        )
        self.INFER_ESS_THRESHOLD = int(
            thresholds.get("INFER_ESS_THRESHOLD", 200)
        )
        self.HDI_PROB1 = float(thresholds.get("HDI_PROB1", 0.68))
        self.HDI_PROB2 = float(thresholds.get("HDI_PROB2", 0.95))
        self.PPC_HDI_VALUE_COVERAGE_THRESHOLD = float(
            thresholds.get("PPC_HDI_VALUE_COVERAGE_THRESHOLD", 0.60)
        )
        self.PPC_HDI_OVERLAP_THRESHOLD = float(
            thresholds.get("PPC_HDI_OVERLAP_THRESHOLD", 0.80)
        )
        self.PPC_MEAS_SIGMA_SCALE = float(
            thresholds.get("PPC_MEAS_SIGMA_SCALE", 1.0)
        )

        # ── rc section ──────────────────────────────────────────────
        self.RADIUS_MIN_KPC = float(rc_cfg.get("RADIUS_MIN_KPC", 0.01))
        self.BA_0 = float(rc_cfg.get("BA_0", 0.2))
        self.VEL_SYSTEM_ERROR = float(rc_cfg.get("VEL_SYSTEM_ERROR", 5.0))

        if data_dir is not None:
            self.data_directory = str(data_dir)
        if result_dir is not None:
            self._result_dir_override = result_dir

        self._initialised = True

    # ── Derived paths ───────────────────────────────────────────────────

    def _resolve_root_relative_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.root_dir / path
        return path

    @property
    def data_dir(self) -> Path:
        return self._resolve_root_relative_path(self.data_directory)

    @property
    def result_dir(self) -> Path:
        if self._result_dir_override is not None:
            return self._resolve_root_relative_path(self._result_dir_override)
        return self.data_dir / self.result_directory

    def resolve_result_dir(self, override: str | Path | None = None) -> Path:
        """Return ``result_dir``, optionally overridden by a CLI argument."""
        if override is None:
            return self.result_dir
        return self._resolve_root_relative_path(override)

    def resolve_input_path(self, path_like: str | Path) -> Path:
        """Resolve an input path: if relative, make it absolute w.r.t. root."""
        path = Path(path_like)
        if not path.is_absolute():
            path = self.root_dir / path
        return path

    # ── Subdirectories (used by FitsUtil) ───────────────────────────────

    @property
    def drp_dir(self) -> Path:
        return self.data_dir / "redux"

    @property
    def dap_dir(self) -> Path:
        return self.data_dir / "analysis"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def firefly_dir(self) -> Path:
        return self.data_dir / "firefly"


# Module-level singleton — import this everywhere
settings = Settings()


def init_settings(
    config_path: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
    result_dir: str | Path | None = None,
    root_dir: str | Path | None = None,
) -> Settings:
    """Reload the module-level settings singleton.

    The singleton object is mutated in place so modules that imported
    ``settings`` keep seeing the current configuration.
    """
    return settings.configure(
        config_path=config_path,
        data_dir=data_dir,
        result_dir=result_dir,
        root_dir=root_dir,
    )
