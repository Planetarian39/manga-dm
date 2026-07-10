from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import xarray as xr

from src.config import settings as settings_module
from src.data.results import (
    merge_posterior_samples_file,
    store_posterior_samples_file,
)


class Stage2PipelineTests(unittest.TestCase):
    def tearDown(self) -> None:
        settings_module.init_settings()

    def _install_module(self, name: str, module: types.ModuleType) -> None:
        original = sys.modules.get(name)
        sys.modules[name] = module
        self.addCleanup(self._restore_module, name, original)

    @staticmethod
    def _restore_module(name: str, original) -> None:
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    def test_merge_samples_uses_current_results_module_not_legacy_m200(self) -> None:
        calls: list[tuple[str, Path, set[str] | None]] = []

        fake_results = types.ModuleType("src.data.results")

        def fake_merge_posterior_samples_file(
            filename: str,
            result_dir: str | Path,
            plate_ifus: set[str] | None = None,
        ):
            calls.append((filename, Path(result_dir), plate_ifus))
            return Path(result_dir) / filename

        fake_results.load_posterior_sample_map = lambda *args, **kwargs: {}
        fake_results.merge_posterior_samples_file = fake_merge_posterior_samples_file
        self._install_module("src.data.results", fake_results)

        fake_m200 = types.ModuleType("m200")
        fake_m200._set_result_dir = lambda *args, **kwargs: None

        def legacy_merge_called(*args, **kwargs):
            raise AssertionError("legacy m200 merge should not be called")

        fake_m200.merge_posterior_samples_file = legacy_merge_called
        self._install_module("m200", fake_m200)
        sys.modules.pop("src.pipeline.stage2", None)

        with tempfile.TemporaryDirectory() as tmp:
            settings = settings_module.init_settings(result_dir=tmp)
            stage2 = importlib.import_module("src.pipeline.stage2")
            ifu_file = Path(tmp) / "plateifus.txt"
            ifu_file.write_text("1000-10001\n", encoding="utf-8")

            stage2.merge_samples(ifu_file=ifu_file)

        self.assertEqual(
            calls,
            [
                (
                    settings.nfw_param_cm200_sample_filename,
                    settings.result_dir,
                    {"1000-10001"},
                )
            ],
        )

    def test_merge_samples_rejects_missing_ifu_file(self) -> None:
        sys.modules.pop("src.pipeline.stage2", None)
        stage2 = importlib.import_module("src.pipeline.stage2")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                stage2.merge_samples(
                    ifu_file=Path(tmp) / "missing.txt",
                    result_dir_override=tmp,
                )

    def test_merge_posterior_samples_file_filters_plate_ifus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filename = "samples.nc"
            samples = {
                "log10_M200_samples": np.array([12.0, 12.1]),
                "log10_c_samples": np.array([0.8, 0.9]),
            }
            store_posterior_samples_file(
                "1000-10001", samples, filename, tmp
            )
            store_posterior_samples_file(
                "2000-20002", samples, filename, tmp
            )

            merged = merge_posterior_samples_file(
                filename,
                tmp,
                plate_ifus={"1000-10001"},
            )

            self.assertIsNotNone(merged)
            with xr.open_dataset(merged) as dataset:
                self.assertEqual(
                    dataset.coords["plate_ifu"].values.tolist(),
                    ["1000-10001"],
                )

    def test_run_stage2_fit_uses_current_population_entrypoint_not_legacy_m200(self) -> None:
        calls: list[dict[str, object]] = []

        fake_results = types.ModuleType("src.data.results")
        fake_results.load_posterior_sample_map = lambda *args, **kwargs: {}
        fake_results.merge_posterior_samples_file = lambda *args, **kwargs: None
        self._install_module("src.data.results", fake_results)

        fake_population = types.ModuleType("src.pipeline.population")

        def fake_fit_m200_c_population(**kwargs):
            calls.append(kwargs)

        fake_population.fit_m200_c_population = fake_fit_m200_c_population
        self._install_module("src.pipeline.population", fake_population)

        fake_m200 = types.ModuleType("m200")
        fake_m200._set_result_dir = lambda *args, **kwargs: None

        def legacy_fit_called(*args, **kwargs):
            raise AssertionError("legacy m200 fit should not be called")

        fake_m200.fit_m200_c_mcmc = legacy_fit_called
        self._install_module("m200", fake_m200)
        sys.modules.pop("src.pipeline.stage2", None)

        stage2 = importlib.import_module("src.pipeline.stage2")
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(StringIO()):
                stage2.run_stage2(
                    fit=True,
                    n_cores=3,
                    quality_cut="recommended",
                    result_dir_override=tmp,
                )

        self.assertEqual(
            calls,
            [
                {
                    "quality_cut": "recommended",
                    "result_dir_override": Path(tmp),
                }
            ],
        )

    def test_run_stage2_diagnose_uses_current_population_diagnostics(self) -> None:
        calls: list[dict[str, object]] = []

        fake_results = types.ModuleType("src.data.results")
        fake_results.merge_posterior_samples_file = lambda *args, **kwargs: None
        self._install_module("src.data.results", fake_results)

        fake_population = types.ModuleType("src.pipeline.population")

        def fake_run_m200_c_psis_diagnostics(**kwargs):
            calls.append(kwargs)
            return {"n_bad_k": 0}

        fake_population.run_m200_c_psis_diagnostics = fake_run_m200_c_psis_diagnostics
        self._install_module("src.pipeline.population", fake_population)

        fake_m200 = types.ModuleType("m200")

        def legacy_diagnose_called(*args, **kwargs):
            raise AssertionError("legacy m200 diagnostics should not be called")

        fake_m200.compute_psis_importance_diagnostics = legacy_diagnose_called
        self._install_module("m200", fake_m200)
        sys.modules.pop("src.pipeline.stage2", None)

        stage2 = importlib.import_module("src.pipeline.stage2")
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(StringIO()):
                stage2.run_stage2(
                    diagnose=True,
                    quality_cut="strict",
                    result_dir_override=tmp,
                )

        self.assertEqual(
            calls,
            [
                {
                    "quality_cut": "strict",
                    "result_dir_override": Path(tmp),
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
