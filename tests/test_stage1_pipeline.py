from __future__ import annotations

import sys
import tempfile
import types
import unittest
import importlib
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class Stage1PipelineTests(unittest.TestCase):
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

    def _import_stage1_with_fakes(self, *, stored_rows=None):
        fake_results = types.ModuleType("src.data.results")
        fake_results.get_params_file = lambda *args, **kwargs: None
        fake_results.get_processed_plate_ifus = lambda *args, **kwargs: set()

        def fake_store_params_file(*args, **kwargs):
            if stored_rows is not None:
                stored_rows.append(args)

        fake_results.store_params_file = fake_store_params_file
        fake_results.store_posterior_samples_file = lambda *args, **kwargs: None
        self._install_module("src.data.results", fake_results)

        fake_catalog = types.ModuleType("src.data.catalog")

        class FakeDrpallUtil:
            def __init__(self, path):
                self.path = Path(path)

        fake_catalog.DrpallUtil = FakeDrpallUtil
        fake_catalog.get_plateifu_list = lambda *args, **kwargs: []
        self._install_module("src.data.catalog", fake_catalog)

        fake_fits = types.ModuleType("src.data.fits")

        class FakeFitsUtil:
            def __init__(self, data_dir):
                self.data_dir = Path(data_dir)

            def get_drpall_file(self):
                return self.data_dir / "drpall.fits"

            def get_firefly_file(self):
                return self.data_dir / "firefly.fits"

            def get_maps_file(self, plate_ifu, checksum=False, download=False):
                return self.data_dir / f"{plate_ifu}.fits"

        fake_fits.FitsUtil = FakeFitsUtil
        self._install_module("src.data.fits", fake_fits)

        fake_firefly = types.ModuleType("src.data.firefly")

        class FakeFireflyUtil:
            def __init__(self, path):
                self.path = Path(path)

            def close(self):
                pass

        fake_firefly.FireflyUtil = FakeFireflyUtil
        self._install_module("src.data.firefly", fake_firefly)

        fake_maps = types.ModuleType("src.data.maps")

        class FakeMapsUtil:
            def __init__(self, path):
                self.path = Path(path)

            def get_eml_gflux_map(self):
                return np.ones((2, 2)), None, None

            def get_fwhm(self):
                return 2.5

            def get_pixel_scale(self):
                return 0.5

            def close(self):
                pass

        fake_maps.MapsUtil = FakeMapsUtil
        self._install_module("src.data.maps", fake_maps)

        fake_rotation = types.ModuleType("src.models.rotation_curve")

        class FakeRotCurve:
            def __init__(self, *args, **kwargs):
                self.plate_ifu = None

            def set_PLATE_IFU(self, plate_ifu):
                self.plate_ifu = plate_ifu

            def get_vel_obs(self):
                radius = np.array([[1.0, 2.0], [3.0, 4.0]])
                vel = np.array([[10.0, 20.0], [30.0, 40.0]])
                ivar = np.ones((2, 2))
                phi = np.zeros((2, 2))
                return radius, vel, ivar, phi

            def get_radius_fit(self, radius_max, count=1000):
                return np.linspace(0.1, radius_max, 4)

            def fit_vel_rot(self, vel_param, radius_fit=None):
                fit_params = {
                    "result": "success",
                    "inc": 0.5,
                    "Vsys": 1.0,
                    "phi_delta": 0.1,
                    "Rmax": 4.0,
                    "Rt": 2.0,
                    "NRMSE": 0.1,
                    "CHI_SQ_V": 1.0,
                }
                plot_result = {
                    "radius_obs": vel_param["radius_obs"],
                    "vel_obs": vel_param["vel_obs"],
                    "ivar_obs": vel_param["ivar_obs"],
                    "radius_rot": np.asarray(radius_fit),
                    "vel_rot": np.asarray(radius_fit) * 10.0,
                    "stderr_rot": np.ones_like(radius_fit),
                }
                return True, plot_result, fit_params

            @staticmethod
            def evaluate_fit_quality(fit_params, data_count):
                return {
                    "passed": True,
                    "inc_deg": 28.647,
                    "rmax_rt_ratio": 2.0,
                    "summary": "pass",
                    "fail_reasons": [],
                }

        fake_rotation.RotCurve = FakeRotCurve
        self._install_module("src.models.rotation_curve", fake_rotation)

        fake_dm = types.ModuleType("src.models.dm_nfw")
        fake_dm.DmNfw = object
        self._install_module("src.models.dm_nfw", fake_dm)

        fake_viz = types.ModuleType("src.viz.utils")

        class FakePlotUtil:
            def __init__(self, *args, **kwargs):
                pass

            def plot_rv_curves(self, *args, **kwargs):
                pass

        fake_viz.PlotUtil = FakePlotUtil
        self._install_module("src.viz.utils", fake_viz)

        sys.modules.pop("src.pipeline.stage1", None)
        pipeline_pkg = sys.modules.get("src.pipeline")
        if pipeline_pkg is not None and hasattr(pipeline_pkg, "stage1"):
            delattr(pipeline_pkg, "stage1")
        return importlib.import_module("src.pipeline.stage1")

    def test_worker_delegates_to_current_process_function_not_legacy_main(self) -> None:
        calls: list[dict[str, object]] = []
        write_lock = object()

        stage1 = self._import_stage1_with_fakes()

        def fake_process_plate_ifu(**kwargs):
            calls.append(kwargs)

        fake_main = types.ModuleType("main")
        fake_main._set_result_dir = lambda *args, **kwargs: None
        fake_main._set_r0_frac = lambda *args, **kwargs: None
        fake_main._set_m200_prior_dex = lambda *args, **kwargs: None
        fake_main._set_inc_prior_enable = lambda *args, **kwargs: None

        def legacy_process_called(*args, **kwargs):
            raise AssertionError("legacy main process should not be called")

        fake_main.process_plate_ifu = legacy_process_called
        self._install_module("main", fake_main)

        with patch("src.pipeline.stage1.process_plate_ifu", fake_process_plate_ifu):
            with redirect_stdout(StringIO()):
                stage1.process_plate_ifu_worker(
                    "1000-12701",
                    run_nfw=True,
                    debug=True,
                    result_dir_override="custom-results",
                    r0_frac=0.25,
                    m200_prior_dex=0.2,
                    inc_prior_enable=True,
                    write_lock=write_lock,
                )

        self.assertEqual(
            calls,
            [
                {
                    "plate_ifu": "1000-12701",
                    "process_nfw": True,
                    "debug": True,
                    "result_dir_override": "custom-results",
                    "r0_frac": 0.25,
                    "m200_prior_dex": 0.2,
                    "inc_prior_enable": True,
                    "write_lock": write_lock,
                }
            ],
        )

    def test_run_stage1_all_uses_configured_data_list(self) -> None:
        stage1 = self._import_stage1_with_fakes()
        get_plateifu_list = MagicMock(return_value=["1000-10001"])

        with (
            patch.object(stage1, "get_plateifu_list", get_plateifu_list),
            patch.object(stage1, "get_processed_plate_ifus", return_value=set()),
            patch.object(stage1, "process_plate_ifu"),
            redirect_stdout(StringIO()),
        ):
            stage1.run_stage1(ifu="all", n_cores=1)

        get_plateifu_list.assert_called_once_with(
            filepath=stage1.settings.data_dir / stage1.PLATES_FILENAME
        )

    def test_run_stage1_nfw_uses_successful_nfw_samples_for_completion(self) -> None:
        stage1 = self._import_stage1_with_fakes()

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(
                    stage1,
                    "get_processed_plate_ifus",
                    return_value={"1000-10001"},
                ) as get_processed,
                patch.object(stage1, "process_plate_ifu") as process,
                redirect_stdout(StringIO()),
            ):
                stage1.run_stage1(
                    ifu="1000-10001",
                    nfw=True,
                    n_cores=1,
                    result_dir_override=tmp,
                )

            get_processed.assert_called_once_with(
                stage1.settings.nfw_param_cm200_filename,
                stage1.settings.resolve_result_dir(tmp),
                successful_only=True,
                required_sample_filename=(
                    stage1.settings.nfw_param_cm200_sample_filename
                ),
            )
            process.assert_not_called()

    def test_run_stage1_parallel_passes_one_lock_to_workers(self) -> None:
        stage1 = self._import_stage1_with_fakes()
        write_lock = object()
        worker_args: list[tuple[object, ...]] = []

        class FakeManager:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def Lock(self):
                return write_lock

        class FakePool:
            def __init__(self, processes):
                self.processes = processes

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def starmap(self, function, args):
                worker_args.extend(args)
                return [None] * len(args)

        with (
            patch.object(stage1, "get_processed_plate_ifus", return_value=set()),
            patch.object(stage1.multiprocessing, "Manager", return_value=FakeManager()),
            patch.object(stage1.multiprocessing, "Pool", FakePool),
            redirect_stdout(StringIO()),
        ):
            stage1.run_stage1(ifu="1000-10001", nfw=True, n_cores=2)

        self.assertIs(worker_args[0][-1], write_lock)

    def test_process_plate_ifu_uses_current_modules_for_rc_only(self) -> None:
        stored_rows: list[tuple[object, ...]] = []
        stage1 = self._import_stage1_with_fakes(stored_rows=stored_rows)

        fake_main = types.ModuleType("main")
        fake_main._set_result_dir = lambda *args, **kwargs: None
        fake_main._set_r0_frac = lambda *args, **kwargs: None
        fake_main._set_m200_prior_dex = lambda *args, **kwargs: None
        fake_main._set_inc_prior_enable = lambda *args, **kwargs: None

        def legacy_process_called(*args, **kwargs):
            raise AssertionError("legacy main process should not be called")

        fake_main.process_plate_ifu = legacy_process_called
        self._install_module("main", fake_main)

        with redirect_stdout(StringIO()):
            stage1.process_plate_ifu(
                "1000-12701",
                process_nfw=False,
                result_dir_override="custom-results",
            )

        self.assertEqual(stored_rows[0][0], "1000-12701")
        self.assertEqual(stored_rows[0][2], stage1.settings.rc_param_filename)
        self.assertEqual(stored_rows[0][3], stage1.settings.resolve_result_dir("custom-results"))


if __name__ == "__main__":
    unittest.main()
