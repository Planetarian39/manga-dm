from __future__ import annotations

import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from io import StringIO

from src.config import settings as settings_module
from pathlib import Path

from src.pipeline.selection import generate_robustness_sample, select_and_download


class SelectionPipelineTests(unittest.TestCase):
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

    def test_generate_robustness_sample_uses_current_population_module(self) -> None:
        calls: list[dict[str, object]] = []

        fake_population = types.ModuleType("src.models.population")

        def fake_generate_robustness_sample(**kwargs):
            calls.append(kwargs)
            return None

        fake_population.generate_robustness_sample = fake_generate_robustness_sample
        self._install_module("src.models.population", fake_population)

        fake_m200 = types.ModuleType("m200")
        fake_m200._set_result_dir = lambda *args, **kwargs: None

        def legacy_generate_called(*args, **kwargs):
            raise AssertionError("legacy m200 sample should not be called")

        fake_m200.generate_robustness_sample = legacy_generate_called
        self._install_module("m200", fake_m200)

        with redirect_stdout(StringIO()):
            generate_robustness_sample(n=7, result_dir_override="custom-results")

        self.assertEqual(
            calls,
            [
                {
                    "n_sample": 7,
                    "result_dir_override": "custom-results",
                }
            ],
        )

    def test_select_and_download_uses_current_data_modules_not_legacy_plates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            output_file = Path(tmp) / "selected.txt"
            settings_module.init_settings(data_dir=data_dir)

            downloads: list[tuple[str, str]] = []

            fake_fits = types.ModuleType("src.data.fits")

            class FakeFitsUtil:
                def __init__(self, data_dir_arg):
                    self.data_dir = Path(data_dir_arg)

                def get_drpall_file(self):
                    return self.data_dir / "drpall.fits"

                def get_maps_file(self, plateifu, checksum=True):
                    downloads.append(("maps", plateifu))

                def get_image_file(self, plateifu):
                    downloads.append(("image", plateifu))

            fake_fits.FitsUtil = FakeFitsUtil
            self._install_module("src.data.fits", fake_fits)

            fake_catalog = types.ModuleType("src.data.catalog")

            class FakeDrpallUtil:
                def __init__(self, drpall_file):
                    self.drpall_file = Path(drpall_file)

                def search_plateifu_by_inc(self, inc_min, inc_max):
                    return ["1000-12701", "1001-12702"], [inc_min, inc_max]

            fake_catalog.DrpallUtil = FakeDrpallUtil
            self._install_module("src.data.catalog", fake_catalog)

            fake_plates = types.ModuleType("plates")

            def legacy_plates_called(*args, **kwargs):
                raise AssertionError("legacy plates should not be called")

            fake_plates.main = legacy_plates_called
            self._install_module("plates", fake_plates)

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                selected = select_and_download(
                    inc_min=30.0,
                    inc_max=65.0,
                    ifu_file=str(output_file),
                    download=True,
                )

            self.assertEqual(selected, ["1000-12701", "1001-12702"])
            self.assertEqual(output_file.read_text(encoding="utf-8").splitlines(), selected)
            self.assertEqual(
                sorted(downloads),
                sorted(
                    [
                        ("maps", "1000-12701"),
                        ("image", "1000-12701"),
                        ("maps", "1001-12702"),
                        ("image", "1001-12702"),
                    ]
                ),
            )


if __name__ == "__main__":
    unittest.main()
