from __future__ import annotations

import io
import tempfile
import textwrap
import sys
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from src.cli.main import main
from src.config import settings as settings_module


class CliSettingsTests(unittest.TestCase):
    def tearDown(self) -> None:
        settings_module.init_settings()

    def _install_fake_module(self, name: str, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        original = sys.modules.get(name)
        sys.modules[name] = module
        self.addCleanup(self._restore_module, name, original)
        return module

    @staticmethod
    def _restore_module(name: str, original) -> None:
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    def test_init_settings_uses_explicit_config_and_cli_directory_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [file]
                    data_directory = "configured-data"
                    result_directory = "configured-results"
                    rc_param_filename = "configured-rc.csv"
                    """
                ).strip(),
                encoding="utf-8",
            )

            configured = settings_module.init_settings(
                config_path=config_path,
                data_dir="cli-data",
                result_dir="cli-results",
            )

            self.assertEqual(configured.rc_param_filename, "configured-rc.csv")
            self.assertEqual(configured.data_dir, configured.root_dir / "cli-data")
            self.assertEqual(configured.result_dir, configured.root_dir / "cli-results")

    def test_missing_config_path_exits_with_clear_error(self) -> None:
        missing = Path(tempfile.gettempdir()) / "manga-dm-missing-config.toml"

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                main(["--config", str(missing), "stage1", "--ifu", "test"])

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("config file not found", stderr.getvalue())
        self.assertIn(str(missing), stderr.getvalue())

    def test_stage1_receives_global_result_dir_override(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_run_stage1(**kwargs):
            calls.append(kwargs)

        self._install_fake_module("src.pipeline.stage1", run_stage1=fake_run_stage1)
        main(["--result-dir", "custom-results", "stage1", "--ifu", "test"])

        self.assertEqual(calls[0]["result_dir_override"], "custom-results")

    def test_stage2_receives_global_result_dir_override(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_run_stage2(**kwargs):
            calls.append(kwargs)

        self._install_fake_module("src.pipeline.stage2", run_stage2=fake_run_stage2)
        main(["--result-dir", "custom-results", "stage2", "--fit"])

        self.assertEqual(calls[0]["result_dir_override"], "custom-results")

    def test_merge_receives_global_result_dir_override(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_merge_samples(**kwargs):
            calls.append(kwargs)

        self._install_fake_module("src.pipeline.stage2", merge_samples=fake_merge_samples)
        main(
            [
                "--result-dir",
                "custom-results",
                "merge",
                "--ifu-file",
                "plateifus.txt",
            ]
        )

        self.assertEqual(calls[0]["result_dir_override"], "custom-results")

    def test_sample_receives_global_result_dir_override(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_generate_robustness_sample(**kwargs):
            calls.append(kwargs)

        self._install_fake_module(
            "src.pipeline.selection",
            generate_robustness_sample=fake_generate_robustness_sample,
        )
        main(["--result-dir", "custom-results", "sample", "--n", "3"])

        self.assertEqual(calls[0]["result_dir_override"], "custom-results")


if __name__ == "__main__":
    unittest.main()
