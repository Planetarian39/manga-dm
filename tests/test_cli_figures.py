from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

from src.cli.main import main
from src.config import settings as settings_module


class CliFiguresTests(unittest.TestCase):
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

    def test_figures_command_dispatches_to_current_viz_modules(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def recorder(name):
            def _record(**kwargs):
                calls.append((name, kwargs))
                return kwargs.get("output_path")

            return _record

        fake_velocity = types.ModuleType("src.viz.velocity_maps")
        fake_velocity.plot_velocity_field_comparison = recorder("velocity_comparison")
        fake_velocity.plot_velocity_field_panels = recorder("velocity_panels")
        self._install_module("src.viz.velocity_maps", fake_velocity)

        fake_rc = types.ModuleType("src.viz.rc_curves")
        fake_rc.plot_rc_fit_summary_comparison = recorder("rc_comparison")
        fake_rc.plot_rc_fit_summary_panels = recorder("rc_panels")
        self._install_module("src.viz.rc_curves", fake_rc)

        fake_paper = types.ModuleType("src.viz.paper")
        fake_paper.plot_m200_c_summary_comparison = recorder("m200_comparison")
        fake_paper.plot_m200_c_summary_panels = recorder("m200_panels")
        self._install_module("src.viz.paper", fake_paper)

        fake_figure = types.ModuleType("figure")

        def legacy_figure_called(*args, **kwargs):
            raise AssertionError("legacy figure should not be called")

        fake_figure.main = legacy_figure_called
        self._install_module("figure", fake_figure)

        main(
            [
                "figures",
                "--ifu",
                "1000-12701",
                "1001-12702",
                "1002-12703",
                "1003-12704",
                "--output-dir",
                "figure-output",
            ]
        )

        self.assertEqual(
            [name for name, _ in calls],
            [
                "velocity_comparison",
                "m200_comparison",
                "m200_panels",
                "rc_comparison",
                "rc_panels",
                "velocity_panels",
            ],
        )
        self.assertEqual(calls[0][1]["plateifus"], ["1000-12701", "1001-12702", "1002-12703", "1003-12704"])
        self.assertEqual(calls[0][1]["output_path"], Path("figure-output") / "velocity_field_comparison.png")


if __name__ == "__main__":
    unittest.main()
