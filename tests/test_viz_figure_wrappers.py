from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

from src.viz.paper import plot_m200_c_summary_comparison, plot_m200_c_summary_panels
from src.viz.rc_curves import plot_rc_fit_summary_comparison, plot_rc_fit_summary_panels
from src.viz.velocity_maps import plot_velocity_field_comparison, plot_velocity_field_panels


class VizFigureWrapperTests(unittest.TestCase):
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

    def test_wrappers_dispatch_to_current_figure_panels_module(self) -> None:
        calls: list[str] = []

        fake_panels = types.ModuleType("src.viz.figure_panels")

        def recorder(name):
            def _record(*args, **kwargs):
                calls.append(name)
                return kwargs.get("output_path", Path(f"{name}.png"))

            return _record

        fake_panels.plot_m200_c_summary_comparison = recorder("m200_comparison")
        fake_panels.plot_m200_c_summary_panels = recorder("m200_panels")
        fake_panels.plot_rc_fit_summary_comparison = recorder("rc_comparison")
        fake_panels.plot_rc_fit_summary_panels = recorder("rc_panels")
        fake_panels.plot_velocity_field_comparison = recorder("velocity_comparison")
        fake_panels.plot_velocity_field_panels = recorder("velocity_panels")
        self._install_module("src.viz.figure_panels", fake_panels)

        fake_figure = types.ModuleType("figure")

        def legacy_called(*args, **kwargs):
            raise AssertionError("legacy figure should not be called")

        fake_figure.plot_m200_c_summary_comparison = legacy_called
        fake_figure.plot_m200_c_summary_panels = legacy_called
        fake_figure.plot_rc_fit_summary_comparison = legacy_called
        fake_figure.plot_rc_fit_summary_panels = legacy_called
        fake_figure.plot_velocity_field_comparison = legacy_called
        fake_figure.plot_velocity_field_panels = legacy_called
        self._install_module("figure", fake_figure)

        kwargs = {"plateifus": ["a", "b", "c", "d"], "output_path": Path("out.png")}
        plot_velocity_field_comparison(**kwargs)
        plot_velocity_field_panels(**kwargs)
        plot_rc_fit_summary_comparison(**kwargs)
        plot_rc_fit_summary_panels(**kwargs)
        plot_m200_c_summary_comparison(**kwargs)
        plot_m200_c_summary_panels(**kwargs)

        self.assertEqual(
            calls,
            [
                "velocity_comparison",
                "velocity_panels",
                "rc_comparison",
                "rc_panels",
                "m200_comparison",
                "m200_panels",
            ],
        )


if __name__ == "__main__":
    unittest.main()
