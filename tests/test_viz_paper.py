from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

from src.viz.paper import plot_m200_c_relation_all, plot_sample_attrition_pipeline


class VizPaperTests(unittest.TestCase):
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

    def test_sample_attrition_lives_in_viz_paper_not_legacy_m200(self) -> None:
        fake_m200 = types.ModuleType("m200")

        def legacy_called(*args, **kwargs):
            raise AssertionError("legacy m200 should not be called")

        fake_m200.plot_sample_attrition_pipeline = legacy_called
        self._install_module("m200", fake_m200)

        self.assertEqual(plot_sample_attrition_pipeline.__module__, "src.viz.paper")

    def test_m200_relation_plot_does_not_import_legacy_m200(self) -> None:
        fake_m200 = types.ModuleType("m200")

        def legacy_called(*args, **kwargs):
            raise AssertionError("legacy m200 should not be called")

        fake_m200.plot_m200_c_relation_all = legacy_called
        self._install_module("m200", fake_m200)

        import tempfile
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            output = plot_m200_c_relation_all(
                np.array([1.0e12, 2.0e12, 3.0e12]),
                np.array([8.0, 7.0, 6.0]),
                fit_results={
                    "log10_c0_median": 0.9,
                    "alpha_median": -0.1,
                    "sigma_int_median": 0.2,
                },
                output_dir=tmp,
            )

        self.assertEqual(output, Path(tmp) / "c-M_relation_all.png")


if __name__ == "__main__":
    unittest.main()
