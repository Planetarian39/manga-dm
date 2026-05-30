from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.viz.posterior import annotate_pair_marginals, plot_population_inference_diagnostics


class VizPosteriorTests(unittest.TestCase):
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

    def test_annotate_pair_marginals_does_not_import_legacy_dm(self) -> None:
        fake_dm = types.ModuleType("dm")

        def legacy_called(*args, **kwargs):
            raise AssertionError("legacy dm should not be called")

        fake_dm._annotate_pair_marginals = legacy_called
        self._install_module("dm", fake_dm)

        fig, axes = plt.subplots(2, 2)
        self.addCleanup(plt.close, fig)

        class Values:
            values = np.array([1.0, 2.0, 3.0])

        annotate_pair_marginals(
            axes,
            {"M200": Values(), "c": Values()},
            ["M200", "c"],
            plot_median_line=True,
        )

        self.assertIn("M_{200}", axes[0, 0].get_title())

    def test_population_diagnostics_writes_current_plot_without_legacy_m200(self) -> None:
        fake_m200 = types.ModuleType("m200")

        def legacy_called(*args, **kwargs):
            raise AssertionError("legacy m200 should not be called")

        fake_m200.plot_population_inference_diagnostics = legacy_called
        self._install_module("m200", fake_m200)

        with tempfile.TemporaryDirectory() as tmp:
            output = plot_population_inference_diagnostics(
                M200=np.array([1.0e12, 2.0e12, 3.0e12]),
                c=np.array([8.0, 7.0, 6.0]),
                fit_results={
                    "log10_c0_median": 0.9,
                    "alpha_median": -0.1,
                    "sigma_int_median": 0.2,
                },
                plot_suffix="_test",
                output_dir=tmp,
            )

        self.assertEqual(output, Path(tmp) / "c-M_relation_diagnostics_test.png")


if __name__ == "__main__":
    unittest.main()
