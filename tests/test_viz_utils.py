from __future__ import annotations

import sys
import unittest

import numpy as np

from src.viz.utils import PlotUtil, plot_posterior_1d_hdi


class VizUtilsTests(unittest.TestCase):
    def test_plot_posterior_1d_hdi_is_current_implementation(self) -> None:
        sys.modules.pop("plot_util", None)

        fig, ax, stats = plot_posterior_1d_hdi(
            np.array([1.0, 2.0, 3.0, 4.0]),
            annotate=False,
            show_legend=False,
        )
        self.addCleanup(fig.clf)

        self.assertEqual(stats["point_label"], "Median")
        self.assertNotIn("plot_util", sys.modules)
        self.assertIsNotNone(ax)

    def test_plotutil_is_available_from_current_viz_module(self) -> None:
        class FakeFitsUtil:
            pass

        plot_util = PlotUtil(FakeFitsUtil())
        self.assertIsInstance(plot_util, PlotUtil)


if __name__ == "__main__":
    unittest.main()
