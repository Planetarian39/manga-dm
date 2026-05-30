from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import settings as settings_module
import src.models.population as population


class PopulationPipelineTests(unittest.TestCase):
    def tearDown(self) -> None:
        settings_module.init_settings()

    def test_fit_population_loads_data_and_calls_current_model(self) -> None:
        calls: list[tuple[str, object]] = []
        original_get_data = population.get_m200_c_data
        original_fit = population.fit_m200_c_mcmc
        self.addCleanup(setattr, population, "get_m200_c_data", original_get_data)
        self.addCleanup(setattr, population, "fit_m200_c_mcmc", original_fit)

        def fake_get_m200_c_data(**kwargs):
            calls.append(("get_data", kwargs))
            return {
                "M200": np.array([1.0e12, 2.0e12, 3.0e12]),
                "c": np.array([8.0, 7.0, 6.0]),
                "log10_gmm_weights": np.array([[1.0], [1.0], [1.0]], dtype=object),
                "log10_gmm_means": np.array(
                    [
                        [[12.0, 0.9]],
                        [[12.3, 0.85]],
                        [[12.5, 0.8]],
                    ],
                    dtype=object,
                ),
                "log10_gmm_covariances": np.array(
                    [
                        [[[0.1, 0.0], [0.0, 0.1]]],
                        [[[0.1, 0.0], [0.0, 0.1]]],
                        [[[0.1, 0.0], [0.0, 0.1]]],
                    ],
                    dtype=object,
                ),
                "log10_M200_posterior_samples": np.array([None, None, None], dtype=object),
                "log10_c_posterior_samples": np.array([None, None, None], dtype=object),
                "log10_M200_prior_mu": np.array([12.0, 12.3, 12.5]),
                "log10_M200_prior_sigma": np.array([0.3, 0.3, 0.3]),
                "log10_M200_prior_lower": np.array([10.0, 10.0, 10.0]),
                "log10_M200_prior_upper": np.array([14.0, 14.0, 14.0]),
                "log10_c_prior_mu": np.array([0.9, 0.85, 0.8]),
                "log10_c_prior_sigma": np.array([0.2, 0.2, 0.2]),
            }

        def fake_fit_m200_c_mcmc(M200_obs, c_obs, **kwargs):
            calls.append(
                (
                    "fit",
                    {
                        "M200_obs": M200_obs,
                        "c_obs": c_obs,
                        **kwargs,
                    },
                )
            )
            return {"dataset_label": kwargs["dataset_label"]}

        population.get_m200_c_data = fake_get_m200_c_data
        population.fit_m200_c_mcmc = fake_fit_m200_c_mcmc

        with tempfile.TemporaryDirectory() as tmp:
            result = population.fit_m200_c_population(
                quality_cut="strict",
                result_dir_override=tmp,
                sample_cap=25,
            )

        get_call = calls[0][1]
        self.assertEqual(get_call["quality_cut"], "strict")
        self.assertEqual(Path(get_call["result_dir_override"]), Path(tmp))
        self.assertEqual(result["dataset_label"], "all")

        fit_call = calls[1][1]
        np.testing.assert_allclose(fit_call["M200_obs"], [1.0e12, 2.0e12, 3.0e12])
        np.testing.assert_allclose(fit_call["c_obs"], [8.0, 7.0, 6.0])
        self.assertEqual(fit_call["sample_cap"], 25)
        self.assertEqual(fit_call["dataset_label"], "all")
        self.assertIsNotNone(fit_call["log10_gmm_weights"])
        self.assertIsNone(fit_call["log10_M200_posterior_samples"])


if __name__ == "__main__":
    unittest.main()
