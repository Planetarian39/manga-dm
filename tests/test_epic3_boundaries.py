from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_tree(module_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / module_path).read_text(encoding="utf-8"))


def _defined_functions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _imported_names_from(tree: ast.Module, module_name: str) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            names.update(alias.name for alias in node.names)
    return names


class Epic3BoundaryTests(unittest.TestCase):
    def test_population_no_longer_defines_plot_or_selection_functions(self) -> None:
        tree = _module_tree("src/models/population.py")
        functions = _defined_functions(tree)

        self.assertFalse(
            [name for name in functions if name.startswith("plot_")],
            "plot functions should live in src.viz",
        )
        self.assertNotIn("generate_robustness_sample", functions)
        self.assertFalse(
            [name for name in functions if name.startswith("_filter_dataframe_by_")],
            "sample filters should live in src.pipeline.selection",
        )

    def test_population_does_not_import_pandas_xarray_or_viz(self) -> None:
        imports = _imported_modules(_module_tree("src/models/population.py"))

        self.assertNotIn("pandas", imports)
        self.assertNotIn("xarray", imports)
        self.assertNotIn("matplotlib.pyplot", imports)
        self.assertFalse(
            [module for module in imports if module.startswith("src.viz")],
            "models must not depend on the visualization layer",
        )

    def test_data_selection_and_viz_own_epic3_responsibilities(self) -> None:
        catalog_functions = _defined_functions(_module_tree("src/data/catalog.py"))
        results_functions = _defined_functions(_module_tree("src/data/results.py"))
        selection_functions = _defined_functions(_module_tree("src/pipeline/selection.py"))
        paper_functions = _defined_functions(_module_tree("src/viz/paper.py"))
        posterior_functions = _defined_functions(_module_tree("src/viz/posterior.py"))

        self.assertIn("get_m200_c_data", results_functions)
        self.assertIn("build_base_sample_catalog", catalog_functions)
        self.assertIn("generate_robustness_sample", selection_functions)
        self.assertIn("_filter_dataframe_by_success", selection_functions)
        self.assertIn("plot_m200_c_relation_all", paper_functions)
        self.assertIn("plot_sample_attrition_pipeline", paper_functions)
        self.assertIn("plot_population_posterior_diagnostics", posterior_functions)

    def test_population_posterior_uses_model_prior_constants(self) -> None:
        imports = _imported_names_from(
            _module_tree("src/viz/posterior.py"),
            "src.config.constants",
        )

        self.assertTrue(
            {
                "LOG10_C0_PRIOR_MEAN",
                "LOG10_C0_PRIOR_SIGMA",
                "ALPHA_PRIOR_MEAN",
                "ALPHA_PRIOR_SIGMA",
            }.issubset(imports),
            "posterior diagnostics must use the same population prior defaults as the model",
        )


if __name__ == "__main__":
    unittest.main()
