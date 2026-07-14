from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "scripts"
    / "mcmc"
    / "gen_prior_figures.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_prior_figures", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class McmcPriorFigureGeneratorTests(unittest.TestCase):
    def test_generator_is_import_safe_and_invocation_independent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            previous_cwd = Path.cwd()
            os.chdir(temp_path)
            try:
                before = set(temp_path.iterdir())
                generator = _load_generator()
                self.assertEqual(set(temp_path.iterdir()), before)
                self.assertEqual(generator.format_head_count(1), "1 head")
                self.assertEqual(generator.format_head_count(2), "2 heads")

                standalone_dir = temp_path / "standalone"
                all_dir = temp_path / "all"
                standalone = generator.generate_posterior_by_sample_size(
                    standalone_dir, seed=42
                )
                generated = generator.generate_all(all_dir, seed=42)
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(
                [path.name for path in generated],
                [
                    "posterior-by-sample-size.png",
                    "posterior-mean-by-sample-size.png",
                    "prior-posterior-small-large-data.png",
                ],
            )
            self.assertEqual(_sha256(standalone), _sha256(generated[0]))
            self.assertTrue(all(path.stat().st_size > 0 for path in generated))


if __name__ == "__main__":
    unittest.main()
