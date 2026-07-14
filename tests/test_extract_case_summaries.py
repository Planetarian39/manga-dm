import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "docs" / "extract_case_summaries.py"


class ExtractCaseSummariesTests(unittest.TestCase):
    def test_exports_quantiles_sample_count_and_correlation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_dir = temp / "input"
            input_dir.mkdir()
            source = input_dir / "11743-9102_nfw_param_cm200_samples.nc"
            with h5py.File(source, "w") as dataset:
                dataset.attrs["plate_ifu"] = "11743-9102"
                dataset.create_dataset("log10_M200_samples", data=np.array([11.0, 12.0, 13.0]))
                dataset.create_dataset("log10_c_samples", data=np.array([1.2, 1.0, 0.8]))
                dataset.create_dataset("sample_count", data=3)

            output = temp / "summaries.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-dir",
                    str(input_dir),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schemaVersion"], 1)
            self.assertEqual(len(payload["cases"]), 1)
            case = payload["cases"][0]
            self.assertEqual(case["galaxyId"], "11743-9102")
            self.assertEqual(case["sampleCount"], 3)
            self.assertEqual(case["log10M200"]["median"], 12.0)
            self.assertEqual(case["log10C"]["median"], 1.0)
            self.assertEqual(case["correlation"], -1.0)

    def test_committed_summary_matches_allowlisted_posteriors(self):
        input_dir = ROOT / "docs" / "public" / "downloads" / "posteriors"
        committed = ROOT / "docs" / "public" / "meta" / "case-study-summaries.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            generated = Path(temp_dir) / "case-study-summaries.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-dir",
                    str(input_dir),
                    "--output",
                    str(generated),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(generated.read_text(encoding="utf-8")),
                json.loads(committed.read_text(encoding="utf-8")),
            )


if __name__ == "__main__":
    unittest.main()
