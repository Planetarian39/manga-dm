from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.data.catalog import get_plateifu_list
from src.data.fits import FitsUtil
from src.data.results import get_processed_plate_ifus, store_params_file


class DataIoTests(unittest.TestCase):
    def test_missing_plateifu_list_does_not_fall_back_to_test_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.txt"
            self.assertEqual(get_plateifu_list(missing), [])

    def test_fits_util_creates_firefly_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            util = FitsUtil(Path(tmp))
            self.assertTrue(util.firefly_dir.is_dir())

    def test_failed_firefly_download_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            util = FitsUtil(Path(tmp))
            util.dl_firefly_mastar = lambda filename: False

            with self.assertRaises(FileNotFoundError):
                util.get_firefly_file()

    def test_maps_file_without_sidecar_is_valid_when_checksum_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            util = FitsUtil(Path(tmp))
            maps_file = (
                util.dap_dir
                / "manga-8550-12704-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz"
            )
            maps_file.touch()

            self.assertEqual(
                util.get_maps_file("8550-12704", checksum=False, download=False),
                maps_file,
            )

    def test_processed_ifus_require_success_and_sample_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            pd.DataFrame.from_dict(
                {
                    "1000-10001": {"result": "success"},
                    "2000-20002": {"result": "failure"},
                    "3000-30003": {"result": "success"},
                },
                orient="index",
            ).to_csv(result_dir / "nfw.csv")
            (result_dir / "1000-10001_samples.nc").touch()
            (result_dir / "2000-20002_samples.nc").touch()

            self.assertEqual(
                get_processed_plate_ifus(
                    "nfw.csv",
                    result_dir,
                    successful_only=True,
                    required_sample_filename="samples.nc",
                ),
                {"1000-10001"},
            )

    def test_locked_csv_updates_keep_all_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock = threading.Lock()

            def write_row(index: int) -> None:
                store_params_file(
                    f"{index:04d}-{index:05d}",
                    {"result": "success", "value": index},
                    "params.csv",
                    tmp,
                    write_lock=lock,
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(write_row, range(30)))

            stored = pd.read_csv(Path(tmp) / "params.csv", index_col=0)
            self.assertEqual(len(stored), 30)


if __name__ == "__main__":
    unittest.main()
