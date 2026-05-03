"""FITS file I/O, download, and checksum utilities.

Migrated from ``src-orig/util/fits_util.py``.  URL constants are imported
from ``src.config.constants``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table

from src.config.constants import MAPS_BASE_URL, REDUX_BASE_URL, FIREFLY_BASE_URL


class FitsUtil:
    data_dir: Path
    drp_dir: Path
    dap_dir: Path
    images_dir: Path
    firefly_dir: Path

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.drp_dir = self.data_dir / "redux"
        self.dap_dir = self.data_dir / "analysis"
        self.images_dir = self.data_dir / "images"
        self.firefly_dir = self.data_dir / "firefly"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.drp_dir.mkdir(parents=True, exist_ok=True)
        self.dap_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # ── File locators ─────────────────────────────────────────────────

    def get_maps_file(self, plateifu: str, checksum: bool = False, download: bool = True) -> Path:
        """Return the local path to the MAPS FITS file for *plateifu*.

        Downloads the file if missing (unless *download* is False).
        Optionally verifies SHA256 *checksum*.
        """
        plateifu = plateifu.strip()
        if "-" not in plateifu:
            raise ValueError("plateifu must be in 'plate-ifu' format, e.g. '8550-12704'")
        plate, ifu = plateifu.split("-", 1)
        filename = f"manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz"
        ret_path = self.dap_dir / filename

        if ret_path.exists() and ret_path.with_suffix('.sha256').exists():
            if not checksum:
                return ret_path

            sha256_checksum = self._compute_sha256(ret_path)
            checksum_file = ret_path.with_suffix('.sha256')
            with open(checksum_file, 'r', encoding='utf-8') as cf:
                line = cf.readline().strip()

            parts = line.split(maxsplit=1)
            stored_checksum = parts[0] if parts else ""
            stored_name = ""
            if len(parts) > 1:
                stored_name = parts[1].lstrip("*").strip()

            if stored_name and stored_name != filename:
                print(f"Checksum file name mismatch for {ret_path}: expected {filename}, got {stored_name}; re-downloading.")
            elif stored_checksum and sha256_checksum == stored_checksum:
                print(f"MAPS file checksum success: {ret_path}")
                return ret_path
            else:
                print(f"Checksum mismatch for {ret_path}; re-downloading.")

        if not download:
            return None

        print(f" Warning: MAPS file {filename} need to be downloaded.")
        try:
            if ret_path.exists():
                ret_path.unlink()
            if ret_path.with_suffix('.sha256').exists():
                ret_path.with_suffix('.sha256').unlink()
        except Exception:
            pass

        dl_success = self.dl_maps(plateifu, filename)
        if not dl_success:
            raise FileNotFoundError(f"Unable to obtain MAPS file: {filename}")

        sha256_checksum = self._compute_sha256(ret_path)
        checksum_file = ret_path.with_suffix('.sha256')
        with open(checksum_file, 'w', encoding='utf-8', newline='\n') as cf:
            cf.write(f"{sha256_checksum} *{filename}\n")
        return ret_path

    def get_drpall_file(self) -> Path:
        """Return the local path to the DRPALL FITS file, downloading if needed."""
        filename = "drpall-v3_1_1.fits"
        ret_path = self.drp_dir / filename
        if not ret_path.exists():
            print(f" Warning: drpall file {filename} need to be downloaded first.")
            self.dl_drpall(filename)
        return ret_path

    def get_firefly_file(self) -> Path:
        """Return the local path to the Firefly MASTAR FITS file."""
        filename = "manga-firefly-v3_1_1-mastar.fits"
        ret_path = self.firefly_dir / filename
        if not ret_path.exists():
            print(f" Warning: firefly file {filename} need to be downloaded first.")
            self.dl_firefly_mastar(filename)
        return ret_path

    def get_image_file(self, plateifu: str) -> Path:
        """Return the local path to the galaxy image PNG, downloading if needed."""
        plateifu = plateifu.strip()
        if "-" not in plateifu:
            raise ValueError("plateifu must be in 'plate-ifu' format, e.g. '7957-3701'")
        plate, ifu = plateifu.split("-", 1)
        filename = f"manga-{plate}-{ifu}.png"
        ret_path = self.images_dir / filename

        if not ret_path.exists():
            print(f" Warning: image file {filename} need to be downloaded first.")
            dl_success = self.dl_image(plateifu, filename)
            if not dl_success:
                raise FileNotFoundError(f"Unable to obtain image file: {filename}")
        return ret_path

    # ── Downloaders ───────────────────────────────────────────────────

    def dl_drpall(self, filename: str) -> bool:
        url = f"{REDUX_BASE_URL}/{filename}"
        return self._download_file(url, self.drp_dir / filename, file_type_str="DRPALL")

    def dl_firefly_mastar(self, filename: str) -> bool:
        url = f"{FIREFLY_BASE_URL}/{filename}"
        return self._download_file(url, self.firefly_dir / filename, file_type_str="FIREFLY MASTAR")

    def dl_image(self, plateifu: str, filename: str) -> bool:
        plateifu = plateifu.strip()
        if "-" not in plateifu:
            raise ValueError("plateifu must be in 'plate-ifu' format, e.g. '7957-3701'")
        plate, ifu = plateifu.split("-", 1)
        if not filename:
            filename = f"manga-{plate}-{ifu}.png"
        url = f"{REDUX_BASE_URL}/{plate}/images/{ifu}.png"
        return self._download_file(url, self.images_dir / filename, file_type_str="image")

    def dl_maps(self, plateifu: str, filename: str) -> bool:
        plateifu = plateifu.strip()
        if "-" not in plateifu:
            raise ValueError("plateifu must be in 'plate-ifu' format, e.g. '7443-12703'")
        plate, ifu = plateifu.split("-", 1)
        if not filename:
            filename = f"manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARHC2.fits.gz"
        url = f"{MAPS_BASE_URL}/{plate}/{ifu}/{filename}"
        return self._download_file(url, self.dap_dir / filename, file_type_str="MAPS")

    def find_galaxies(self) -> list[str]:
        """Find galaxies that meet criteria from DRPALL file (placeholder)."""
        drpall_file = self.get_drpall_file()
        with fits.open(drpall_file) as hdul:
            table = Table(hdul[1].data)
            plateifus = table['PLATEIFU']
            return [str(pifu) for pifu in plateifus]

    # ── Private helpers ───────────────────────────────────────────────

    def _download_file(self, url: str, target_path: Path, file_type_str: str = "file") -> bool:
        if target_path.exists():
            print(f"{file_type_str} already exists: {target_path}")
            return True

        try:
            print(f"Downloading {file_type_str} from {url}...")
            resp = requests.get(url, stream=True, timeout=60)
        except requests.RequestException as exc:
            print(f"Request for {file_type_str} failed: {exc}")
            return False

        if resp.status_code != 200:
            print(f"Download of {file_type_str} failed HTTP {resp.status_code}: {url}")
            return False

        try:
            with open(target_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except OSError as exc:
            print(f"Failed to write {file_type_str}: {exc}")
            try:
                if target_path.exists():
                    target_path.unlink()
            except Exception:
                pass
            return False

        print(f"{file_type_str} download complete: {target_path}")
        return True

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA256 checksum of a file and return the hex digest."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except OSError as exc:
            raise IOError(f"Unable to read file for sha256 computation: {path}: {exc}") from exc
        return h.hexdigest()
