"""Reproducible download of GSE109169, the GPL5175 annotation and gene-set libraries.

Every downloaded file is recorded in ``data/raw/provenance.json`` with its source URL,
SHA-256 checksum, size and retrieval time, so a later run can verify it used the same inputs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from oncogene.config import Paths, load_config

log = logging.getLogger("oncogene.data.download")

SERIES_MATRIX_FILE = "GSE109169_series_matrix.txt.gz"
PLATFORM_TABLE_FILE = "GPL5175_platform_table.txt.gz"
PROVENANCE_FILE = "provenance.json"
USER_AGENT = "oncogene-explorer/0.1 (academic research pipeline)"


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, dest: Path, *, gzip_output: bool = False, retries: int = 3,
           timeout: int = 120) -> None:
    """Stream ``url`` to ``dest`` atomically, optionally gzip-compressing on the fly."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout,
                              headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                opener = gzip.open if gzip_output else open
                with opener(tmp, "wb") as out:
                    for block in r.iter_content(chunk_size=1 << 20):
                        out.write(block)
            shutil.move(tmp, dest)
            return
        except (requests.RequestException, OSError) as err:
            last_err = err
            log.warning("Download attempt %d/%d failed for %s: %s", attempt, retries, url, err)
            time.sleep(2 * attempt)
    if tmp.exists():
        tmp.unlink()
    raise RuntimeError(f"Failed to download {url}") from last_err


def _record(prov: dict[str, Any], name: str, url: str, path: Path) -> None:
    prov[name] = {
        "url": url,
        "file": path.name,
        "sha256": sha256sum(path),
        "bytes": path.stat().st_size,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def download_all(config: dict[str, Any] | None = None, paths: Paths | None = None,
                 force: bool = False) -> dict[str, Any]:
    """Download every raw input. Existing files are kept unless ``force`` is set."""
    cfg = config or load_config()
    paths = (paths or Paths()).ensure()
    ds = cfg["dataset"]
    prov_path = paths.raw / PROVENANCE_FILE
    prov: dict[str, Any] = json.loads(prov_path.read_text()) if prov_path.exists() else {}

    jobs = [
        ("series_matrix", ds["series_matrix_url"], paths.raw / SERIES_MATRIX_FILE, False),
        # The platform table is ~87 MB uncompressed; store it gzipped.
        ("platform_table", ds["platform_table_url"], paths.raw / PLATFORM_TABLE_FILE, True),
    ]
    for lib in ds.get("gene_set_libraries", []):
        jobs.append((f"geneset_{lib}", ds["gene_set_url_template"].format(library=lib),
                     paths.raw / f"{lib}.gmt", False))

    for name, url, dest, gz in jobs:
        if dest.exists() and not force:
            log.info("Using cached %s (%s)", name, dest.name)
            if name not in prov:
                _record(prov, name, url, dest)
            continue
        log.info("Downloading %s from %s", name, url)
        _fetch(url, dest, gzip_output=gz)
        _record(prov, name, url, dest)
        log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)

    prov_path.write_text(json.dumps(prov, indent=2))
    log.info("Provenance written to %s", prov_path)
    return prov
