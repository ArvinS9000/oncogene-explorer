"""Capture dashboard screenshots (and optionally a walkthrough video) with Playwright.

Uses an installed Chrome/Edge (no browser download):
    pip install playwright
    python scripts/capture_screenshots.py [--video] [--channel chrome|msedge]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
DEMO = ROOT / "docs" / "demo"
PORT = 8599
PAGES = [
    ("", "1_overview"),
    ("dataset_explorer", "2_dataset_explorer"),
    ("gene_expression", "3_gene_expression"),
    ("models", "4_models"),
    ("gene_importance", "5_gene_importance"),
    ("pathways", "6_pathways"),
    ("findings", "7_findings"),
]


def wait_for_server(url: str, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Streamlit did not start at {url}")


def trim_bottom(path: Path, pad: int = 48) -> None:
    """Crop trailing rows that contain only the page background."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    a = np.asarray(img)[:, 320:, :].astype(int)          # ignore the sidebar
    bg = a[-1, -1]
    content = np.flatnonzero((np.abs(a - bg).sum(axis=2) > 12).any(axis=1))
    if content.size:
        img.crop((0, 0, img.width, min(img.height, content[-1] + pad))).save(path)


def settle(page, extra_ms: int = 2500) -> None:
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=60000)
    # Wait until Streamlit has finished running the script (no 'running' status widget).
    for _ in range(120):
        if page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(extra_ms)


def main() -> None:
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true", help="also record a walkthrough video (.webm)")
    ap.add_argument("--channel", default="chrome")
    args = ap.parse_args()
    SHOTS.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
                             "--server.port", str(PORT), "--server.headless", "true"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://localhost:{PORT}"
    try:
        wait_for_server(base)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel, headless=True)
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
            page = ctx.new_page()
            for path, name in PAGES:
                page.set_viewport_size({"width": 1440, "height": 900})
                page.goto(f"{base}/{path}")
                settle(page, 4000)
                # Streamlit scrolls inside its own container, so grow the viewport to the content height.
                js = "document.querySelector('[data-testid=stMain]')?.scrollHeight || 900"
                h, prev = page.evaluate(js), -1
                for _ in range(20):  # wait until lazy charts/tables stop changing the height
                    if h == prev:
                        break
                    page.wait_for_timeout(800)
                    prev, h = h, page.evaluate(js)
                page.set_viewport_size({"width": 1440, "height": int(min(max(h + 700, 1500), 6000))})
                page.wait_for_timeout(2500)
                errors = page.locator("[data-testid='stException'], [data-testid='stAlertContentError']").count()
                out = SHOTS / f"{name}.png"
                page.screenshot(path=str(out), full_page=True)
                trim_bottom(out)
                print(f"{name}: saved ({'ERRORS ON PAGE: ' + str(errors) if errors else 'no errors'})")
            ctx.close()

            if args.video:
                DEMO.mkdir(parents=True, exist_ok=True)
                vctx = browser.new_context(viewport={"width": 1280, "height": 760},
                                           record_video_dir=str(DEMO), record_video_size={"width": 1280, "height": 760})
                vp = vctx.new_page()
                for path, _ in PAGES:
                    vp.goto(f"{base}/{path}")
                    settle(vp, 2500)
                    height = vp.evaluate("document.querySelector('[data-testid=stMain]')?.scrollHeight || 3000")
                    main = vp.locator("[data-testid='stMain']")
                    for y in range(0, min(int(height), 4200), 380):
                        main.evaluate(f"el => el.scrollTo({{top: {y}, behavior: 'smooth'}})")
                        vp.wait_for_timeout(650)
                    vp.wait_for_timeout(800)
                video = vp.video.path()
                vctx.close()
                out = DEMO / "oncogene_explorer_walkthrough.webm"
                shutil.move(video, out)
                print("video:", out)
            browser.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
