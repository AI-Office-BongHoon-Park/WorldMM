#!/usr/bin/env python3
"""Render the final WorldMM summary HTML deck into a six-slide PPTX."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.util import Inches


@dataclass
class SlideSpec:
    html_path: Path
    label: str
    viewport_w: int = 1280
    viewport_h: int = 720
    wait_ms: int = 500


SLIDE_W_INCHES = 13.333
SLIDE_H_INCHES = 7.5


def launch_chromium(playwright):
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        executable = shutil.which(name)
        if executable:
            return playwright.chromium.launch(executable_path=executable)
    return playwright.chromium.launch()


def discover_slides(repo: Path) -> List[SlideSpec]:
    slides_dir = repo / "docs" / "slides"
    filenames = ["worldmm-final-summary.html"] + [f"worldmm-final-case-{idx:02d}.html" for idx in range(1, 6)]
    specs: List[SlideSpec] = []
    for filename in filenames:
        html_path = slides_dir / filename
        if not html_path.exists():
            raise FileNotFoundError(f"Missing final deck slide: {html_path}")
        specs.append(SlideSpec(html_path, html_path.stem))
    return specs


def render_to_png(specs: List[SlideSpec], tmpdir: Path) -> List[Path]:
    out: List[Path] = []
    with sync_playwright() as playwright:
        browser = launch_chromium(playwright)
        for index, spec in enumerate(specs):
            page = browser.new_page(viewport={"width": spec.viewport_w, "height": spec.viewport_h}, device_scale_factor=1)
            page.goto(f"file://{spec.html_path.resolve()}", wait_until="networkidle")
            page.wait_for_timeout(spec.wait_ms)
            png = tmpdir / f"slide_{index:02d}_{spec.label}.png"
            page.screenshot(path=str(png), clip={"x": 0, "y": 0, "width": 1280, "height": 720})
            page.close()
            out.append(png)
            print(f"  rendered  {spec.label:<30s} -> {png.name}")
        browser.close()
    return out


def assemble_pptx(pngs: List[Path], out_pptx: Path) -> None:
    pres = Presentation()
    pres.slide_width = Inches(SLIDE_W_INCHES)
    pres.slide_height = Inches(SLIDE_H_INCHES)
    blank_layout = pres.slide_layouts[6]
    for png in pngs:
        slide = pres.slides.add_slide(blank_layout)
        slide.shapes.add_picture(str(png), Inches(0), Inches(0), width=Inches(SLIDE_W_INCHES), height=Inches(SLIDE_H_INCHES))
    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    pres.save(str(out_pptx))
    size_mb = out_pptx.stat().st_size / (1024 * 1024)
    print(f"wrote {out_pptx} ({len(pngs)} slides · {size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("docs/slides/pptx/worldmm_final_deck.pptx"))
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    specs = discover_slides(repo)
    if args.keep_tmp:
        tmpdir = repo / ".tmp_worldmm_final_deck"
        tmpdir.mkdir(exist_ok=True)
        pngs = render_to_png(specs, tmpdir)
        assemble_pptx(pngs, repo / args.out)
        print(f"kept screenshots in {tmpdir}")
        return

    with tempfile.TemporaryDirectory(prefix="worldmm_final_deck_") as tmp:
        pngs = render_to_png(specs, Path(tmp))
        assemble_pptx(pngs, repo / args.out)


if __name__ == "__main__":
    main()
