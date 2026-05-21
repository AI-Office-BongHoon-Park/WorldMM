#!/usr/bin/env python3
"""Render every WorldMM HTML slide into a single .pptx via headless chromium.

Each input HTML is loaded at 1280x720 (1920x1080 for the place-graph since it
benefits from extra room), screenshotted to PNG, and dropped as a full-bleed
image onto one 16:9 slide. The result is a self-contained .pptx that mirrors
the spatial-hero V3 deck plus the capstone and the interactive D3 place graph.
"""

from __future__ import annotations

import argparse
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from playwright.sync_api import sync_playwright


def launch_chromium(playwright):
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        executable = shutil.which(name)
        if executable:
            return playwright.chromium.launch(executable_path=executable)
    return playwright.chromium.launch()
from pptx import Presentation
from pptx.util import Inches


@dataclass
class SlideSpec:
    html_path: Path
    label: str
    viewport_w: int = 1400
    viewport_h: int = 800
    clip_w: int = 1330
    clip_h: int = 770
    wait_ms: int = 700


SLIDE_W_INCHES = 13.333
SLIDE_H_INCHES = 7.5


def discover_slides(repo: Path) -> List[SlideSpec]:
    slides_dir = repo / "docs" / "slides"
    capstone = slides_dir / "spatial-capstone-v3.html"
    visual_axis = slides_dir / "visual-axis-evolution.html"
    gs_mock = slides_dir / "spatial-retrieval-gs-mock.html"
    place_graph = slides_dir / "place-graph-A1_JAKE-DAY1.html"
    aea_place_graph = slides_dir / "place-graph-AEA-loc5_script4_seq6_rec1.html"
    case_files_in_order = [
        "spatial-hero-SH-A-001.html",
        "spatial-hero-SH-B-002.html",
        "spatial-hero-SH-C-006.html",
        "spatial-hero-SH-E-004.html",
        "spatial-hero-SH-F-005.html",
        "spatial-hero-SH-A-002.html",
        "spatial-hero-SH-B-001.html",
        "spatial-hero-SH-B-005.html",
        "spatial-hero-SH-B-008.html",
    ]
    order: List[SlideSpec] = []
    if capstone.exists():
        order.append(SlideSpec(capstone, "capstone", 1400, 880, 1330, 860))
    if visual_axis.exists():
        order.append(SlideSpec(visual_axis, "visual-axis-evolution"))
    if gs_mock.exists():
        order.append(SlideSpec(gs_mock, "spatial-retrieval-gs-mock"))
    if place_graph.exists():
        order.append(SlideSpec(place_graph, "place-graph", 1920, 1080, 1880, 1040, wait_ms=1800))
    if aea_place_graph.exists():
        order.append(SlideSpec(aea_place_graph, "aea-place-graph", 1920, 1080, 1880, 1040, wait_ms=1800))
    case_count = 0
    for name in case_files_in_order:
        p = slides_dir / name
        if p.exists():
            order.append(SlideSpec(p, name.replace(".html", "")))
            case_count += 1
    if case_count != len(case_files_in_order):
        raise FileNotFoundError(f"Expected {len(case_files_in_order)} spatial-hero slides, found {case_count}.")
    return order


def render_to_png(specs: List[SlideSpec], tmpdir: Path) -> List[Path]:
    out: List[Path] = []
    with sync_playwright() as p:
        browser = launch_chromium(p)
        for i, s in enumerate(specs):
            page = browser.new_page(viewport={"width": s.viewport_w, "height": s.viewport_h})
            page.goto(f"file://{s.html_path.resolve()}")
            page.wait_for_timeout(s.wait_ms)
            png = tmpdir / f"slide_{i:02d}_{s.label}.png"
            page.screenshot(path=str(png), clip={"x": 0, "y": 0, "width": s.clip_w, "height": s.clip_h})
            page.close()
            out.append(png)
            print(f"  rendered  {s.label:<30s}  →  {png.name}")
        browser.close()
    return out


def assemble_pptx(pngs: List[Path], out_pptx: Path) -> None:
    pres = Presentation()
    pres.slide_width = Inches(SLIDE_W_INCHES)
    pres.slide_height = Inches(SLIDE_H_INCHES)
    blank_layout = pres.slide_layouts[6]

    for png in pngs:
        slide = pres.slides.add_slide(blank_layout)
        slide.shapes.add_picture(
            str(png),
            Inches(0),
            Inches(0),
            width=Inches(SLIDE_W_INCHES),
            height=Inches(SLIDE_H_INCHES),
        )

    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    pres.save(str(out_pptx))
    size_mb = out_pptx.stat().st_size / (1024 * 1024)
    print(f"\nwrote {out_pptx}  ({len(pngs)} slides · {size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("docs/slides/pptx/worldmm_spatial_deck.pptx"))
    parser.add_argument("--keep-tmp", action="store_true", help="Do not delete the temporary PNG directory.")
    args = parser.parse_args()

    specs = discover_slides(args.repo)
    if not specs:
        raise SystemExit("No slide files found under docs/slides/. Aborting.")
    print(f"Discovered {len(specs)} slides.")
    for s in specs:
        print(f"  - {s.html_path.relative_to(args.repo)}")

    tmpdir = Path(tempfile.mkdtemp(prefix="worldmm_slides_"))
    print(f"\nRendering PNGs into {tmpdir}")
    pngs = render_to_png(specs, tmpdir)
    print()
    assemble_pptx(pngs, args.out)
    if not args.keep_tmp:
        for p in pngs:
            p.unlink(missing_ok=True)
        try:
            tmpdir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
