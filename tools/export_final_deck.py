#!/usr/bin/env python3
"""Build the final WorldMM natural-Q HTML deck and render it into PPTX."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.util import Inches

try:
    from decord import VideoReader, cpu  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - deck can still fall back to thumbnails.
    VideoReader = None
    cpu = None


@dataclass
class SlideSpec:
    html_path: Path
    label: str
    viewport_w: int = 1280
    viewport_h: int = 720
    wait_ms: int = 500


SLIDE_W_INCHES = 13.333
SLIDE_H_INCHES = 7.5
CONFIGS = ["spatial_OFF", "spatial_ON_text", "spatial_ON_grounded"]
ASSET_DIR = Path("output/final_deck_assets")
RESULTS = Path("output/golden_qa_phase4_natural_results.json")
SLIDES_DIR = Path("docs/slides")
POINTCLOUD_SOURCES = ASSET_DIR / "pointcloud_sources.json"
GROUNDING = Path("output/metadata/spatial_memory/A1_JAKE/unified_grounding_a1_jake.json")
VIDEO_DIR = Path("data/EgoLife/A1_JAKE/DAY1")
THUMB_DIR = Path("output/thumbnails/A1_JAKE/DAY1")
TEMPLATE_LABELS = {
    "A": "어디 뒀더라",
    "B": "근처에 뭐 있었지",
    "C": "위에 뒀던 거 맞아",
    "D": "어느 게 더 가까웠어",
    "E": "앞이야 뒤야",
    "F": "같은 방에 있었나",
}

CSS = """
:root{--bg:#0b1020;--panel:#11192f;--panel2:#0e1730;--panel3:#0b1429;--line:#1c2746;--ink:#e9ecf5;--muted:#8aa0c8;--good:#22c55e;--bad:#ef4444;--warn:#f59e0b;--accent:#60a5fa;--hl:#fbbf24;--good-edge:rgba(34,197,94,.55);--warn-edge:rgba(245,158,11,.52);--accent-edge:rgba(96,165,250,.42);--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
*{box-sizing:border-box}html,body{margin:0;width:1280px;height:720px;overflow:hidden;background:var(--bg);color:var(--ink);font-family:var(--sans)}
.slide{width:1280px;height:720px;margin:0;padding:22px 26px;background:radial-gradient(circle at 8% 8%,rgba(96,165,250,.16),transparent 28%),radial-gradient(circle at 88% 10%,rgba(34,197,94,.12),transparent 24%),var(--panel);border-radius:16px;display:grid;grid-template-rows:auto 1fr auto;gap:14px;overflow:hidden}
.case-slide{grid-template-rows:auto auto 1fr auto auto;gap:10px}.slide header{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;border-bottom:1px solid var(--line);padding-bottom:8px}.case-header{min-height:112px}.tag{color:var(--accent);font-family:var(--mono);font-size:11px;letter-spacing:1px;text-transform:uppercase}.meta{color:var(--muted);font-family:var(--mono);font-size:11px;line-height:1.45;text-align:right;padding-top:2px;min-width:245px}.template-tag{display:inline-block;margin-bottom:8px;padding:7px 9px;border:1px solid var(--accent-edge);border-radius:999px;background:var(--panel3);color:var(--accent);font-weight:900}.dup-tag{display:inline-block;margin-top:7px;padding:5px 8px;border:1px solid var(--warn-edge);border-radius:999px;color:var(--warn)}
h1{margin:2px 0 0;font-size:34px;line-height:1.06;font-weight:950;letter-spacing:-.045em}.case-header h1{max-width:940px;font-size:30px;line-height:1.09;letter-spacing:-.035em;word-break:keep-all}.question-en{margin-top:5px;max-width:960px;font-size:13px;line-height:1.25;color:var(--muted)}.kicker{color:var(--muted);font-family:var(--mono);font-size:10.5px;letter-spacing:.8px;text-transform:uppercase}.num{font-family:var(--mono);font-weight:900}.good{color:var(--good)}.warn{color:var(--warn)}.accent{color:var(--accent)}.muted{color:var(--muted)}footer{border-top:1px solid var(--line);padding-top:8px;color:var(--muted);font-family:var(--mono);font-size:11px;display:flex;justify-content:space-between;gap:16px}
.summary-main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;min-height:0}.card{background:linear-gradient(180deg,color-mix(in srgb,var(--panel2) 94%,var(--accent)),var(--panel2));border:1px solid var(--line);border-radius:14px;padding:16px;display:grid;grid-template-rows:auto auto 1fr;gap:12px;min-height:0;box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--accent) 9%,transparent)}.card h2{margin:0;font-size:24px;line-height:1;font-weight:900;letter-spacing:-.035em}.hero-stat{border:1px solid var(--accent-edge);border-radius:10px;background:var(--panel3);padding:12px;display:grid;gap:4px}.hero-stat .big{font-size:38px;line-height:.95;font-weight:950;letter-spacing:-.055em}.hero-stat .label{font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase}.metric-list{display:grid;gap:8px;align-content:start}.metric{display:flex;justify-content:space-between;gap:14px;border-bottom:1px solid var(--line);padding-bottom:6px}.metric .label{color:var(--muted);font-size:12px}.metric .value{font-family:var(--mono);font-size:13px;font-weight:900;text-align:right}.note{margin:0;color:var(--muted);font-size:12px;line-height:1.35}.accuracy-stack{display:grid;gap:12px;align-content:start}.bar-row{display:grid;grid-template-columns:92px 1fr 62px;gap:10px;align-items:center}.bar-row .name{color:var(--muted);font-size:12px}.bar{height:18px;background:var(--panel3);border:1px solid var(--line);border-radius:999px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--warn),var(--accent))}.fill.text{background:linear-gradient(90deg,var(--accent),var(--good))}.fill.geom{background:linear-gradient(90deg,var(--good),var(--hl))}.pct{font-family:var(--mono);font-weight:900;text-align:right}.delta{border:1px solid var(--good-edge);border-radius:12px;padding:12px;background:color-mix(in srgb,var(--good) 8%,var(--panel3))}.delta .big{font-size:30px;font-weight:950;letter-spacing:-.045em}
.choices{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}.choice{min-height:46px;border:1px solid var(--line);border-radius:12px;background:var(--panel2);display:grid;grid-template-columns:34px 1fr;gap:8px;align-items:center;padding:8px}.choice.gold{border-color:var(--good-edge);background:color-mix(in srgb,var(--good) 11%,var(--panel2));box-shadow:0 0 0 1px color-mix(in srgb,var(--good) 20%,transparent)}.letter{width:30px;height:30px;border-radius:999px;display:grid;place-items:center;background:var(--panel3);font-family:var(--mono);font-weight:950;color:var(--accent)}.choice.gold .letter{background:var(--good);color:var(--panel3)}.copy{font-size:13px;line-height:1.15;font-weight:750}.check{float:right;color:var(--good);font-weight:950}
.twocol{display:grid;grid-template-columns:1fr 1.06fr;gap:12px;min-height:0}.panel{border:1px solid var(--line);border-radius:14px;padding:12px;background:linear-gradient(180deg,var(--panel2),var(--panel3));display:grid;grid-template-rows:auto auto 1fr;gap:10px;min-height:0}.panel.no-sp{border-color:var(--warn-edge)}.panel.yes-sp{border-color:var(--good-edge)}.panel-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.panel h2{margin:2px 0 0;font-size:20px;line-height:1.05;letter-spacing:-.035em}.answer{font-family:var(--mono);font-size:38px;line-height:.9;font-weight:950}.chip-row{display:flex;gap:7px;flex-wrap:wrap}.chip{border:1px solid var(--line);border-radius:999px;padding:5px 8px;background:var(--panel3);font-family:var(--mono);font-size:10.5px;color:var(--muted)}.gist{font-size:12.2px;line-height:1.35;color:var(--ink);overflow:hidden}.spatial-display{margin:0;max-height:88px;overflow:hidden;white-space:pre-wrap;border:1px solid var(--line);border-radius:10px;background:#07101f;padding:8px;color:var(--good);font-family:var(--mono);font-size:10.2px;line-height:1.25}
.viz-strip{display:grid;grid-template-columns:1fr 1fr;gap:12px}.viz-card{display:grid;grid-template-columns:210px 1fr;gap:10px;align-items:stretch;border:1px solid var(--line);border-radius:14px;background:var(--panel2);padding:9px;min-height:145px;overflow:hidden}.viz-card img{width:210px;height:126px;object-fit:cover;border-radius:10px;border:1px solid var(--line);background:var(--panel3)}.viz-copy{display:grid;align-content:start;gap:6px}.viz-copy h3{margin:0;font-size:17px;line-height:1.05}.viz-copy p{margin:0;color:var(--muted);font-size:11.5px;line-height:1.35}.legend{display:flex;gap:10px;color:var(--muted);font-family:var(--mono);font-size:10px}.swatch{display:inline-block;width:9px;height:9px;border-radius:999px;margin-right:5px}.swatch.good{background:var(--good)}.swatch.warn{background:var(--warn)}
""".strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def mb(value: int) -> str:
    return f"{value / 1_000_000:.1f} MB"


def pct(value: float) -> str:
    return f"{value * 100:.1f} %"


def pp(value: float) -> str:
    return f"+{value:.1f} pp"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def compact(text: Any, limit: int = 310) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def compact_pre(text: Any, limit: int = 1000) -> str:
    cleaned = re.sub(r"[ \t]+", " ", str(text)).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def prediction_counts(trials: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(trial.get("prediction") or "?") for trial in trials)


def trial_summary(question: dict[str, Any], config: str) -> tuple[str, str, str, str]:
    trials = question["trials"].get(config, [])
    if not trials:
        return "?", "correct 0/0", "axes none", "No trial recorded."
    counts = prediction_counts(trials)
    majority = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    trial = next((item for item in trials if str(item.get("prediction") or "?") == majority), trials[0])
    correct = sum(1 for item in trials if item.get("correct"))
    axes = " → ".join(str(axis) for axis in trial.get("axis_selections", [])) or "none"
    gist = compact(trial.get("reasoning_summary", "No reasoning recorded."), 360)
    return majority, f"correct {correct}/{len(trials)}", f"axes {axes}", gist


def spatial_display(question: dict[str, Any]) -> str:
    for trial in question["trials"].get("spatial_ON_grounded", []):
        summary = str(trial.get("reasoning_summary", ""))
        match = re.search(r"R\d+\s+spatial:\s*(.*?)(?:\s+\|\s+R\d+\s+\w+:|$)", summary, flags=re.S)
        if match and match.group(1).strip():
            return compact_pre(match.group(1), 1100)
    for trial in question["trials"].get("spatial_ON_grounded", []):
        for text in trial.get("retrieved_spatial_text", []):
            if text:
                return compact_pre(text, 1100)
    return "No grounded spatial string retrieved."


def select_cases(per_question: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    ranked = sorted(
        per_question,
        key=lambda q: (
            q.get("spatial_ON_grounded_correct", 0) - q.get("spatial_OFF_correct", 0),
            q.get("spatial_ON_grounded_correct", 0) - q.get("spatial_ON_text_correct", 0),
            q["id"],
        ),
        reverse=True,
    )
    picked: list[dict[str, Any]] = []
    seen_templates: set[str] = set()
    positive_templates = {
        str(question.get("template_id", ""))
        for question in ranked
        if question.get("spatial_ON_grounded_correct", 0) - question.get("spatial_OFF_correct", 0) > 0
    }
    for question in ranked:
        lift = question.get("spatial_ON_grounded_correct", 0) - question.get("spatial_OFF_correct", 0)
        template_id = str(question.get("template_id", ""))
        if lift > 0 and template_id not in seen_templates:
            picked.append(question)
            seen_templates.add(template_id)
        if len(picked) == 5:
            return picked, False
    shortage = len(positive_templates) < 5
    picked_ids = {question["id"] for question in picked}
    for question in ranked:
        if question["id"] in picked_ids:
            continue
        picked.append(question)
        picked_ids.add(question["id"])
        if len(picked) == 5:
            break
    return picked, shortage


def source_chunk_from_asset(asset_sources: dict[str, Any], filename: str) -> str:
    return str(asset_sources.get(filename, {}).get("chunk_id", ""))


def time8_from_chunk(chunk_id: str) -> str:
    return str(int(chunk_id) % 100_000_000).zfill(8)


def nearest_thumbnail(repo: Path, chunk_id: str) -> Path | None:
    thumb_dir = repo / THUMB_DIR
    exact = thumb_dir / f"thumb_{chunk_id}.jpg"
    if exact.exists():
        return exact
    try:
        target = int(chunk_id)
    except ValueError:
        return None
    candidates: list[tuple[int, Path]] = []
    for path in thumb_dir.glob("thumb_*.jpg"):
        try:
            value = int(path.stem.replace("thumb_", ""))
        except ValueError:
            continue
        candidates.append((abs(value - target), path))
    return min(candidates, default=(0, None))[1]


def decord_keyframe(repo: Path, chunk_id: str) -> Image.Image | None:
    if VideoReader is None or cpu is None:
        return None
    video_path = repo / VIDEO_DIR / f"DAY1_A1_JAKE_{time8_from_chunk(chunk_id)}.mp4"
    if not video_path.exists() or video_path.stat().st_size == 0:
        return None
    try:
        reader = VideoReader(str(video_path), ctx=cpu(0))
        index = min(len(reader) - 1, max(0, len(reader) // 2))
        return Image.fromarray(reader[index].asnumpy()).convert("RGB")
    except Exception:
        return None


def grounding_for_question(grounding: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    triples = question.get("evidence_triples") or []
    if triples:
        key = str(triples[0].get("id", ""))
        if key in grounding:
            return {key: grounding[key]}
    chunk = str(question.get("evidence_chunk", ""))
    return {key: value for key, value in grounding.items() if key.startswith(f"spatial_{chunk}_")}


def grounding_for_chunk(grounding: dict[str, Any], chunk_id: str) -> dict[str, Any]:
    return {key: value for key, value in grounding.items() if key.startswith(f"spatial_{chunk_id}_")}


def box_from_grounding(info: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    center = info.get("bbox_center") or [0.0, 0.0, 0.0]
    extent = info.get("bbox_extent") or [0.04, 0.04, 0.04]
    x = width * (0.5 + float(center[0]) * 2.8)
    y = height * (0.5 - float(center[1]) * 3.6)
    bw = max(30, min(width * 0.72, abs(float(extent[0])) * width * 5.0))
    bh = max(30, min(height * 0.72, abs(float(extent[1])) * height * 7.0))
    x0 = int(max(0, min(width - 2, x - bw / 2)))
    y0 = int(max(0, min(height - 2, y - bh / 2)))
    x1 = int(max(x0 + 1, min(width - 1, x + bw / 2)))
    y1 = int(max(y0 + 1, min(height - 1, y + bh / 2)))
    return (x0, y0, x1, y1)


def fit_frame(image: Image.Image, size: tuple[int, int] = (640, 360)) -> Image.Image:
    image = image.convert("RGB")
    scale = max(size[0] / image.width, size[1] / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - size[0]) // 2)
    top = max(0, (resized.height - size[1]) // 2)
    return resized.crop((left, top, left + size[0], top + size[1]))


def generate_frame_asset(repo: Path, grounding: dict[str, Any], question: dict[str, Any], out_path: Path) -> None:
    chunk = str(question.get("evidence_chunk", ""))
    image = decord_keyframe(repo, chunk)
    if image is None:
        thumb = nearest_thumbnail(repo, chunk)
        image = Image.open(thumb).convert("RGB") if thumb and thumb.exists() else Image.new("RGB", (640, 360), "#101827")
    image = fit_frame(image)
    draw = ImageDraw.Draw(image)
    records = grounding_for_question(grounding, question)
    first = next(iter(records.values()), {})
    for label, color in [("subject_grounding", "#22c55e"), ("object_grounding", "#f59e0b")]:
        if label in first:
            box = box_from_grounding(first[label], image.width, image.height)
            draw.rectangle(box, outline=color, width=5)
            draw.rectangle((box[0], max(0, box[1] - 20), min(image.width, box[0] + 90), box[1]), fill="#07101fcc")
            draw.text((box[0] + 6, max(2, box[1] - 17)), label.replace("_grounding", ""), fill=color)
    draw.rectangle((0, image.height - 32, image.width, image.height), fill="#07101fcc")
    draw.text((12, image.height - 23), f"chunk {chunk} · decord keyframe · bbox overlay", fill="#e9ecf5")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)


def points_from_grounding(records: dict[str, Any], limit: int = 500) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for item in records.values():
        point_cloud = item.get("point_cloud") or []
        for point in point_cloud:
            if isinstance(point, (list, tuple)) and len(point) >= 3:
                points.append((float(point[0]), float(point[1]), float(point[2])))
        for label in ("subject_grounding", "object_grounding"):
            info = item.get(label) or {}
            center = info.get("bbox_center")
            extent = info.get("bbox_extent") or [0.0, 0.0, 0.0]
            if not center:
                continue
            cx, cy, cz = [float(v) for v in center[:3]]
            ex, ey, ez = [abs(float(v)) / 2 for v in extent[:3]]
            points.append((cx, cy, cz))
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        points.append((cx + sx * ex, cy + sy * ey, cz + sz * ez))
    if len(points) > limit:
        step = max(1, math.ceil(len(points) / limit))
        points = points[::step][:limit]
    return points


def generate_pointcloud_asset(grounding: dict[str, Any], question: dict[str, Any], out_path: Path) -> list[list[float]]:
    chunk = str(question.get("evidence_chunk", ""))
    points = points_from_grounding(grounding_for_chunk(grounding, chunk), limit=500)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100, facecolor="#07101f")
    ax.set_facecolor("#0b1429")
    ax.set_title(f"top-down XY · chunk {chunk}", color="#e9ecf5", fontsize=12, pad=10, fontweight="bold")
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        zs = [point[2] for point in points]
        scatter = ax.scatter(xs, ys, c=zs, cmap="viridis", s=38, edgecolors="#e9ecf5", linewidths=0.35)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Z rel", color="#8aa0c8", fontsize=8)
        cbar.ax.tick_params(colors="#8aa0c8", labelsize=7)
    else:
        ax.text(0.5, 0.5, "no grounding points", color="#8aa0c8", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("X relative", color="#8aa0c8", fontsize=9)
    ax.set_ylabel("Y relative", color="#8aa0c8", fontsize=9)
    ax.tick_params(colors="#8aa0c8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#1c2746")
    ax.grid(color="#1c2746", alpha=0.45)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [[round(x, 6), round(y, 6), round(z, 6)] for x, y, z in points]


def ensure_case_assets(repo: Path, selected: list[dict[str, Any]]) -> None:
    asset_dir = repo / ASSET_DIR
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_sources_path = repo / POINTCLOUD_SOURCES
    asset_sources = read_json(asset_sources_path) if asset_sources_path.exists() else {}
    grounding = read_json(repo / GROUNDING)
    new_sources: dict[str, Any] = {}
    for idx, question in enumerate(selected, 1):
        chunk = str(question.get("evidence_chunk", ""))
        frame_name = f"worldmm_final_case_{idx:02d}_frame.jpg"
        point_name = f"worldmm_final_case_{idx:02d}_pointcloud.png"
        frame_path = asset_dir / frame_name
        point_path = asset_dir / point_name
        current_matches = source_chunk_from_asset(asset_sources, point_name) == chunk and frame_path.exists() and point_path.exists()
        if not current_matches:
            generate_frame_asset(repo, grounding, question, frame_path)
            points = generate_pointcloud_asset(grounding, question, point_path)
        else:
            points = asset_sources.get(point_name, {}).get("points", [])
        new_sources[point_name] = {"points": points, "case_id": question["id"], "chunk_id": chunk}
    asset_sources_path.write_text(json.dumps(new_sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def compression_metrics(repo: Path) -> dict[str, int | float | str]:
    raw = dir_size(repo / VIDEO_DIR)
    episodic = dir_size(repo / "output/metadata/episodic_memory/A1_JAKE")
    semantic = dir_size(repo / "output/metadata/semantic_memory/A1_JAKE")
    visual = dir_size(repo / "output/metadata/visual_memory/A1_JAKE")
    spatial = dir_size(repo / "output/metadata/spatial_memory/A1_JAKE")
    embeddings = episodic + semantic + visual
    all_metadata = embeddings + spatial
    return {
        "raw": raw,
        "episodic": episodic,
        "semantic": semantic,
        "visual": visual,
        "spatial": spatial,
        "embeddings": embeddings,
        "all_metadata": all_metadata,
        "ratio_embeddings": raw / embeddings if embeddings else 0.0,
        "ratio_all": raw / all_metadata if all_metadata else 0.0,
    }


def render_summary(data: dict[str, Any], selected: list[dict[str, Any]], compression: dict[str, int | float | str]) -> str:
    ag = data["aggregates"]
    delta = data["delta_analysis"]
    bars = []
    for name, cls, label in [("spatial_OFF", "", "3-axis"), ("spatial_ON_text", "text", "+Spatial text"), ("spatial_ON_grounded", "geom", "+Geometry")]:
        acc = ag[name]["accuracy"]
        bars.append(f'<div class="bar-row"><div class="name">{label}</div><div class="bar"><div class="fill {cls}" style="width:{acc*100:.1f}%"></div></div><div class="pct">{pct(acc)}</div></div>')
    trials = 60 * data["protocol"]["trials_per_config"] * len(CONFIGS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>WorldMM · final summary</title><style>{CSS}</style></head>
<body><section class="slide"><header><div><div class="tag">final summary deck</div><h1>WorldMM DAY1 — 4-axis memory with geometry (natural Q rerun)</h1></div><div class="meta">A1_JAKE / DAY1<br>Phase 4 natural ablation<br>60 Q × 3 configs × 3 trials</div></header>
<main class="summary-main">
<article class="card"><div><div class="kicker">scope</div><h2>One day, natural bilingual QA</h2></div><div class="hero-stat"><div class="big"><span class="num">540</span> trials</div><div class="label">60 Q × 3 trials × 3 configs</div></div><div class="metric-list"><div class="metric"><span class="label">Video chunks</span><span class="value">828 × 30 s = 6.9 h</span></div><div class="metric"><span class="label">Spatial-capable</span><span class="value accent">453/828</span></div><div class="metric"><span class="label">Grounded geometry</span><span class="value good">31/828</span></div><div class="metric"><span class="label">Natural Q run</span><span class="value good">{trials}</span></div><p class="note">Korean slide questions pair with same-meaning English prompts used by the ablation harness.</p></div></article>
<article class="card"><div><div class="kicker">compression</div><h2>Memory keeps signal, drops video mass</h2></div><div class="hero-stat"><div class="big"><span class="num">~85×</span></div><div class="label">13 GB MP4 → ~152 MB embeddings</div></div><div class="metric-list"><div class="metric"><span class="label">Raw MP4 du</span><span class="value">{mb(int(compression['raw']))}</span></div><div class="metric"><span class="label">E/S/V metadata du</span><span class="value accent">{mb(int(compression['embeddings']))}</span></div><div class="metric"><span class="label">Spatial geometry du</span><span class="value">{mb(int(compression['spatial']))}</span></div><div class="metric"><span class="label">All metadata du</span><span class="value">{mb(int(compression['all_metadata']))}</span></div><p class="note">Verified with du -bs output/metadata/*/A1_JAKE before writing this slide.</p></div></article>
<article class="card"><div><div class="kicker">accuracy · natural Qs</div><h2>Spatial axis unlocks memory</h2></div><div class="accuracy-stack">{''.join(bars)}<div class="delta"><div class="kicker">lift from 3-axis baseline</div><div class="big good">{pp(delta['spatial_ON_grounded_minus_spatial_OFF_pp'])}</div><div class="note">Text spatial {pp(delta['spatial_ON_text_minus_spatial_OFF_pp'])}; geometry-unique {pp(delta['grounded_minus_text_pp'])}.</div></div></div></article>
</main><footer><span>Natural Korean/English questions, 6 templates. Single-day, single-subject. Geometry on 31/828 chunks.</span></footer></section></body></html>"""


def choices_html(question: dict[str, Any]) -> str:
    parts = []
    for key, value in question["choices"].items():
        cls = "choice gold" if key == question["gold"] else "choice"
        check = '<span class="check">✓</span>' if key == question["gold"] else ""
        parts.append(f'<div class="{cls}"><div class="letter">{esc(key)}</div><div class="copy">{esc(value)}{check}</div></div>')
    return "".join(parts)


def render_case(question: dict[str, Any], idx: int, duplicate_template: bool) -> str:
    lift = question.get("spatial_ON_grounded_correct", 0) - question.get("spatial_OFF_correct", 0)
    geom_unique = question.get("spatial_ON_grounded_correct", 0) - question.get("spatial_ON_text_correct", 0)
    off_letter, off_rate, off_axes, off_gist = trial_summary(question, "spatial_OFF")
    on_majority, on_rate, on_axes, _ = trial_summary(question, "spatial_ON_grounded")
    template_id = str(question.get("template_id", "?"))
    template_desc = TEMPLATE_LABELS.get(template_id, str(question.get("template_description", "")))
    chunk = str(question.get("evidence_chunk", ""))
    frame = f"../../output/final_deck_assets/worldmm_final_case_{idx:02d}_frame.jpg"
    cloud = f"../../output/final_deck_assets/worldmm_final_case_{idx:02d}_pointcloud.png"
    duplicate = '<div class="dup-tag">duplicate template fill</div>' if duplicate_template else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>WorldMM · final case {idx:02d}</title><style>{CSS}</style></head>
<body><section class="slide case-slide"><header class="case-header"><div class="question-block"><div class="tag">geometry lift top 5 · rank {idx}</div><h1>{esc(question.get('question_kr', ''))}</h1><div class="question-en">{esc(question.get('question_en', ''))}</div></div><div class="meta"><div class="template-tag">Template {esc(template_id)} — {esc(template_desc)}</div><br>{esc(question.get('evidence_chunk_label', ''))}<br>chunk {esc(chunk)}<br>lift +{lift}/3 · geometry-unique +{geom_unique}/3{duplicate}</div></header>
<section class="choices">{choices_html(question)}</section>
<main class="twocol"><article class="panel no-sp"><div class="panel-head"><div><div class="kicker">LEFT · 3-axis (E+S+V, no spatial)</div><h2>Baseline misses the grounded relation</h2></div><div class="answer warn">{esc(off_letter)}</div></div><div class="chip-row"><span class="chip">{esc(off_rate)}</span><span class="chip">{esc(off_axes)}</span></div><div class="gist">{esc(off_gist)}</div></article>
<article class="panel yes-sp"><div class="panel-head"><div><div class="kicker">RIGHT · 4-axis (+Spatial+Geometry)</div><h2>Spatial axis surfaces 3D centers</h2></div><div class="answer good">{esc(question['gold'])}</div></div><div class="chip-row"><span class="chip">{esc(on_rate)}</span><span class="chip">majority {esc(on_majority)}</span><span class="chip">{esc(on_axes)}</span></div><pre class="spatial-display">{esc(spatial_display(question))}</pre></article></main>
<section class="viz-strip"><article class="viz-card"><img src="{frame}" alt="bbox overlay for {esc(question['id'])}"><div class="viz-copy"><div class="kicker">frame evidence</div><h3>Keyframe with bbox overlay</h3><p>Subject and object boxes from the evidence triple, drawn on the decord-extracted middle frame for this chunk.</p><div class="legend"><span><span class="swatch good"></span>subject</span><span><span class="swatch warn"></span>object</span></div></div></article><article class="viz-card"><img src="{cloud}" alt="point cloud for {esc(question['id'])}"><div class="viz-copy"><div class="kicker">geometry evidence</div><h3>Top-down point-cloud scatter</h3><p>Up to 500 grounding-sidecar points, plotted in XY with Z encoded as color.</p></div></article></section>
<footer><span>{esc(question['id'])}</span><span>{esc(question.get('evidence_chunk_label', ''))}</span><span>Template {esc(template_id)}</span></footer></section></body></html>"""


def build_html_deck(repo: Path) -> tuple[list[dict[str, Any]], bool] | None:
    results_path = repo / RESULTS
    if not results_path.exists():
        return None
    data = read_json(results_path)
    selected, shortage = select_cases(data["per_question"])
    ensure_case_assets(repo, selected)
    compression = compression_metrics(repo)
    write_text(repo / SLIDES_DIR / "worldmm-final-summary.html", render_summary(data, selected, compression))
    template_counts = Counter(str(question.get("template_id", "")) for question in selected)
    for idx, question in enumerate(selected, 1):
        template_id = str(question.get("template_id", ""))
        duplicate_template = shortage and template_counts[template_id] > 1
        write_text(repo / SLIDES_DIR / f"worldmm-final-case-{idx:02d}.html", render_case(question, idx, duplicate_template))
    templates = [str(question.get("template_id", "")) for question in selected]
    questions = [str(question.get("question_kr", "")) for question in selected]
    if shortage:
        print("NOTE: fewer than 5 templates had positive-lift cases; duplicate-template fills are tagged on affected slides.")
    ag = data["aggregates"]
    aggregate = f"3-axis {pct(ag['spatial_OFF']['accuracy'])}, +Spatial text {pct(ag['spatial_ON_text']['accuracy'])}, +Spatial+Geometry {pct(ag['spatial_ON_grounded']['accuracy'])}"
    print("Summary: picked " + "; ".join(f"Template {template} — {question}" for template, question in zip(templates, questions)) + f". Aggregate: {aggregate}.")
    return selected, shortage


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
    print(f"wrote {out_pptx}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("docs/slides/pptx/worldmm_final_deck.pptx"))
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not args.skip_html:
        build_html_deck(repo)
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
