"""Draw a dependency-free SVG of the measured GPU A/B summary."""
from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path


COLORS = {"hf": "#3b75af", "vllm": "#d17a32"}
LABELS = {"hf": "Transformers StaticCache", "vllm": "vLLM"}


def draw_panel(parts: list[str], rows: list[dict], *, left: int, title: str,
               field: str, deviation: str, upper: float, tick: float) -> None:
    top, width, height = 84, 480, 245
    parts.append(f'<text class="panel" x="{left}" y="58">{escape(title)}</text>')
    parts.append(f'<path d="M{left},{top}v{height}h{width}" fill="none" stroke="#63748a"/>')
    value = 0.0
    while value <= upper:
        y = top + height * (1 - value / upper)
        parts.append(f'<path d="M{left},{y:.1f}h{width}" stroke="#e7ebf0"/>')
        parts.append(f'<text class="tick" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{value:g}</text>')
        value += tick
    group_width = width / len(rows)
    bar_width = 30
    for index, row in enumerate(rows):
        center = left + group_width * (index + 0.5)
        parts.append(f'<text class="tick" x="{center:.1f}" y="351" text-anchor="middle">{row["batch_size"]}</text>')
        for backend, side in (("hf", -1), ("vllm", 1)):
            measurement = float(row[f"{backend}_{field}_mean"])
            uncertainty = float(row[f"{backend}_{deviation}_stdev"] or 0)
            x = center + side * 17 - bar_width / 2
            y = top + height * (1 - min(measurement, upper) / upper)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{top+height-y:.1f}" fill="{COLORS[backend]}"/>')
            if uncertainty:
                error_top = top + height * (1 - min(measurement + uncertainty, upper) / upper)
                error_bottom = top + height * (1 - max(measurement - uncertainty, 0) / upper)
                mid = x + bar_width / 2
                parts.append(f'<path d="M{mid:.1f},{error_top:.1f}V{error_bottom:.1f}m-5,0h10m-10,{error_top-error_bottom:.1f}h10" fill="none" stroke="#1d2733" stroke-width="1.5"/>')
    parts.append(f'<text class="tick" x="{left+width/2}" y="377" text-anchor="middle">提交批量（请求数）</text>')


def plot(summary: str, output: str) -> None:
    with open(summary, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(int(row["hf_successful_runs"]) < 3 or
                       int(row["vllm_successful_runs"]) < 3 for row in rows):
        raise ValueError("plot requires at least three successful runs per engine and batch")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="445" viewBox="0 0 1120 445">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#243044}'
        '.panel{font-size:18px;font-weight:bold}.tick{font-size:12px}.note{font-size:12px}</style>',
    ]
    draw_panel(parts, rows, left=75, title="输出吞吐（token/s，均值 ± 标准差）",
               field="tokens_per_s", deviation="tokens_per_s", upper=450, tick=100)
    draw_panel(parts, rows, left=635, title="设备已用峰值（MiB，含预热）",
               field="peak_memory_mib", deviation="peak_memory_mib", upper=15360, tick=3000)
    for index, backend in enumerate(("hf", "vllm")):
        x = 80 + index * 250
        parts.append(f'<rect x="{x}" y="403" width="16" height="12" fill="{COLORS[backend]}"/>')
        parts.append(f'<text class="note" x="{x+23}" y="414">{LABELS[backend]}</text>')
    parts.append('<text class="note" x="605" y="414">同卡三轮；引擎 A/B，不能归因于分页单因素</text>')
    parts.append('</svg>')
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == "__main__":
    main()
