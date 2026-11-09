"""Dependency-free SVG charts for the CPU simulation results."""
from html import escape
from pathlib import Path

COLORS = {"naive": "#b83b36", "paged": "#2474b6", "paged_dynamic": "#23845b"}
LABELS = {"naive": "连续预留", "paged": "一次性分页", "paged_dynamic": "逐步扩页"}


def write_comparison(path, rows, naive_total, paged_total):
    capacities = sorted({row[0] for row in rows})
    width, height = 1120, 430
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;font-size:13px;fill:#243044}.title{font-size:18px;font-weight:bold}</style>']
    panels = [("OOM 失败率 (%)", 2, 100), ("时间加权 GPU 块占用率 (%)", 3, 100)]
    for panel_idx, (title, metric, upper) in enumerate(panels):
        x0 = 55 + panel_idx * 370
        y0, plot_w, plot_h = 65, 305, 260
        parts.append(f'<text class="title" x="{x0}" y="35">{escape(title)}</text>')
        parts.append(f'<path d="M{x0},{y0}v{plot_h}h{plot_w}" fill="none" stroke="#65758a"/>')
        for value in (0, 25, 50, 75, 100):
            y = y0 + plot_h * (1 - value / upper)
            parts.append(f'<path d="M{x0},{y:.1f}h{plot_w}" stroke="#e5e9ee"/>')
            parts.append(f'<text x="{x0-35}" y="{y+4:.1f}">{value}</text>')
        for value in (capacities[0], capacities[len(capacities)//2], capacities[-1]):
            x = x0 + plot_w * (value - capacities[0]) / (capacities[-1] - capacities[0])
            parts.append(f'<text x="{x-18:.1f}" y="345">{value}</text>')
        for mode in COLORS:
            values = {row[0]: row[metric] * 100 for row in rows if row[1] == mode}
            points = " ".join(f"{x0 + plot_w*(cap-capacities[0])/(capacities[-1]-capacities[0]):.1f},{y0 + plot_h*(1-values[cap]/upper):.1f}" for cap in capacities)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[mode]}" stroke-width="2.5"/>')
            legend_y = 372 + 18 * list(COLORS).index(mode)
            parts.append(f'<path d="M{x0},{legend_y}h22" stroke="{COLORS[mode]}" stroke-width="3"/>')
            parts.append(f'<text x="{x0+28}" y="{legend_y+4}">{LABELS[mode]}</text>')
    x0 = 815
    parts.append(f'<text class="title" x="{x0}" y="35">累计预留块需求</text>')
    for i, (label, value, color) in enumerate((("连续预留", naive_total, COLORS["naive"]),
                                               ("按实际长度", paged_total, COLORS["paged"]))):
        y = 100 + i * 125
        bar_w = 220 * value / naive_total
        parts.append(f'<text x="{x0}" y="{y-10}">{label}: {value:,}</text>')
        parts.append(f'<rect x="{x0}" y="{y}" width="{bar_w:.1f}" height="45" fill="{color}"/>')
    parts.append(f'<text x="{x0}" y="365">减少 {(1-paged_total/naive_total)*100:.1f}%（非实测碎片率）</text>')
    parts.append('</svg>')
    Path(path).write_text("\n".join(parts), encoding="utf-8")
