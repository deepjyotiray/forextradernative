"""
Analytics dashboard routes for Phase 3 visualizations.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .visual_analytics import generate_visual_suite


router = APIRouter(tags=["analytics"])

_BASE_DIR = Path(__file__).resolve().parent.parent
_OUTPUT_ROOT = _BASE_DIR / "analytics_outputs"
_LATEST_DIR = _OUTPUT_ROOT / "latest"


def _build_analytics_page(result: dict, days: int) -> str:
    generated_at = result.get("generated_at", "")
    charts = result.get("charts", {})
    timestamp_token = generated_at.replace(":", "").replace("-", "").replace(".", "")

    chart_cards = []
    for chart in charts.values():
        relative_path = Path(chart["path"]).resolve().relative_to(_OUTPUT_ROOT.resolve()).as_posix()
        asset_url = f"/analytics-assets/{relative_path}?v={timestamp_token}"
        chart_cards.append(
            f"""
            <section class="card">
              <div class="card-head">
                <div>
                  <h2>{chart["title"]}</h2>
                  <p>{'Ready' if chart.get('has_data') else 'No source data yet'}</p>
                </div>
                <a class="link" href="{asset_url}" target="_blank" rel="noopener">Open image</a>
              </div>
              <img src="{asset_url}" alt="{chart['title']}" loading="lazy">
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Visual Analytics Dashboard</title>
  <style>
    :root {{
      --bg: #edf2f4;
      --panel: #ffffff;
      --ink: #14213d;
      --muted: #5c677d;
      --line: #d6dde6;
      --accent: #0f6ba8;
      --accent-2: #e09f3e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(224,159,62,.18), transparent 28%),
        linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
      margin-bottom: 22px;
      flex-wrap: wrap;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      line-height: 1;
      letter-spacing: -1px;
    }}
    .sub {{
      color: var(--muted);
      font-size: 14px;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .btn, .link {{
      text-decoration: none;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      padding: 10px 14px;
      border-radius: 10px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .btn.primary {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .stat {{
      background: rgba(255,255,255,.72);
      backdrop-filter: blur(12px);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .stat .k {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
      font-size: 11px;
      margin-bottom: 6px;
    }}
    .stat .v {{
      font-size: 24px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 10px 30px rgba(20,33,61,.06);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 12px;
    }}
    .card h2 {{
      margin: 0 0 4px;
      font-size: 18px;
    }}
    .card p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fafbfd;
    }}
    @media (max-width: 700px) {{
      .wrap {{ padding: 18px 12px 28px; }}
      h1 {{ font-size: 26px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div>
        <h1>Visual Analytics Dashboard</h1>
        <div class="sub">Phase 3 charts generated from the local analytics pipeline.</div>
      </div>
      <div class="actions">
        <a class="btn primary" href="/analytics?days={days}&refresh=1">Refresh Charts</a>
        <a class="btn" href="/analytics/api/generate?days={days}">JSON Summary</a>
      </div>
    </section>
    <section class="meta">
      <div class="stat"><div class="k">Lookback</div><div class="v">{days}d</div></div>
      <div class="stat"><div class="k">Completed Trades</div><div class="v">{result.get('completed_trade_count', 0)}</div></div>
      <div class="stat"><div class="k">Decisions</div><div class="v">{result.get('decision_count', 0)}</div></div>
      <div class="stat"><div class="k">Generated</div><div class="v" style="font-size:16px">{generated_at or 'n/a'}</div></div>
    </section>
    <section class="grid">
      {''.join(chart_cards)}
    </section>
  </main>
</body>
</html>"""


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_dashboard(
    days: int = Query(30, ge=1, le=365),
    refresh: int = Query(0, ge=0, le=1),
):
    manifest_path = _LATEST_DIR / "manifest.json"

    if refresh or not _LATEST_DIR.exists():
        result = generate_visual_suite(days=days, output_dir=str(_LATEST_DIR))
    else:
        if manifest_path.exists():
            result = json.loads(manifest_path.read_text(encoding="utf-8"))
            if result.get("period_days") != days:
                result = generate_visual_suite(days=days, output_dir=str(_LATEST_DIR))
        else:
            result = generate_visual_suite(days=days, output_dir=str(_LATEST_DIR))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return HTMLResponse(_build_analytics_page(result, days))


@router.get("/analytics/api/generate")
async def analytics_generate(days: int = Query(30, ge=1, le=365)):
    result = generate_visual_suite(days=days, output_dir=str(_LATEST_DIR))
    manifest_path = _LATEST_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return JSONResponse(result)
