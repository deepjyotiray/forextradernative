"""
Analytics dashboard routes for Phase 3 visualizations.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .ai_analysis import ai_analysis_service
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
  <link rel="icon" href="/site-icon.ico" sizes="any">
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
    .stack {{
      display: grid;
      gap: 16px;
      margin-bottom: 16px;
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
    .mini-actions {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .ai-status {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .ai-summary {{
      white-space: pre-wrap;
      margin: 0;
      padding: 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(15,107,168,.06), rgba(255,255,255,.7));
      font: 500 14px/1.55 "Segoe UI", system-ui, sans-serif;
      min-height: 132px;
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
        <a class="btn" href="/analytics/api/ai-summary?days={days}">AI Summary JSON</a>
      </div>
    </section>
    <section class="meta">
      <div class="stat"><div class="k">Lookback</div><div class="v">{days}d</div></div>
      <div class="stat"><div class="k">Completed Trades</div><div class="v">{result.get('completed_trade_count', 0)}</div></div>
      <div class="stat"><div class="k">Decisions</div><div class="v">{result.get('decision_count', 0)}</div></div>
      <div class="stat"><div class="k">Generated</div><div class="v" style="font-size:16px">{generated_at or 'n/a'}</div></div>
    </section>
    <section class="stack">
      <section class="card">
        <div class="card-head">
          <div>
            <h2>AI Quick Analysis</h2>
            <p>Short commentary generated from the existing analytics and order-review data.</p>
          </div>
          <div class="mini-actions">
            <button class="btn" type="button" onclick="loadAiSummary(1)">Refresh AI Analysis</button>
            <a class="link" href="/analytics/api/ai-summary?days={days}" target="_blank" rel="noopener">Open JSON</a>
          </div>
        </div>
        <div class="ai-status" id="aiSummaryStatus">Loading AI analysis...</div>
        <pre class="ai-summary" id="aiSummaryText">Loading...</pre>
      </section>
    </section>
    <section class="grid">
      {''.join(chart_cards)}
    </section>
  </main>
  <script>
    async function loadAiSummary(refresh) {{
      const statusEl = document.getElementById('aiSummaryStatus');
      const textEl = document.getElementById('aiSummaryText');
      statusEl.textContent = refresh ? 'Refreshing AI analysis...' : 'Loading AI analysis...';
      textEl.textContent = 'Loading...';
      try {{
        const response = await fetch(`/analytics/api/ai-summary?days={days}&refresh=${{refresh ? 1 : 0}}`);
        const payload = await response.json();
        if (payload.status === 'ready') {{
          const cached = payload.cached ? 'Cached' : 'Fresh';
          statusEl.textContent = `${{cached}} summary using ${{payload.model}} at ${{payload.generated_at}}`;
          textEl.textContent = payload.analysis || 'No summary returned.';
          return;
        }}
        if (payload.status === 'disabled') {{
          statusEl.textContent = 'AI analysis is disabled';
          textEl.textContent = payload.message || 'Set OPENAI_API_KEY to enable AI analysis.';
          return;
        }}
        statusEl.textContent = 'AI analysis unavailable';
        textEl.textContent = payload.error || payload.message || 'AI analysis failed.';
      }} catch (error) {{
        statusEl.textContent = 'AI analysis unavailable';
        textEl.textContent = String(error);
      }}
    }}
    loadAiSummary(0);
  </script>
</body>
</html>"""


def _build_day_analysis_page(ist_date: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Day Trade Analysis - {ist_date}</title>
  <link rel="icon" href="/site-icon.ico" sizes="any">
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #14213d;
      --muted: #5c677d;
      --line: #d6dde6;
      --accent: #0f6ba8;
      --accent-2: #e09f3e;
      --green: #1f8f57;
      --red: #c0392b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(15,107,168,.12), transparent 24%),
        radial-gradient(circle at bottom left, rgba(224,159,62,.14), transparent 28%),
        linear-gradient(180deg, #f9fbfd 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 980px;
      margin: 0 auto;
      padding: 28px 18px 36px;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
      line-height: 1;
      letter-spacing: -.04em;
    }}
    .sub {{
      color: var(--muted);
      font-size: 14px;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .btn {{
      text-decoration: none;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      padding: 10px 14px;
      border-radius: 10px;
      font-weight: 600;
      cursor: pointer;
    }}
    .btn.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .stat, .card {{
      background: rgba(255,255,255,.85);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(20,33,61,.06);
    }}
    .stat {{
      padding: 14px 16px;
    }}
    .k {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 6px;
    }}
    .v {{
      font-size: 24px;
      font-weight: 700;
    }}
    .card {{
      padding: 16px;
      margin-bottom: 14px;
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .analysis {{
      white-space: pre-wrap;
      margin: 0;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: linear-gradient(180deg, rgba(15,107,168,.06), rgba(255,255,255,.7));
      font: 500 14px/1.55 "Segoe UI", system-ui, sans-serif;
      min-height: 148px;
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 12px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    li + li {{
      margin-top: 8px;
    }}
    .good {{ color: var(--green); }}
    .bad {{ color: var(--red); }}
    code {{
      background: rgba(20,33,61,.06);
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div>
        <h1>Day Trade Analysis</h1>
        <div class="sub">On-demand AI review for IST trading day <code>{ist_date}</code>.</div>
      </div>
      <div class="actions">
        <button class="btn primary" type="button" onclick="loadDaySummary(1)">Refresh Analysis</button>
        <a class="btn" href="/analytics/api/day-summary?date={ist_date}" target="_blank" rel="noopener">Open JSON</a>
        <a class="btn" href="/analytics">Back to Visual Dashboard</a>
      </div>
    </section>
    <section class="meta" id="metaGrid">
      <div class="stat"><div class="k">Selected Day</div><div class="v">{ist_date}</div></div>
      <div class="stat"><div class="k">Trades</div><div class="v" id="metaTrades">-</div></div>
      <div class="stat"><div class="k">Win Rate</div><div class="v" id="metaWinRate">-</div></div>
      <div class="stat"><div class="k">PnL</div><div class="v" id="metaPnl">-</div></div>
    </section>
    <section class="card">
      <h2>AI Summary</h2>
      <div class="status" id="summaryStatus">Loading day analysis...</div>
      <pre class="analysis" id="summaryText">Loading...</pre>
    </section>
    <section class="grid">
      <section class="card">
        <h2>Decision Flow</h2>
        <div id="decisionFlow">Loading...</div>
      </section>
      <section class="card">
        <h2>Patterns</h2>
        <div id="patterns">Loading...</div>
      </section>
      <section class="card">
        <h2>Recommendations</h2>
        <div id="recommendations">Loading...</div>
      </section>
      <section class="card">
        <h2>Order Review</h2>
        <div id="orderReview">Loading...</div>
      </section>
    </section>
  </main>
  <script>
    const IST_DATE = {json.dumps(ist_date)};
    const INITIAL_REFRESH = 0;
    function fmtPct(v) {{
      return typeof v === 'number' ? `${{(v * 100).toFixed(1)}}%` : '-';
    }}
    function fmtPnl(v) {{
      if (typeof v !== 'number') return '-';
      return `${{v >= 0 ? '+' : ''}}${{v.toFixed(2)}}`;
    }}
    function esc(v) {{
      return String(v ?? '').replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
    }}
    function list(items) {{
      return items && items.length ? `<ul>${{items.map(item => `<li>${{esc(item)}}</li>`).join('')}}</ul>` : '<div>No data.</div>';
    }}
    async function loadDaySummary(refresh) {{
      const statusEl = document.getElementById('summaryStatus');
      const textEl = document.getElementById('summaryText');
      statusEl.textContent = refresh ? 'Refreshing day analysis...' : 'Loading day analysis...';
      textEl.textContent = 'Loading...';
      try {{
        const response = await fetch(`/analytics/api/day-summary?date=${{encodeURIComponent(IST_DATE)}}&refresh=${{refresh ? 1 : 0}}`);
        const payload = await response.json();
        const snapshot = payload.snapshot || {{}};
        const perf = snapshot.performance_summary || {{}};
        const decision = snapshot.decision_summary || {{}};
        const directional = snapshot.directional_analysis || {{}};
        const session = snapshot.session_analysis || {{}};
        const orderReview = snapshot.order_review || {{}};

        document.getElementById('metaTrades').textContent = perf.total_trades ?? '-';
        document.getElementById('metaWinRate').textContent = fmtPct(perf.win_rate);
        document.getElementById('metaPnl').textContent = fmtPnl(perf.total_pnl);
        document.getElementById('metaPnl').className = `v ${{(perf.total_pnl || 0) >= 0 ? 'good' : 'bad'}}`;

        document.getElementById('decisionFlow').innerHTML = `
          <div><strong>Total decisions:</strong> ${{decision.total_decisions ?? 0}}</div>
          <div><strong>Taken:</strong> ${{decision.trades_taken ?? 0}}</div>
          <div><strong>Skipped:</strong> ${{decision.trades_skipped ?? 0}}</div>
          <div><strong>Completed:</strong> ${{decision.trades_completed ?? 0}}</div>
          <div><strong>Conversion:</strong> ${{fmtPct(decision.conversion_rate)}}</div>
        `;
        document.getElementById('patterns').innerHTML = `
          <div><strong>Better direction:</strong> ${{directional.better_direction || 'n/a'}}</div>
          <div><strong>Best session:</strong> ${{session.best_session?.name || 'n/a'}}</div>
          <div><strong>Worst session:</strong> ${{session.worst_session?.name || 'n/a'}}</div>
          <div><strong>Report note:</strong> ${{esc(snapshot.report_error || 'none')}}</div>
        `;
        document.getElementById('recommendations').innerHTML = list(snapshot.recommendations || []);
        const topStrategies = (orderReview.strategies || []).map(row => `${{row.strategy}}: ${{row.trades}} trades, pnl ${{fmtPnl(row.total_pnl)}}`);
        const topReasons = (orderReview.close_reason_categories || []).map(row => `${{row.close_reason_category}}: ${{row.trades}} trades`);
        document.getElementById('orderReview').innerHTML = `
          <div><strong>Wins / Losses / BE:</strong> ${{orderReview.summary?.wins ?? 0}} / ${{orderReview.summary?.losses ?? 0}} / ${{orderReview.summary?.breakeven ?? 0}}</div>
          <div style="margin-top:10px"><strong>Top strategies</strong></div>
          ${{list(topStrategies)}}
          <div style="margin-top:10px"><strong>Top close reasons</strong></div>
          ${{list(topReasons)}}
        `;

        if (payload.status === 'ready') {{
          const freshness = payload.cached ? 'Cached' : 'Fresh';
          statusEl.textContent = `${{freshness}} summary using ${{payload.model}} at ${{payload.generated_at}}`;
          textEl.textContent = payload.analysis || 'No summary returned.';
          return;
        }}
        if (payload.status === 'disabled') {{
          statusEl.textContent = 'AI analysis is disabled';
          textEl.textContent = payload.message || 'Set OPENAI_API_KEY to enable AI analysis.';
          return;
        }}
        statusEl.textContent = 'AI analysis unavailable';
        textEl.textContent = payload.error || payload.message || 'Day analysis failed.';
      }} catch (error) {{
        statusEl.textContent = 'AI analysis unavailable';
        textEl.textContent = String(error);
      }}
    }}
    loadDaySummary(INITIAL_REFRESH);
  </script>
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


@router.get("/analytics/api/ai-summary")
async def analytics_ai_summary(
    days: int = Query(30, ge=1, le=365),
    refresh: int = Query(0, ge=0, le=1),
):
    return JSONResponse(ai_analysis_service.generate_summary(days=days, refresh=bool(refresh)))


@router.get("/analytics/day", response_class=HTMLResponse)
async def analytics_day_page(
    date: str = Query(..., min_length=10, max_length=10),
    refresh: int = Query(0, ge=0, le=1),
):
    return HTMLResponse(_build_day_analysis_page(date).replace("const INITIAL_REFRESH = 0;", f"const INITIAL_REFRESH = {1 if refresh else 0};"))


@router.get("/analytics/api/day-summary")
async def analytics_day_summary(
    date: str = Query(..., min_length=10, max_length=10),
    refresh: int = Query(0, ge=0, le=1),
):
    return JSONResponse(ai_analysis_service.generate_day_summary(ist_date=date, refresh=bool(refresh)))
