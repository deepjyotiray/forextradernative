# Remote Debugging Bundle

This repo already writes detailed runtime evidence on the trading machine, but most of it lives in local files that are excluded from git. To make remote analysis practical, use the portable debug bundle export.

## What it includes

- Current runtime logs such as `trader.log`, `deploy_watcher.log`, and startup logs
- Recent decision logs from `logs/` and `backup_logs/`
- `orders.db` and `analytics.db` snapshots
- Runtime config and profile files
- Deployment/version metadata
- Restart-scoped artifacts from `recent_logs_since_restart/`

Large rolling logs such as `trade_attribution.jsonl` are exported as recent tails so the bundle stays portable.

## Local export on the trading machine

```powershell
python .\export_debug_bundle.py --recent-days 7
```

This writes a zip file under `debug_exports/`.

## Remote export over the API

By default the debug export endpoints are local-only. To allow a remote machine to pull a bundle, set an environment variable on the trading machine before starting the service:

```powershell
$env:DEBUG_BUNDLE_TOKEN = "set-a-long-random-token-here"
```

Then restart the trading service. The remote machine can inspect availability first:

```powershell
Invoke-RestMethod "https://<your-host>/debug/manifest?recent_days=7&token=<token>"
```

And download a fresh bundle:

```powershell
Invoke-WebRequest "https://<your-host>/debug/export?recent_days=7&token=<token>" -OutFile .\debug_bundle.zip
```

## Notes

- If `DEBUG_BUNDLE_TOKEN` is not configured, `/debug/manifest` and `/debug/export` only accept localhost requests.
- The bundle includes `manifest.json` describing which files were copied fully and which were tailed.
- This is intended for debugging and enhancement analysis, not as a git-tracked artifact.
