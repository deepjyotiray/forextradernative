"""
Phase 4 CLI dashboard runner with matplotlib overview and export workflows.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Callable, Optional

import pandas as pd


class DashboardCLI:
    """Interactive terminal dashboard built on the Phase 3 visualization engine."""

    def __init__(
        self,
        days: int = 30,
        refresh_seconds: int = 15,
        output_root: Optional[str] = None,
        backend: str = "Agg",
        data_loader: Optional[Callable[[int], tuple[pd.DataFrame, pd.DataFrame]]] = None,
        sync_loader: Optional[Callable[[int], dict]] = None,
    ) -> None:
        self.days = days
        self.refresh_seconds = refresh_seconds
        self.output_root = Path(output_root) if output_root else Path("analytics_outputs") / "cli_dashboard"
        self.output_root.mkdir(parents=True, exist_ok=True)

        self._bootstrap_matplotlib(backend)
        self.data_loader = data_loader or self._default_data_loader
        self.sync_loader = sync_loader or self._default_sync_loader

    def _bootstrap_matplotlib(self, backend: str) -> None:
        import matplotlib

        matplotlib.use(backend)
        import matplotlib.pyplot as plt

        from .core_analytics import PerformanceMetricsCalculator
        from .visual_analytics import VisualAnalyticsEngine

        self.plt = plt
        self.visual_engine = VisualAnalyticsEngine()
        self.metrics_calculator = PerformanceMetricsCalculator
        self.backend = plt.get_backend()
        self.interactive = "agg" not in self.backend.lower()

    def _default_data_loader(self, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.visual_engine.load_trade_frames(days)

    def _default_sync_loader(self, hours: int) -> dict:
        from .data_pipeline import sync_data

        return sync_data(hours)

    def sync_pipeline(self) -> dict:
        """Sync attribution logs into the analytics database."""
        return self.sync_loader(max(self.days * 24, 72))

    def load_latest_data(self, sync_first: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Load the freshest dashboard data available."""
        sync_result = {"synced": False, "message": "sync skipped"}
        if sync_first:
            sync_result = self.sync_pipeline()
        decision_df, completed_df = self.data_loader(self.days)
        return decision_df, completed_df, sync_result

    def build_summary_text(
        self,
        decision_df: pd.DataFrame,
        completed_df: pd.DataFrame,
        sync_result: Optional[dict] = None,
    ) -> str:
        """Build a terminal summary block for the current dashboard state."""
        lines = [
            "",
            "Phase 4 CLI Dashboard",
            f"Lookback: {self.days} days | Backend: {self.backend} | Refresh: {self.refresh_seconds}s",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if sync_result:
            sync_state = sync_result.get("message") or (
                f"decisions={sync_result.get('decisions_synced', 0)}, outcomes={sync_result.get('outcomes_synced', 0)}"
            )
            lines.append(f"Sync: {sync_state}")

        if completed_df.empty:
            lines.extend(
                [
                    f"Decisions loaded: {len(decision_df)}",
                    "Completed trades: 0",
                    "Status: no completed trade data available yet.",
                ]
            )
            return "\n".join(lines)

        metrics = self.metrics_calculator.calculate_basic_metrics(completed_df)
        conversion_rate = float((decision_df["decision"] == "TRADE_TAKEN").mean()) if len(decision_df) > 0 and "decision" in decision_df.columns else 0.0

        lines.extend(
            [
                f"Decisions loaded: {len(decision_df)} | Completed trades: {len(completed_df)} | Conversion: {conversion_rate:.1%}",
                f"Win rate: {metrics.get('win_rate', 0):.1%} | Total PnL: {metrics.get('total_pnl', 0):.2f} | Profit factor: {self._fmt_ratio(metrics.get('profit_factor', 0))}",
                f"Avg win: {metrics.get('avg_win', 0):.2f} | Avg loss: {metrics.get('avg_loss', 0):.2f} | Max DD: {metrics.get('max_drawdown', 0):.2f}",
                f"Best trade: {metrics.get('best_trade', 0):.2f} | Worst trade: {metrics.get('worst_trade', 0):.2f}",
            ]
        )
        return "\n".join(lines)

    def export_overview_snapshot(
        self,
        decision_df: Optional[pd.DataFrame] = None,
        completed_df: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None,
        sync_first: bool = True,
    ) -> dict:
        """Export a single overview dashboard image."""
        if decision_df is None or completed_df is None:
            decision_df, completed_df, _ = self.load_latest_data(sync_first=sync_first)

        target_path = output_path or str(self.output_root / "dashboard_overview.png")
        return self.visual_engine.export_dashboard_snapshot(
            decision_df=decision_df,
            completed_df=completed_df,
            output_path=target_path,
            days=self.days,
        )

    def export_full_chart_suite(self, output_dir: Optional[str] = None, sync_first: bool = True) -> dict:
        """Export the full Phase 3 chart bundle."""
        decision_df, completed_df, _ = self.load_latest_data(sync_first=sync_first)
        target_dir = output_dir or str(self.output_root / f"bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        return self.visual_engine.generate_visual_suite_from_dataframes(
            decision_df=decision_df,
            completed_df=completed_df,
            output_dir=target_dir,
            metadata={"period_days": self.days},
        )

    def show_dashboard(self, auto_refresh: bool = False, sync_first: bool = True) -> Optional[str]:
        """Open the overview dashboard in a matplotlib window when supported."""
        snapshot_path = self.output_root / "dashboard_overview.png"
        self.export_overview_snapshot(output_path=str(snapshot_path), sync_first=sync_first)

        if not self.interactive:
            print(f"\nBackend `{self.backend}` is non-interactive. Snapshot exported to {snapshot_path}")
            return str(snapshot_path)

        fig, axis = self.plt.subplots(figsize=(16, 10))
        image = self.plt.imread(snapshot_path)
        artist = axis.imshow(image)
        axis.axis("off")
        if hasattr(fig.canvas.manager, "set_window_title"):
            fig.canvas.manager.set_window_title("Analytics Dashboard Overview")
        self.plt.tight_layout()
        self.plt.show(block=False)
        self.plt.pause(0.1)

        if not auto_refresh:
            print(f"\nDashboard window opened. Snapshot source: {snapshot_path}")
            return str(snapshot_path)

        print(f"\nAuto-refreshing dashboard every {self.refresh_seconds}s. Close the figure or press Ctrl+C to stop.")
        try:
            while self.plt.fignum_exists(fig.number):
                self.plt.pause(self.refresh_seconds)
                if not self.plt.fignum_exists(fig.number):
                    break
                self.export_overview_snapshot(output_path=str(snapshot_path), sync_first=True)
                updated_image = self.plt.imread(snapshot_path)
                artist.set_data(updated_image)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
        except KeyboardInterrupt:
            print("\nAuto-refresh stopped.")

        return str(snapshot_path)

    def prompt_for_days(self) -> None:
        raw = input(f"New lookback days [current {self.days}]: ").strip()
        if not raw:
            return
        try:
            value = int(raw)
        except ValueError:
            print("Invalid number.")
            return
        if value < 1 or value > 365:
            print("Days must be between 1 and 365.")
            return
        self.days = value
        print(f"Lookback updated to {self.days} days.")

    def prompt_for_refresh(self) -> None:
        raw = input(f"New refresh seconds [current {self.refresh_seconds}]: ").strip()
        if not raw:
            return
        try:
            value = int(raw)
        except ValueError:
            print("Invalid number.")
            return
        if value < 2 or value > 3600:
            print("Refresh seconds must be between 2 and 3600.")
            return
        self.refresh_seconds = value
        print(f"Refresh interval updated to {self.refresh_seconds}s.")

    def run(self) -> int:
        """Run the interactive menu loop."""
        while True:
            print(
                "\nPhase 4 Dashboard Menu\n"
                " 1. Refresh summary\n"
                " 2. Open dashboard overview\n"
                " 3. Auto-refresh dashboard overview\n"
                " 4. Export overview snapshot\n"
                " 5. Export full chart bundle\n"
                " 6. Change lookback days\n"
                " 7. Change refresh interval\n"
                " 8. Sync analytics DB now\n"
                " 9. Quit\n"
            )

            choice = input("Select option: ").strip()

            if choice == "1":
                decision_df, completed_df, sync_result = self.load_latest_data(sync_first=True)
                print(self.build_summary_text(decision_df, completed_df, sync_result))
            elif choice == "2":
                self.show_dashboard(auto_refresh=False, sync_first=True)
            elif choice == "3":
                self.show_dashboard(auto_refresh=True, sync_first=True)
            elif choice == "4":
                result = self.export_overview_snapshot(sync_first=True)
                print(f"\nOverview exported to {result['path']}")
            elif choice == "5":
                result = self.export_full_chart_suite(sync_first=True)
                print(f"\nChart bundle exported to {result['output_dir']}")
            elif choice == "6":
                self.prompt_for_days()
            elif choice == "7":
                self.prompt_for_refresh()
            elif choice == "8":
                print(f"\nSync result: {self.sync_pipeline()}")
            elif choice == "9":
                print("Exiting CLI dashboard.")
                return 0
            else:
                print("Invalid option.")

    def _fmt_ratio(self, value: float) -> str:
        if value == float("inf"):
            return "inf"
        return f"{value:.2f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4 CLI dashboard for analytics and visualization.")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days.")
    parser.add_argument("--refresh-seconds", type=int, default=15, help="Auto-refresh interval for the dashboard window.")
    parser.add_argument("--output-root", type=str, default=None, help="Directory for dashboard exports.")
    parser.add_argument("--backend", type=str, default="Agg", help="Matplotlib backend. Use TkAgg/QtAgg for interactive windows.")
    parser.add_argument("--summary", action="store_true", help="Print a one-shot summary and exit.")
    parser.add_argument("--export-overview", action="store_true", help="Export a one-shot overview PNG and exit.")
    parser.add_argument("--export-bundle", action="store_true", help="Export the full chart bundle and exit.")
    parser.add_argument("--show", action="store_true", help="Open the dashboard overview window and exit.")
    parser.add_argument("--auto-refresh", action="store_true", help="Open the dashboard overview and refresh until closed.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cli = DashboardCLI(
        days=args.days,
        refresh_seconds=args.refresh_seconds,
        output_root=args.output_root,
        backend=args.backend,
    )

    if args.summary:
        decision_df, completed_df, sync_result = cli.load_latest_data(sync_first=True)
        print(cli.build_summary_text(decision_df, completed_df, sync_result))
        return 0

    if args.export_overview:
        result = cli.export_overview_snapshot(sync_first=True)
        print(result["path"])
        return 0

    if args.export_bundle:
        result = cli.export_full_chart_suite(sync_first=True)
        print(result["output_dir"])
        return 0

    if args.auto_refresh:
        cli.show_dashboard(auto_refresh=True, sync_first=True)
        return 0

    if args.show:
        cli.show_dashboard(auto_refresh=False, sync_first=True)
        return 0

    return cli.run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
