"""
Phase 3 visual analytics components for completed trade analysis.

Exports matplotlib figures for:
- Equity curve and drawdown
- Win-rate breakdowns
- Quality score analysis
- Trade duration distributions
- Execution quality metrics
- Filter impact visualization
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))

import matplotlib.pyplot as plt

from .core_analytics import (
    PerformanceMetricsCalculator,
    StatisticalAnalyzer,
    TradeSegmentationEngine,
)
from .data_pipeline import get_completed_trades_only, get_trade_data


@dataclass(frozen=True)
class ChartSpec:
    key: str
    title: str
    filename: str


class VisualAnalyticsEngine:
    """Generate reusable Phase 3 chart assets from trade analytics data."""

    _CHARTS: tuple[ChartSpec, ...] = (
        ChartSpec("equity_curve", "Equity Curve", "equity_curve.png"),
        ChartSpec("win_rate_breakdown", "Win Rate Breakdown", "win_rate_breakdown.png"),
        ChartSpec("quality_score_analysis", "Quality Score Analysis", "quality_score_analysis.png"),
        ChartSpec("trade_duration_distribution", "Trade Duration Distribution", "trade_duration_distribution.png"),
        ChartSpec("execution_quality_metrics", "Execution Quality Metrics", "execution_quality_metrics.png"),
        ChartSpec("filter_impact_visualization", "Filter Impact Visualization", "filter_impact_visualization.png"),
    )

    def __init__(self) -> None:
        self.colors = {
            "profit": "#2E8B57",
            "loss": "#C44536",
            "neutral": "#5C677D",
            "primary": "#1B4965",
            "accent": "#E09F3E",
            "secondary": "#3C91E6",
            "background": "#F7F9FB",
            "grid": "#D8E1E8",
        }
        self._configure_style()

    def _configure_style(self) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": self.colors["background"],
                "axes.edgecolor": self.colors["grid"],
                "axes.labelcolor": "#24303A",
                "axes.titleweight": "bold",
                "grid.color": self.colors["grid"],
                "grid.alpha": 0.45,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.frameon": False,
            }
        )

    def generate_visual_suite(self, days: int = 30, output_dir: Optional[str] = None) -> Dict:
        """Load pipeline data and export the full Phase 3 chart suite."""
        decision_df, completed_df = self.load_trade_frames(days)
        return self.generate_visual_suite_from_dataframes(
            decision_df=decision_df,
            completed_df=completed_df,
            output_dir=output_dir,
            metadata={"period_days": days},
        )

    def load_trade_frames(self, days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load decision and completed-trade dataframes from the pipeline."""
        decision_df = self._prepare_decision_df(get_trade_data(days))
        completed_df = self._prepare_completed_df(get_completed_trades_only(days))
        return decision_df, completed_df

    def generate_visual_suite_from_dataframes(
        self,
        decision_df: pd.DataFrame,
        completed_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Export the full chart bundle from provided dataframes."""
        decision_df = self._prepare_decision_df(decision_df)
        completed_df = (
            self._prepare_completed_df(completed_df)
            if completed_df is not None
            else self._prepare_completed_df(decision_df)
        )
        export_dir = self._resolve_output_dir(output_dir)

        builders = {
            "equity_curve": lambda: self.build_equity_curve(completed_df),
            "win_rate_breakdown": lambda: self.build_win_rate_breakdown(completed_df),
            "quality_score_analysis": lambda: self.build_quality_score_analysis(completed_df),
            "trade_duration_distribution": lambda: self.build_trade_duration_distribution(completed_df),
            "execution_quality_metrics": lambda: self.build_execution_quality_metrics(completed_df),
            "filter_impact_visualization": lambda: self.build_filter_impact_visualization(decision_df, completed_df),
        }

        chart_exports = {}
        for chart in self._CHARTS:
            figure = builders[chart.key]()
            output_path = export_dir / chart.filename
            figure.savefig(output_path, dpi=160, bbox_inches="tight")
            plt.close(figure)
            chart_exports[chart.key] = {
                "title": chart.title,
                "path": str(output_path),
                "has_data": bool(
                    not completed_df.empty if chart.key != "filter_impact_visualization" else not decision_df.empty
                ),
            }

        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(export_dir),
            "chart_count": len(chart_exports),
            "completed_trade_count": int(len(completed_df)),
            "decision_count": int(len(decision_df)),
            "charts": chart_exports,
        }
        if metadata:
            result.update(metadata)
        return result

    def export_dashboard_snapshot(
        self,
        decision_df: pd.DataFrame,
        completed_df: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None,
        days: Optional[int] = None,
    ) -> Dict:
        """Export a single overview dashboard figure."""
        decision_df = self._prepare_decision_df(decision_df)
        completed_df = (
            self._prepare_completed_df(completed_df)
            if completed_df is not None
            else self._prepare_completed_df(decision_df)
        )

        target_path = Path(output_path) if output_path else self._resolve_output_dir(None) / "dashboard_overview.png"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        figure = self.build_dashboard_overview(decision_df=decision_df, completed_df=completed_df, days=days)
        figure.savefig(target_path, dpi=170, bbox_inches="tight")
        plt.close(figure)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "path": str(target_path),
            "completed_trade_count": int(len(completed_df)),
            "decision_count": int(len(decision_df)),
            "period_days": days,
        }

    def generate_dashboard_snapshot(self, days: int = 30, output_path: Optional[str] = None) -> Dict:
        """Load pipeline data and export a single overview dashboard PNG."""
        decision_df, completed_df = self.load_trade_frames(days)
        return self.export_dashboard_snapshot(
            decision_df=decision_df,
            completed_df=completed_df,
            output_path=output_path,
            days=days,
        )

    def build_dashboard_overview(
        self,
        decision_df: pd.DataFrame,
        completed_df: Optional[pd.DataFrame] = None,
        days: Optional[int] = None,
    ):
        """Generate a single overview dashboard for CLI viewing and export."""
        decision_df = self._prepare_decision_df(decision_df)
        completed_df = (
            self._prepare_completed_df(completed_df)
            if completed_df is not None
            else self._prepare_completed_df(decision_df)
        )

        fig = plt.figure(figsize=(16, 10))
        grid = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.25, 1.25], hspace=0.34, wspace=0.28)

        summary_ax = fig.add_subplot(grid[0, 0])
        equity_ax = fig.add_subplot(grid[0, 1:])
        direction_ax = fig.add_subplot(grid[1, 0])
        session_ax = fig.add_subplot(grid[1, 1])
        quality_ax = fig.add_subplot(grid[1, 2])
        duration_ax = fig.add_subplot(grid[2, 0])
        execution_ax = fig.add_subplot(grid[2, 1])
        filter_ax = fig.add_subplot(grid[2, 2])

        fig.suptitle(
            f"Analytics Dashboard Overview{f' - Last {days} Days' if days else ''}",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )
        fig.text(
            0.995,
            0.985,
            datetime.now(timezone.utc).strftime("UTC %Y-%m-%d %H:%M:%S"),
            ha="right",
            va="top",
            fontsize=10,
            color=self.colors["neutral"],
        )

        if completed_df.empty:
            self._render_overview_empty_state(summary_ax, equity_ax, direction_ax, session_ax, quality_ax, duration_ax, execution_ax, filter_ax, decision_df)
            fig.subplots_adjust(top=0.92, bottom=0.06, left=0.05, right=0.98)
            return fig

        metrics = PerformanceMetricsCalculator.calculate_basic_metrics(completed_df)
        direction_metrics = PerformanceMetricsCalculator.calculate_directional_metrics(completed_df)
        session_metrics = PerformanceMetricsCalculator.calculate_session_metrics(completed_df)
        quality_segments = TradeSegmentationEngine.segment_by_quality_score(completed_df)
        duration_stats = StatisticalAnalyzer.calculate_trade_duration_stats(completed_df.dropna(subset=["trade_duration"]))

        self._draw_summary_panel(summary_ax, metrics, completed_df, decision_df)
        self._draw_overview_equity(equity_ax, completed_df)
        self._draw_overview_direction(direction_ax, direction_metrics)
        self._draw_overview_session(session_ax, session_metrics)
        self._draw_overview_quality(quality_ax, completed_df, quality_segments)
        self._draw_overview_duration(duration_ax, completed_df, duration_stats)
        self._draw_overview_execution(execution_ax, completed_df)
        self._draw_overview_filter(filter_ax, decision_df, completed_df)

        fig.subplots_adjust(top=0.92, bottom=0.06, left=0.05, right=0.98)
        return fig

    def build_equity_curve(self, df: pd.DataFrame):
        """Generate equity curve and drawdown chart."""
        df = self._prepare_completed_df(df)
        if df.empty:
            return self._empty_figure("Equity Curve", "No completed trades available.")

        metrics = PerformanceMetricsCalculator.calculate_basic_metrics(df)
        ordered = df.sort_values("unix_time").reset_index(drop=True)
        cumulative_pnl = ordered["pnl"].fillna(0).cumsum()
        running_peak = cumulative_pnl.cummax()
        drawdown = running_peak - cumulative_pnl
        x_axis, x_label = self._x_axis_values(ordered)

        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[3, 1])
        top_ax, bottom_ax = axes

        top_ax.plot(x_axis, cumulative_pnl, color=self.colors["primary"], linewidth=2.2)
        top_ax.fill_between(
            x_axis,
            cumulative_pnl,
            0,
            where=cumulative_pnl >= 0,
            color=self.colors["profit"],
            alpha=0.12,
        )
        top_ax.fill_between(
            x_axis,
            cumulative_pnl,
            0,
            where=cumulative_pnl < 0,
            color=self.colors["loss"],
            alpha=0.12,
        )
        top_ax.axhline(0, color=self.colors["neutral"], linestyle="--", linewidth=1)
        top_ax.set_title("Equity Curve")
        top_ax.set_ylabel("Cumulative PnL")

        summary = (
            f"Trades: {metrics['total_trades']}   "
            f"Win rate: {metrics['win_rate']:.1%}   "
            f"Total PnL: {metrics['total_pnl']:.2f}   "
            f"Max DD: {metrics['max_drawdown']:.2f}"
        )
        top_ax.text(
            0.01,
            0.98,
            summary,
            transform=top_ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": self.colors["grid"], "alpha": 0.9},
        )

        bottom_ax.fill_between(x_axis, drawdown, color=self.colors["loss"], alpha=0.25)
        bottom_ax.plot(x_axis, drawdown, color=self.colors["loss"], linewidth=1.8)
        bottom_ax.set_ylabel("Drawdown")
        bottom_ax.set_xlabel(x_label)
        bottom_ax.set_title("Peak-to-Trough Drawdown")

        self._format_time_axis(bottom_ax, x_axis)
        fig.tight_layout()
        return fig

    def build_win_rate_breakdown(self, df: pd.DataFrame):
        """Generate direction/session/trend win-rate comparison charts."""
        df = self._prepare_completed_df(df)
        if df.empty:
            return self._empty_figure("Win Rate Breakdown", "No completed trades available.")

        direction_metrics = PerformanceMetricsCalculator.calculate_directional_metrics(df)
        session_metrics = PerformanceMetricsCalculator.calculate_session_metrics(df)
        bias_metrics = PerformanceMetricsCalculator.calculate_bias_alignment_metrics(df)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self._plot_rate_bar(
            axes[0],
            labels=["LONG", "SHORT"],
            rates=[
                direction_metrics["long"].get("win_rate", 0),
                direction_metrics["short"].get("win_rate", 0),
            ],
            counts=[
                direction_metrics["long"].get("trade_count", 0),
                direction_metrics["short"].get("trade_count", 0),
            ],
            title="By Direction",
        )
        self._plot_rate_bar(
            axes[1],
            labels=list(session_metrics["sessions"].keys()),
            rates=[v.get("win_rate", 0) for v in session_metrics["sessions"].values()],
            counts=[v.get("trade_count", 0) for v in session_metrics["sessions"].values()],
            title="By Session",
        )
        self._plot_rate_bar(
            axes[2],
            labels=["WITH_TREND", "COUNTER_TREND"],
            rates=[
                bias_metrics["with_trend"].get("win_rate", 0),
                bias_metrics["counter_trend"].get("win_rate", 0),
            ],
            counts=[
                bias_metrics["with_trend"].get("trade_count", 0),
                bias_metrics["counter_trend"].get("trade_count", 0),
            ],
            title="By Bias Alignment",
        )
        fig.suptitle("Win Rate Breakdown", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    def build_quality_score_analysis(self, df: pd.DataFrame):
        """Generate quality score scatter and segment effectiveness charts."""
        df = self._prepare_completed_df(df)
        if df.empty:
            return self._empty_figure("Quality Score Analysis", "No completed trades available.")

        filtered = df.dropna(subset=["quality_score", "pnl"])
        if filtered.empty:
            return self._empty_figure("Quality Score Analysis", "Quality score data is missing.")

        segments = TradeSegmentationEngine.segment_by_quality_score(filtered)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        outcome_colors = filtered["outcome"].map(
            {"WIN": self.colors["profit"], "LOSS": self.colors["loss"], "BE": self.colors["neutral"]}
        ).fillna(self.colors["secondary"])
        axes[0].scatter(
            filtered["quality_score"],
            filtered["pnl"],
            c=outcome_colors,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            s=52,
        )
        axes[0].axhline(0, color=self.colors["neutral"], linestyle="--", linewidth=1)
        axes[0].set_title("Quality Score vs PnL")
        axes[0].set_xlabel("Quality Score")
        axes[0].set_ylabel("PnL")

        segment_labels = list(segments.keys())
        win_rates = [segments[key].get("win_rate", 0) for key in segment_labels]
        avg_pnls = [segments[key].get("avg_pnl", 0) for key in segment_labels]
        counts = [segments[key].get("trade_count", 0) for key in segment_labels]
        x_positions = np.arange(len(segment_labels))

        rate_bars = axes[1].bar(
            x_positions,
            win_rates,
            color=self.colors["secondary"],
            alpha=0.85,
            label="Win rate",
        )
        pnl_ax = axes[1].twinx()
        pnl_ax.plot(
            x_positions,
            avg_pnls,
            color=self.colors["accent"],
            marker="o",
            linewidth=2,
            label="Avg PnL",
        )
        axes[1].set_xticks(x_positions)
        axes[1].set_xticklabels(segment_labels)
        axes[1].set_ylim(0, 1.0)
        axes[1].set_ylabel("Win Rate")
        pnl_ax.set_ylabel("Average PnL")
        axes[1].set_title("Segment Performance by Score Range")

        for bar, count in zip(rate_bars, counts):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                min(bar.get_height() + 0.03, 0.97),
                f"n={count}",
                ha="center",
                fontsize=9,
            )

        handles = [rate_bars, pnl_ax.lines[0]]
        axes[1].legend(handles, ["Win rate", "Avg PnL"], loc="upper left")
        fig.tight_layout()
        return fig

    def build_trade_duration_distribution(self, df: pd.DataFrame):
        """Generate distribution and box plot views for trade durations."""
        df = self._prepare_completed_df(df)
        if df.empty:
            return self._empty_figure("Trade Duration Distribution", "No completed trades available.")

        filtered = df.dropna(subset=["trade_duration"])
        if filtered.empty:
            return self._empty_figure("Trade Duration Distribution", "Trade duration data is missing.")

        winners = filtered[filtered["outcome"] == "WIN"]["trade_duration"]
        losers = filtered[filtered["outcome"] == "LOSS"]["trade_duration"]
        be_trades = filtered[filtered["outcome"] == "BE"]["trade_duration"]
        duration_stats = StatisticalAnalyzer.calculate_trade_duration_stats(filtered)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        bins = min(max(int(np.sqrt(len(filtered))), 5), 18)
        if not winners.empty:
            axes[0].hist(winners, bins=bins, alpha=0.65, color=self.colors["profit"], label="WIN")
        if not losers.empty:
            axes[0].hist(losers, bins=bins, alpha=0.65, color=self.colors["loss"], label="LOSS")
        if not be_trades.empty:
            axes[0].hist(be_trades, bins=bins, alpha=0.65, color=self.colors["neutral"], label="BE")
        axes[0].set_title("Duration Histogram")
        axes[0].set_xlabel("Trade Duration (seconds)")
        axes[0].set_ylabel("Trade Count")
        axes[0].legend(loc="upper right")

        box_data = []
        box_labels = []
        if not winners.empty:
            box_data.append(winners.values)
            box_labels.append("WIN")
        if not losers.empty:
            box_data.append(losers.values)
            box_labels.append("LOSS")
        if not be_trades.empty:
            box_data.append(be_trades.values)
            box_labels.append("BE")

        axes[1].boxplot(
            box_data if box_data else [filtered["trade_duration"].values],
            tick_labels=box_labels if box_labels else ["ALL"],
            patch_artist=True,
            boxprops={"facecolor": "#CFE8F6", "edgecolor": self.colors["primary"]},
            medianprops={"color": self.colors["loss"], "linewidth": 2},
        )
        axes[1].set_title("Duration by Outcome")
        axes[1].set_ylabel("Trade Duration (seconds)")

        stats_label = (
            f"Median: {duration_stats['all_trades']['median']:.0f}s\n"
            f"Mean: {duration_stats['all_trades']['mean']:.0f}s\n"
            f"Max: {duration_stats['all_trades']['max']:.0f}s"
        )
        axes[1].text(
            0.98,
            0.98,
            stats_label,
            transform=axes[1].transAxes,
            va="top",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": self.colors["grid"], "alpha": 0.9},
        )

        fig.tight_layout()
        return fig

    def build_execution_quality_metrics(self, df: pd.DataFrame):
        """Generate execution-quality views from spread/tick metrics."""
        df = self._prepare_completed_df(df)
        if df.empty:
            return self._empty_figure("Execution Quality Metrics", "No completed trades available.")

        filtered = df.dropna(subset=["pnl"], how="any")
        if filtered.empty:
            return self._empty_figure("Execution Quality Metrics", "Execution quality fields are missing.")

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        scatter_definitions = [
            ("spread_mean", "Spread Mean vs PnL", axes[0, 0]),
            ("tick_ratio", "Tick Ratio vs PnL", axes[0, 1]),
            ("tick_velocity", "Tick Velocity vs PnL", axes[1, 0]),
            ("spread_percentile", "Spread Percentile vs PnL", axes[1, 1]),
        ]

        for field, title, axis in scatter_definitions:
            chart_df = filtered.dropna(subset=[field]) if field in filtered.columns else pd.DataFrame()
            if chart_df.empty:
                axis.text(0.5, 0.5, f"No {field} data", ha="center", va="center", transform=axis.transAxes)
                axis.set_title(title)
                continue

            colors = chart_df["outcome"].map(
                {"WIN": self.colors["profit"], "LOSS": self.colors["loss"], "BE": self.colors["neutral"]}
            ).fillna(self.colors["secondary"])
            axis.scatter(
                chart_df[field],
                chart_df["pnl"],
                c=colors,
                alpha=0.72,
                s=46,
                edgecolors="white",
                linewidths=0.4,
            )
            axis.axhline(0, color=self.colors["neutral"], linestyle="--", linewidth=1)
            axis.set_title(title)
            axis.set_xlabel(field.replace("_", " ").title())
            axis.set_ylabel("PnL")

        fig.suptitle("Execution Quality Metrics", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    def build_filter_impact_visualization(self, decision_df: pd.DataFrame, completed_df: Optional[pd.DataFrame] = None):
        """Generate skip-reason and decision-conversion visualizations."""
        decision_df = self._prepare_decision_df(decision_df)
        completed_df = (
            self._prepare_completed_df(completed_df)
            if completed_df is not None
            else self._prepare_completed_df(decision_df)
        )

        if decision_df.empty:
            return self._empty_figure("Filter Impact Visualization", "No decision data available.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        skipped = decision_df[decision_df["decision"] == "TRADE_SKIPPED"].copy()
        if skipped.empty or "reason" not in skipped.columns:
            axes[0].text(0.5, 0.5, "No skipped-trade reasons available", ha="center", va="center", transform=axes[0].transAxes)
            axes[0].set_title("Top Skip Reasons")
        else:
            skip_reasons = (
                skipped["reason"]
                .fillna("Unknown")
                .map(lambda value: str(value).split("|")[0].strip() or "Unknown")
                .value_counts()
                .head(6)
                .sort_values()
            )
            axes[0].barh(skip_reasons.index, skip_reasons.values, color=self.colors["loss"], alpha=0.85)
            axes[0].set_title("Top Skip Reasons")
            axes[0].set_xlabel("Count")

        decision_pivot = (
            decision_df.assign(session=decision_df["session"].fillna("UNKNOWN"))
            .pivot_table(index="session", columns="decision", values="unix_time", aggfunc="count", fill_value=0)
            .sort_index()
        )
        if decision_pivot.empty:
            axes[1].text(0.5, 0.5, "No session decision data available", ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title("Decision Flow by Session")
        else:
            taken = decision_pivot.get("TRADE_TAKEN", pd.Series(0, index=decision_pivot.index))
            skipped_counts = decision_pivot.get("TRADE_SKIPPED", pd.Series(0, index=decision_pivot.index))
            positions = np.arange(len(decision_pivot.index))
            axes[1].bar(positions, taken.values, color=self.colors["profit"], label="Taken")
            axes[1].bar(positions, skipped_counts.values, bottom=taken.values, color=self.colors["loss"], label="Skipped")
            axes[1].set_xticks(positions)
            axes[1].set_xticklabels(decision_pivot.index, rotation=20)
            axes[1].set_ylabel("Signal Count")
            axes[1].set_title("Decision Flow by Session")
            axes[1].legend(loc="upper right")

        conversion_rate = 0.0
        if len(decision_df) > 0:
            conversion_rate = float((decision_df["decision"] == "TRADE_TAKEN").mean())
        completed_rate = 0.0
        taken_count = int((decision_df["decision"] == "TRADE_TAKEN").sum())
        if taken_count > 0:
            completed_rate = len(completed_df) / taken_count

        fig.text(
            0.5,
            0.01,
            (
                f"Conversion rate: {conversion_rate:.1%}   "
                f"Taken trades: {taken_count}   "
                f"Completed-trade coverage: {completed_rate:.1%}"
            ),
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        return fig

    def _draw_summary_panel(self, axis, metrics: Dict, completed_df: pd.DataFrame, decision_df: pd.DataFrame) -> None:
        axis.set_title("Performance Snapshot")
        axis.axis("off")

        conversion_rate = float((decision_df["decision"] == "TRADE_TAKEN").mean()) if len(decision_df) > 0 and "decision" in decision_df.columns else 0.0
        rows = [
            ("Completed Trades", f"{metrics.get('total_trades', 0)}"),
            ("Win Rate", f"{metrics.get('win_rate', 0):.1%}"),
            ("Total PnL", f"{metrics.get('total_pnl', 0):.2f}"),
            ("Profit Factor", self._fmt_ratio(metrics.get("profit_factor", 0))),
            ("Avg Win / Loss", f"{metrics.get('avg_win', 0):.2f} / {metrics.get('avg_loss', 0):.2f}"),
            ("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}"),
            ("Best / Worst", f"{metrics.get('best_trade', 0):.2f} / {metrics.get('worst_trade', 0):.2f}"),
            ("Decision Conversion", f"{conversion_rate:.1%}"),
        ]

        for idx, (label, value) in enumerate(rows):
            y = 0.93 - (idx * 0.105)
            axis.text(0.02, y, label, fontsize=10, color=self.colors["neutral"], transform=axis.transAxes)
            axis.text(0.98, y, value, fontsize=11, fontweight="bold", ha="right", transform=axis.transAxes)

    def _draw_overview_equity(self, axis, completed_df: pd.DataFrame) -> None:
        ordered = completed_df.sort_values("unix_time").reset_index(drop=True)
        cumulative_pnl = ordered["pnl"].fillna(0).cumsum()
        running_peak = cumulative_pnl.cummax()
        drawdown = running_peak - cumulative_pnl
        x_axis, x_label = self._x_axis_values(ordered)

        axis.plot(x_axis, cumulative_pnl, color=self.colors["primary"], linewidth=2.2, label="Equity")
        axis.fill_between(x_axis, cumulative_pnl, 0, color=self.colors["secondary"], alpha=0.12)
        axis.plot(x_axis, -drawdown, color=self.colors["loss"], linewidth=1.5, linestyle="--", label="-Drawdown")
        axis.axhline(0, color=self.colors["neutral"], linewidth=1, linestyle=":")
        axis.set_title("Equity and Drawdown")
        axis.set_xlabel(x_label)
        axis.set_ylabel("PnL")
        axis.legend(loc="upper left")
        self._format_time_axis(axis, x_axis)

    def _draw_overview_direction(self, axis, direction_metrics: Dict) -> None:
        labels = ["LONG", "SHORT"]
        rates = [
            direction_metrics["long"].get("win_rate", 0),
            direction_metrics["short"].get("win_rate", 0),
        ]
        counts = [
            direction_metrics["long"].get("trade_count", 0),
            direction_metrics["short"].get("trade_count", 0),
        ]
        self._plot_rate_bar(axis, labels, rates, counts, "Directional Win Rate")

    def _draw_overview_session(self, axis, session_metrics: Dict) -> None:
        labels = list(session_metrics["sessions"].keys())
        rates = [entry.get("win_rate", 0) for entry in session_metrics["sessions"].values()]
        counts = [entry.get("trade_count", 0) for entry in session_metrics["sessions"].values()]
        self._plot_rate_bar(axis, labels, rates, counts, "Session Win Rate")

    def _draw_overview_quality(self, axis, completed_df: pd.DataFrame, quality_segments: Dict) -> None:
        filtered = completed_df.dropna(subset=["quality_score", "pnl"])
        axis.set_title("Quality Score Edge")
        if filtered.empty:
            axis.text(0.5, 0.5, "No quality score data", ha="center", va="center", transform=axis.transAxes)
            return

        axis.scatter(
            filtered["quality_score"],
            filtered["pnl"],
            c=filtered["outcome"].map({"WIN": self.colors["profit"], "LOSS": self.colors["loss"], "BE": self.colors["neutral"]}).fillna(self.colors["secondary"]),
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidths=0.4,
        )
        axis.axhline(0, color=self.colors["neutral"], linewidth=1, linestyle=":")
        axis.set_xlabel("Quality Score")
        axis.set_ylabel("PnL")

        best_segment = max(
            (entry for entry in quality_segments.values() if entry.get("trade_count", 0) > 0),
            key=lambda entry: entry.get("win_rate", 0),
            default=None,
        )
        if best_segment:
            axis.text(
                0.98,
                0.98,
                f"Best: {best_segment['range']}\nWR {best_segment.get('win_rate', 0):.0%}",
                ha="right",
                va="top",
                transform=axis.transAxes,
                bbox={"facecolor": "white", "edgecolor": self.colors["grid"], "alpha": 0.9},
            )

    def _draw_overview_duration(self, axis, completed_df: pd.DataFrame, duration_stats: Dict) -> None:
        filtered = completed_df.dropna(subset=["trade_duration"])
        axis.set_title("Trade Duration")
        if filtered.empty:
            axis.text(0.5, 0.5, "No duration data", ha="center", va="center", transform=axis.transAxes)
            return

        winners = filtered[filtered["outcome"] == "WIN"]["trade_duration"]
        losers = filtered[filtered["outcome"] == "LOSS"]["trade_duration"]
        bins = min(max(int(np.sqrt(len(filtered))), 5), 16)
        if not winners.empty:
            axis.hist(winners, bins=bins, alpha=0.65, color=self.colors["profit"], label="WIN")
        if not losers.empty:
            axis.hist(losers, bins=bins, alpha=0.65, color=self.colors["loss"], label="LOSS")
        axis.set_xlabel("Seconds")
        axis.set_ylabel("Trades")
        axis.legend(loc="upper right")
        if duration_stats and "all_trades" in duration_stats:
            axis.text(
                0.98,
                0.98,
                f"Median {duration_stats['all_trades']['median']:.0f}s\nMean {duration_stats['all_trades']['mean']:.0f}s",
                ha="right",
                va="top",
                transform=axis.transAxes,
                bbox={"facecolor": "white", "edgecolor": self.colors["grid"], "alpha": 0.9},
            )

    def _draw_overview_execution(self, axis, completed_df: pd.DataFrame) -> None:
        axis.set_title("Execution Quality")
        chart_df = completed_df.dropna(subset=["tick_ratio", "spread_mean", "pnl"], how="any")
        if chart_df.empty:
            axis.text(0.5, 0.5, "No execution metrics", ha="center", va="center", transform=axis.transAxes)
            return

        sc = axis.scatter(
            chart_df["tick_ratio"],
            chart_df["spread_mean"],
            c=chart_df["pnl"],
            cmap="RdYlGn",
            s=np.clip(np.abs(chart_df["pnl"].fillna(0)) * 10 + 40, 40, 180),
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
        )
        axis.set_xlabel("Tick Ratio")
        axis.set_ylabel("Spread Mean")
        colorbar = axis.figure.colorbar(sc, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("PnL")

    def _draw_overview_filter(self, axis, decision_df: pd.DataFrame, completed_df: pd.DataFrame) -> None:
        axis.set_title("Filter Impact")
        if decision_df.empty or "decision" not in decision_df.columns:
            axis.text(0.5, 0.5, "No decision data", ha="center", va="center", transform=axis.transAxes)
            return

        skipped = decision_df[decision_df["decision"] == "TRADE_SKIPPED"].copy()
        conversion_rate = float((decision_df["decision"] == "TRADE_TAKEN").mean()) if len(decision_df) > 0 else 0.0
        taken_count = int((decision_df["decision"] == "TRADE_TAKEN").sum())

        if skipped.empty or "reason" not in skipped.columns:
            axis.axis("off")
            axis.text(0.5, 0.55, f"Conversion Rate\n{conversion_rate:.1%}", ha="center", va="center", fontsize=16, fontweight="bold", transform=axis.transAxes)
            axis.text(0.5, 0.25, f"Taken: {taken_count}\nCompleted: {len(completed_df)}", ha="center", va="center", fontsize=11, color=self.colors["neutral"], transform=axis.transAxes)
            return

        skip_reasons = (
            skipped["reason"]
            .fillna("Unknown")
            .map(lambda value: str(value).split("|")[0].strip() or "Unknown")
            .value_counts()
            .head(4)
            .sort_values()
        )
        axis.barh(skip_reasons.index, skip_reasons.values, color=self.colors["loss"], alpha=0.85)
        axis.set_xlabel("Skip Count")
        axis.text(
            0.98,
            0.06,
            f"Conversion {conversion_rate:.1%}\nTaken {taken_count} | Done {len(completed_df)}",
            ha="right",
            va="bottom",
            transform=axis.transAxes,
            bbox={"facecolor": "white", "edgecolor": self.colors["grid"], "alpha": 0.9},
        )

    def _render_overview_empty_state(self, summary_ax, equity_ax, direction_ax, session_ax, quality_ax, duration_ax, execution_ax, filter_ax, decision_df: pd.DataFrame) -> None:
        for axis in [summary_ax, equity_ax, direction_ax, session_ax, quality_ax, duration_ax, execution_ax]:
            axis.clear()
            axis.axis("off")
            axis.text(0.5, 0.5, "No completed trade data available", ha="center", va="center", transform=axis.transAxes)

        summary_ax.set_title("Performance Snapshot")
        if decision_df.empty:
            filter_ax.clear()
            filter_ax.axis("off")
            filter_ax.text(0.5, 0.5, "No decision data available", ha="center", va="center", transform=filter_ax.transAxes)
            filter_ax.set_title("Filter Impact")
            return

        self._draw_overview_filter(filter_ax, decision_df, self._prepare_completed_df(decision_df))

    def _fmt_ratio(self, value: float) -> str:
        if value == float("inf"):
            return "inf"
        return f"{value:.2f}"

    def _plot_rate_bar(self, axis, labels: Iterable[str], rates: Iterable[float], counts: Iterable[int], title: str) -> None:
        label_list = list(labels)
        rate_list = list(rates)
        count_list = list(counts)
        colors = [
            self.colors["profit"]
            if rate >= 0.55
            else self.colors["accent"]
            if rate >= 0.45
            else self.colors["loss"]
            for rate in rate_list
        ]
        positions = np.arange(len(label_list))
        bars = axis.bar(positions, rate_list, color=colors, alpha=0.9)
        axis.set_xticks(positions)
        axis.set_xticklabels(label_list, rotation=15)
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("Win Rate")
        axis.set_title(title)

        for bar, rate, count in zip(bars, rate_list, count_list):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(rate + 0.04, 0.97),
                f"{rate:.0%}\nn={count}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    def _prepare_decision_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()

        prepared = df.copy()
        if "timestamp" in prepared.columns:
            prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        if "completion_time" in prepared.columns:
            prepared["completion_time"] = pd.to_datetime(prepared["completion_time"], errors="coerce")
        if "unix_time" in prepared.columns:
            prepared = prepared.sort_values("unix_time").reset_index(drop=True)
        return prepared

    def _prepare_completed_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()

        prepared = self._prepare_decision_df(df)
        if "decision" in prepared.columns:
            prepared = prepared[prepared["decision"] == "TRADE_TAKEN"]
        if "outcome" in prepared.columns:
            prepared = prepared.dropna(subset=["outcome"])
        if "unix_time" in prepared.columns:
            prepared = prepared.sort_values("unix_time").reset_index(drop=True)
        return prepared

    def _resolve_output_dir(self, output_dir: Optional[str]) -> Path:
        if output_dir:
            export_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = Path("analytics_outputs") / f"phase3_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir

    def _empty_figure(self, title: str, message: str):
        fig, axis = plt.subplots(figsize=(10, 4.5))
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        return fig

    def _x_axis_values(self, df: pd.DataFrame):
        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
            if timestamps.notna().any():
                return timestamps, "Time"
        return np.arange(1, len(df) + 1), "Trade Number"

    def _format_time_axis(self, axis, x_axis) -> None:
        if len(x_axis) == 0:
            return
        if np.issubdtype(np.array(x_axis).dtype, np.datetime64):
            axis.tick_params(axis="x", rotation=25)


visual_analytics_engine = VisualAnalyticsEngine()


def generate_visual_suite(days: int = 30, output_dir: Optional[str] = None) -> Dict:
    """Generate the full Phase 3 visualization bundle from pipeline data."""
    return visual_analytics_engine.generate_visual_suite(days=days, output_dir=output_dir)


def generate_dashboard_snapshot(days: int = 30, output_path: Optional[str] = None) -> Dict:
    """Generate a single overview dashboard PNG from pipeline data."""
    return visual_analytics_engine.generate_dashboard_snapshot(days=days, output_path=output_path)
