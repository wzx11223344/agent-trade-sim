"""
viz.py — Real-Time Visualization and Report Generation
========================================================
Provides functions for generating charts, animations, and a comprehensive
HTML report from simulation data. All visualizations use Matplotlib for
static charts and Jinja2 for HTML templates.
"""

import math
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker
import numpy as np

# Try to import Jinja2 for HTML report generation
try:
    from jinja2 import Template
    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False


# ------------------------------------------------------------------
# Color Scheme
# ------------------------------------------------------------------

COLORS = {
    "bg": "#0d1117",
    "fg": "#c9d1d9",
    "accent_blue": "#58a6ff",
    "accent_green": "#3fb950",
    "accent_red": "#f85149",
    "accent_orange": "#d2991d",
    "accent_purple": "#bc8cff",
    "accent_cyan": "#39d2c0",
    "grid": "#21262d",
    "panel_bg": "#161b22",
    "buy": "#3fb950",
    "sell": "#f85149",
    "volume": "#58a6ff",
    "value_agent": "#3fb950",
    "momentum_agent": "#58a6ff",
    "noise_agent": "#8b949e",
    "mm_agent": "#d2991d",
}

AGENT_COLORS = {
    "value": COLORS["accent_green"],
    "momentum": COLORS["accent_blue"],
    "noise": COLORS["fg"],
    "market_maker": COLORS["accent_orange"],
}

AGENT_LABELS_CN = {
    "value": "Value Investor",
    "momentum": "Momentum",
    "noise": "Noise Trader",
    "market_maker": "Market Maker",
}


def set_style():
    """Apply the dark style for all Matplotlib charts."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["fg"],
        "axes.titlecolor": COLORS["fg"],
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": COLORS["grid"],
        "text.color": COLORS["fg"],
        "xtick.color": COLORS["fg"],
        "ytick.color": COLORS["fg"],
        "legend.facecolor": COLORS["panel_bg"],
        "legend.edgecolor": COLORS["grid"],
        "legend.labelcolor": COLORS["fg"],
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def live_chart(state: dict, output_path: str) -> str:
    """Generate a single-frame chart showing current market state.

    Args:
        state: Market state dict with price history, spread, etc.
        output_path: Path to save the PNG image.

    Returns:
        The output_path string.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]})

    price_history = state.get("price_history", [])
    tick = state.get("tick", 0)

    if price_history:
        x = range(max(0, len(price_history) - 200), len(price_history))
        y = price_history[-200:] if len(price_history) > 200 else price_history
        ax1.plot(x, y, color=COLORS["accent_blue"], linewidth=1.5, label="Mid Price")
        ax1.fill_between(x, y, min(y) * 0.99, alpha=0.1, color=COLORS["accent_blue"])

    ax1.set_title(f"Market Price — Tick {tick}", fontsize=13, fontweight="bold", color=COLORS["accent_blue"])
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=9)

    # Volume bar
    vol = state.get("volume", 0)
    imbalance = state.get("order_imbalance", 0.5)
    bar_color = COLORS["buy"] if imbalance > 0.5 else COLORS["sell"]
    ax2.bar(0, vol, color=bar_color, alpha=0.8, width=0.4)
    ax2.set_xlim(-1, 1)
    ax2.set_title(f"Volume: {vol:.0f}  |  Imbalance: {imbalance:.3f}", fontsize=10, color=COLORS["fg"])
    ax2.set_xticks([])

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def price_volume_chart(
    price_history: list[float],
    volume_profile: list[float],
    output_path: str,
) -> str:
    """Generate a price chart with volume overlay.

    Top panel: price line with fill area.
    Bottom panel: volume bars colored by price change direction.

    Args:
        price_history: List of mid prices per tick.
        volume_profile: List of volume per tick.
        output_path: Path to save the PNG image.

    Returns:
        The output_path string.
    """
    set_style()
    n = min(len(price_history), len(volume_profile))
    if n == 0:
        return output_path

    prices = price_history[-n:]
    volumes = volume_profile[-n:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]})

    x = range(n)
    ax1.plot(x, prices, color=COLORS["accent_blue"], linewidth=1.2)
    ax1.fill_between(x, prices, min(prices) * 0.99, alpha=0.08, color=COLORS["accent_blue"])
    ax1.set_title("Price History with Volume Profile", fontsize=14, fontweight="bold", color=COLORS["accent_blue"])
    ax1.set_ylabel("Price")

    # Color volume bars by price change
    if n > 1:
        price_changes = np.diff(prices)
        colors_vol = []
        for i in range(1, n):
            colors_vol.append(COLORS["buy"] if price_changes[i - 1] >= 0 else COLORS["sell"])
        colors_vol.append(COLORS["fg"])
    else:
        colors_vol = [COLORS["fg"]] * n

    ax2.bar(x, volumes, color=colors_vol, alpha=0.7, width=1.0)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Tick")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def agent_pnl_chart(
    agents_detail: list[dict],
    price_history: list[float],
    output_path: str,
) -> str:
    """Generate a P&L tracking chart by agent type.

    Shows cumulative P&L evolution for each agent type as stacked
    area or individual lines.

    Args:
        agents_detail: List of per-agent stats dicts.
        price_history: List of prices for reference axis.
        output_path: Path to save the PNG image.

    Returns:
        The output_path string.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar chart of total P&L per agent type
    type_pnl: dict[str, float] = {}
    type_colors: list[str] = []
    type_labels: list[str] = []
    for a in agents_detail:
        atype = a["agent_type"]
        type_pnl[atype] = type_pnl.get(atype, 0) + a["total_pnl"]

    for atype in ["value", "momentum", "noise", "market_maker"]:
        if atype in type_pnl:
            type_labels.append(AGENT_LABELS_CN.get(atype, atype))
            type_colors.append(AGENT_COLORS.get(atype, COLORS["fg"]))

    pnl_values = [type_pnl.get(t, 0) for t in ["value", "momentum", "noise", "market_maker"] if t in type_pnl]
    bars = ax1.bar(type_labels, pnl_values, color=type_colors[:len(type_labels)], alpha=0.85, edgecolor=COLORS["grid"])
    ax1.set_title("Total P&L by Agent Type", fontsize=13, fontweight="bold", color=COLORS["accent_blue"])
    ax1.set_ylabel("P&L ($)")
    ax1.axhline(y=0, color=COLORS["fg"], linewidth=0.5, linestyle="--")
    for bar, val in zip(bars, pnl_values):
        color = COLORS["buy"] if val >= 0 else COLORS["sell"]
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"${val:+.0f}", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=10, fontweight="bold", color=color)

    # Right: Win rate per type
    type_winrate: dict[str, float] = {}
    type_counts: dict[str, tuple[int, int]] = {}
    for a in agents_detail:
        atype = a["agent_type"]
        w, t = type_counts.get(atype, (0, 0))
        if a["total_pnl"] > 0:
            w += 1
        type_counts[atype] = (w, t + 1)

    for atype, (w, t) in type_counts.items():
        type_winrate[atype] = (w / t * 100) if t > 0 else 0

    wr_labels = [AGENT_LABELS_CN.get(t, t) for t in ["value", "momentum", "noise", "market_maker"] if t in type_winrate]
    wr_values = [type_winrate[t] for t in ["value", "momentum", "noise", "market_maker"] if t in type_winrate]
    wr_colors = [AGENT_COLORS.get(t, COLORS["fg"]) for t in ["value", "momentum", "noise", "market_maker"] if t in type_winrate]

    bars2 = ax2.barh(wr_labels, wr_values, color=wr_colors, alpha=0.85, edgecolor=COLORS["grid"])
    ax2.set_title("Win Rate by Agent Type", fontsize=13, fontweight="bold", color=COLORS["accent_green"])
    ax2.set_xlabel("Win Rate (%)")
    ax2.set_xlim(0, 100)
    for bar, val in zip(bars2, wr_values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=10, fontweight="bold",
                 color=COLORS["fg"])

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def order_book_snapshot(
    depth: dict,
    tick: int,
    output_path: str,
) -> str:
    """Generate an L2 order book depth chart (horizontal bar chart).

    Shows bid levels (green, right side) and ask levels (red, left side)
    at a specific tick.

    Args:
        depth: Dict with 'bids' and 'asks' lists of (price, volume) tuples.
        tick: The simulation tick number.
        output_path: Path to save the PNG image.

    Returns:
        The output_path string.
    """
    set_style()
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])

    if not bids and not asks:
        return output_path

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot asks (negative direction)
    if asks:
        ask_prices = [p for p, _ in asks]
        ask_vols = [-v for _, v in asks]
        ax.barh(ask_prices, ask_vols, color=COLORS["sell"], alpha=0.6, height=0.15, label="Ask")
        for p, v in zip(ask_prices, ask_vols):
            ax.text(v - 0.5, p, f"{abs(v):.0f}", va="center", ha="right", fontsize=8, color=COLORS["fg"])

    # Plot bids (positive direction)
    if bids:
        bid_prices = [p for p, _ in bids]
        bid_vols = [v for _, v in bids]
        ax.barh(bid_prices, bid_vols, color=COLORS["buy"], alpha=0.6, height=0.15, label="Bid")
        for p, v in zip(bid_prices, bid_vols):
            ax.text(v + 0.5, p, f"{v:.0f}", va="center", ha="left", fontsize=8, color=COLORS["fg"])

    ax.axvline(x=0, color=COLORS["fg"], linewidth=0.8)
    ax.set_title(f"Order Book Depth — Tick {tick}", fontsize=13, fontweight="bold", color=COLORS["accent_blue"])
    ax.set_xlabel("Volume (Bid + / Ask -)")
    ax.set_ylabel("Price")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def spread_history_chart(
    spread_history: list[float],
    price_history: list[float],
    output_path: str,
) -> str:
    """Generate a chart showing bid-ask spread evolution alongside price.

    Args:
        spread_history: List of absolute spreads per tick.
        price_history: List of mid prices (same length).
        output_path: Path to save the PNG image.

    Returns:
        The output_path string.
    """
    set_style()
    n = min(len(spread_history), len(price_history))
    if n == 0:
        return output_path

    spreads = spread_history[-n:]
    prices = price_history[-n:]

    fig, ax1 = plt.subplots(figsize=(14, 5))

    x = range(n)
    ax1.fill_between(x, spreads, alpha=0.3, color=COLORS["accent_orange"])
    ax1.plot(x, spreads, color=COLORS["accent_orange"], linewidth=1.5, label="Bid-Ask Spread")
    ax1.set_ylabel("Spread ($)", color=COLORS["accent_orange"])
    ax1.tick_params(axis="y", labelcolor=COLORS["accent_orange"])

    ax2 = ax1.twinx()
    ax2.plot(x, prices, color=COLORS["accent_blue"], linewidth=1.0, alpha=0.6, label="Price")
    ax2.set_ylabel("Price", color=COLORS["accent_blue"])
    ax2.tick_params(axis="y", labelcolor=COLORS["accent_blue"])

    ax1.set_title("Bid-Ask Spread vs Price Over Time", fontsize=14, fontweight="bold", color=COLORS["accent_orange"])
    ax1.set_xlabel("Tick")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def agent_activity_heatmap(
    agents_detail: list[dict],
    output_path: str,
) -> str:
    """Generate a heatmap-style chart showing agent activity patterns.

    Displays trade count, total P&L, and position size per agent.

    Args:
        agents_detail: List of per-agent stats dicts.
        output_path: Path to save the PNG image.

    Returns:
        The output_path string.
    """
    set_style()
    if not agents_detail:
        return output_path

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sort agents by type then trade count
    sorted_agents = sorted(agents_detail, key=lambda a: (a["agent_type"], -a["trade_count"]))
    agent_ids = [a["agent_id"] for a in sorted_agents]
    trade_counts = [a["trade_count"] for a in sorted_agents]
    pnls = [a["total_pnl"] for a in sorted_agents]
    positions = [a["position"] for a in sorted_agents]
    colors = [AGENT_COLORS.get(a["agent_type"], COLORS["fg"]) for a in sorted_agents]

    # Trade count
    axes[0].barh(agent_ids, trade_counts, color=colors, alpha=0.8, edgecolor=COLORS["grid"])
    axes[0].set_title("Trade Count", fontsize=12, fontweight="bold", color=COLORS["accent_blue"])
    axes[0].invert_yaxis()

    # P&L
    pnl_colors = [COLORS["buy"] if p >= 0 else COLORS["sell"] for p in pnls]
    axes[1].barh(agent_ids, pnls, color=pnl_colors, alpha=0.8, edgecolor=COLORS["grid"])
    axes[1].set_title("Total P&L ($)", fontsize=12, fontweight="bold", color=COLORS["accent_green"])
    axes[1].axvline(x=0, color=COLORS["fg"], linewidth=0.5)
    axes[1].invert_yaxis()

    # Position
    pos_colors = [COLORS["buy"] if p >= 0 else COLORS["sell"] for p in positions]
    axes[2].barh(agent_ids, positions, color=pos_colors, alpha=0.8, edgecolor=COLORS["grid"])
    axes[2].set_title("Net Position", fontsize=12, fontweight="bold", color=COLORS["accent_purple"])
    axes[2].axvline(x=0, color=COLORS["fg"], linewidth=0.5)
    axes[2].invert_yaxis()

    fig.suptitle("Agent Activity Overview", fontsize=15, fontweight="bold", color=COLORS["fg"], y=1.02)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def animate_simulation(
    snapshots: list[dict],
    depth_history: list[dict],
    output_gif: str,
    max_frames: int = 120,
) -> str:
    """Generate an animated GIF of the simulation: price chart + order book.

    Args:
        snapshots: List of per-tick market snapshot dicts.
        depth_history: List of periodic depth snapshot dicts.
        output_gif: Path to save the GIF file.
        max_frames: Maximum number of frames (sampled evenly).

    Returns:
        The output_gif path string.
    """
    set_style()

    if not snapshots:
        return output_gif

    # Sample frames evenly
    total = len(snapshots)
    step = max(1, total // max_frames)
    frame_indices = list(range(0, total, step))[:max_frames]

    if not frame_indices:
        return output_gif

    temp_dir = Path(output_gif).parent / "_anim_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Precompute price range
    all_prices = [s["mid_price"] for s in snapshots if s.get("mid_price") is not None]
    price_min = min(all_prices) * 0.98 if all_prices else 90
    price_max = max(all_prices) * 1.02 if all_prices else 110

    depth_lookup = {d["tick"]: d for d in depth_history}

    frame_paths = []
    for i, idx in enumerate(frame_indices):
        fig = plt.figure(figsize=(12, 6))

        # Price history up to this tick
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

        # Left: Price chart
        ax1 = fig.add_subplot(gs[0])
        prices_so_far = [s["mid_price"] for s in snapshots[:idx + 1] if s.get("mid_price") is not None]
        if prices_so_far:
            x = range(len(prices_so_far))
            ax1.plot(x, prices_so_far, color=COLORS["accent_blue"], linewidth=1.5)
        ax1.set_ylim(price_min, price_max)
        ax1.set_title(f"Price — Tick {snapshots[idx]['tick']}", fontsize=12, fontweight="bold",
                      color=COLORS["accent_blue"])
        ax1.set_ylabel("Price")

        # Right: Order book depth at nearest snapshot
        ax2 = fig.add_subplot(gs[1])
        cur_tick = snapshots[idx]["tick"]
        # Find nearest depth snapshot
        nearest_depth = None
        for dt in depth_history:
            if dt["tick"] <= cur_tick:
                nearest_depth = dt
            else:
                break

        if nearest_depth:
            asks = nearest_depth.get("asks", [])
            bids = nearest_depth.get("bids", [])
            if asks:
                ap = [p for p, _ in asks]
                av = [-v for _, v in asks]
                ax2.barh(ap, av, color=COLORS["sell"], alpha=0.5, height=0.15)
            if bids:
                bp = [p for p, _ in bids]
                bv = [v for _, v in bids]
                ax2.barh(bp, bv, color=COLORS["buy"], alpha=0.5, height=0.15)
            ax2.axvline(x=0, color=COLORS["fg"], linewidth=0.5)
        ax2.set_title("Order Book", fontsize=12, fontweight="bold", color=COLORS["accent_orange"])

        frame_path = str(temp_dir / f"frame_{i:04d}.png")
        plt.tight_layout()
        fig.savefig(frame_path)
        plt.close(fig)
        frame_paths.append(frame_path)

    # Combine frames into GIF using PIL if available
    gif_created = False
    try:
        from PIL import Image as PILImage
        images = [PILImage.open(fp) for fp in frame_paths]
        if images:
            images[0].save(
                output_gif,
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0,
                optimize=True,
            )
            gif_created = True
    except ImportError:
        pass

    # Cleanup temp frames
    for fp in frame_paths:
        try:
            os.remove(fp)
        except OSError:
            pass
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    if not gif_created:
        print(f"  [WARN] PIL not available; frames saved individually but GIF not created.")
        print(f"  Install Pillow: pip install Pillow")

    return output_gif


def generate_report(
    result,  # SimulationResult
    output_path: str,
    chart_dir: str = "output/charts",
) -> str:
    """Generate a comprehensive HTML report from simulation results.

    Creates an HTML file with:
    - Price history chart
    - Volume profile chart
    - Agent P&L comparison
    - Order book snapshots
    - Key metrics table
    - Agent stats table

    Args:
        result: SimulationResult from a completed run.
        output_path: Path to save the HTML report.
        chart_dir: Directory to save chart PNGs (relative to report).

    Returns:
        The output_path string.
    """
    set_style()
    chart_path = Path(chart_dir)
    chart_path.mkdir(parents=True, exist_ok=True)

    # Generate all charts
    price_volume_chart(
        result.price_history,
        result.volume_profile,
        str(chart_path / "price_volume.png"),
    )

    agent_pnl_chart(
        result.agents_detail,
        result.price_history,
        str(chart_path / "agent_pnl.png"),
    )

    spread_history_chart(
        result.spread_history,
        result.price_history,
        str(chart_path / "spread_history.png"),
    )

    agent_activity_heatmap(
        result.agents_detail,
        str(chart_path / "agent_activity.png"),
    )

    # Order book snapshot at midpoint
    if result.orderbook_depth_history:
        mid_depth = result.orderbook_depth_history[len(result.orderbook_depth_history) // 2]
        order_book_snapshot(mid_depth, mid_depth["tick"], str(chart_path / "orderbook_depth.png"))

    # Build HTML report
    if not _HAS_JINJA2:
        _generate_simple_html(result, output_path)
        return output_path

    template = _get_html_template()
    html = template.render(
        config=result.config,
        timestamp=result.timestamp,
        duration=f"{result.duration_seconds:.1f}",
        total_ticks=result.total_ticks,
        total_trades=result.total_trades,
        total_volume=f"{result.total_volume:,.0f}",
        final_price=f"{result.final_price:.2f}",
        agent_stats=result.agent_stats,
        agents_detail=result.agents_detail,
        snapshots=result.snapshots,
        AGENT_LABELS_CN=AGENT_LABELS_CN,
        AGENT_COLORS=AGENT_COLORS,
        colors=COLORS,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Report saved to {output_path}")
    return output_path


def _get_html_template() -> "Template":
    """Return the Jinja2 HTML template for the simulation report."""
    return Template("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Trade Sim — Simulation Report</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        background: {{ colors.bg }};
        color: {{ colors.fg }};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        line-height: 1.6;
        padding: 40px 60px;
    }
    h1 { color: {{ colors.accent_blue }}; font-size: 32px; margin-bottom: 5px; }
    h2 {
        color: {{ colors.accent_blue }};
        font-size: 22px;
        margin: 40px 0 20px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid {{ colors.grid }};
    }
    .subtitle { color: {{ colors.fg }}; opacity: 0.6; font-size: 14px; margin-bottom: 30px; }
    .metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: {{ colors.panel_bg }};
        border: 1px solid {{ colors.grid }};
        border-radius: 8px;
        padding: 20px;
    }
    .metric-card .label { font-size: 12px; text-transform: uppercase; opacity: 0.6; margin-bottom: 5px; }
    .metric-card .value { font-size: 28px; font-weight: bold; }
    .metric-card .value.green { color: {{ colors.accent_green }}; }
    .metric-card .value.blue { color: {{ colors.accent_blue }}; }
    .metric-card .value.orange { color: {{ colors.accent_orange }}; }
    .metric-card .value.red { color: {{ colors.accent_red }}; }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 14px;
    }
    th {
        background: {{ colors.panel_bg }};
        color: {{ colors.fg }};
        padding: 12px 15px;
        text-align: left;
        border-bottom: 2px solid {{ colors.grid }};
        font-weight: 600;
    }
    td {
        padding: 10px 15px;
        border-bottom: 1px solid {{ colors.grid }};
    }
    tr:hover td { background: {{ colors.panel_bg }}; }
    .chart-container { margin: 30px 0; text-align: center; }
    .chart-container img { max-width: 100%; border-radius: 6px; border: 1px solid {{ colors.grid }}; }
    .pnl-positive { color: {{ colors.accent_green }}; }
    .pnl-negative { color: {{ colors.accent_red }}; }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-value { background: rgba(63,185,80,0.15); color: {{ colors.accent_green }}; }
    .badge-momentum { background: rgba(88,166,255,0.15); color: {{ colors.accent_blue }}; }
    .badge-noise { background: rgba(139,148,158,0.15); color: {{ colors.fg }}; }
    .badge-mm { background: rgba(210,153,29,0.15); color: {{ colors.accent_orange }}; }
    .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid {{ colors.grid }}; opacity: 0.5; font-size: 12px; }
</style>
</head>
<body>

<h1>Agent Trade Sim — Simulation Report</h1>
<p class="subtitle">Multi-Agent Limit Order Book Market Simulation | {{ timestamp }}</p>

<h2>Key Metrics</h2>
<div class="metrics">
    <div class="metric-card">
        <div class="label">Total Ticks</div>
        <div class="value blue">{{ total_ticks }}</div>
    </div>
    <div class="metric-card">
        <div class="label">Total Trades</div>
        <div class="value blue">{{ total_trades }}</div>
    </div>
    <div class="metric-card">
        <div class="label">Total Volume</div>
        <div class="value orange">{{ total_volume }}</div>
    </div>
    <div class="metric-card">
        <div class="label">Final Price</div>
        <div class="value green">${{ final_price }}</div>
    </div>
    <div class="metric-card">
        <div class="label">Duration</div>
        <div class="value">{{ duration }}s</div>
    </div>
    <div class="metric-card">
        <div class="label">Agents</div>
        <div class="value">{{ agents_detail|length }}</div>
    </div>
</div>

<h2>Configuration</h2>
<table>
    <tr><th>Parameter</th><th>Value</th></tr>
    {% for key, val in config.items() %}
    <tr><td>{{ key }}</td><td>{{ val }}</td></tr>
    {% endfor %}
</table>

<h2>Price & Volume</h2>
<div class="chart-container"><img src="charts/price_volume.png" alt="Price and Volume"></div>

<h2>Bid-Ask Spread</h2>
<div class="chart-container"><img src="charts/spread_history.png" alt="Spread History"></div>

{% if orderbook_depth_history %}
<h2>Order Book Depth</h2>
<div class="chart-container"><img src="charts/orderbook_depth.png" alt="Order Book Depth"></div>
{% endif %}

<h2>Agent Performance (by Type)</h2>
<table>
    <tr>
        <th>Type</th>
        <th>Agents</th>
        <th>Total P&L</th>
        <th>Avg P&L</th>
        <th>Total Trades</th>
        <th>Win Rate</th>
        <th>Net Position</th>
    </tr>
    {% for s in agent_stats %}
    <tr>
        <td>
            <span class="badge badge-{{ s.agent_type.replace('_', '-') }}">
            {{ AGENT_LABELS_CN.get(s.agent_type, s.agent_type) }}
            </span>
        </td>
        <td>{{ s.count }}</td>
        <td class="{% if s.total_pnl >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
            ${{ "%.2f"|format(s.total_pnl) }}
        </td>
        <td class="{% if s.avg_pnl >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
            ${{ "%.2f"|format(s.avg_pnl) }}
        </td>
        <td>{{ s.total_trades }}</td>
        <td>{{ "%.1f"|format(s.win_rate * 100) }}%</td>
        <td class="{% if s.net_position >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
            {{ "%.0f"|format(s.net_position) }}
        </td>
    </tr>
    {% endfor %}
</table>

<div class="chart-container"><img src="charts/agent_pnl.png" alt="Agent P&L"></div>

<h2>Agent Activity</h2>
<div class="chart-container"><img src="charts/agent_activity.png" alt="Agent Activity"></div>

<h2>Individual Agent Details</h2>
<table>
    <tr>
        <th>Agent ID</th>
        <th>Type</th>
        <th>Capital</th>
        <th>Position</th>
        <th>Realized P&L</th>
        <th>Unrealized P&L</th>
        <th>Total P&L</th>
        <th>Equity</th>
        <th>Trades</th>
    </tr>
    {% for a in agents_detail %}
    <tr>
        <td>{{ a.agent_id }}</td>
        <td>{{ AGENT_LABELS_CN.get(a.agent_type, a.agent_type) }}</td>
        <td>${{ "%.0f"|format(a.capital) }}</td>
        <td>{{ "%.0f"|format(a.position) }}</td>
        <td class="{% if a.realized_pnl >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
            ${{ "%.2f"|format(a.realized_pnl) }}
        </td>
        <td class="{% if a.unrealized_pnl >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
            ${{ "%.2f"|format(a.unrealized_pnl) }}
        </td>
        <td class="{% if a.total_pnl >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
            ${{ "%.2f"|format(a.total_pnl) }}
        </td>
        <td>${{ "%.2f"|format(a.equity) }}</td>
        <td>{{ a.trade_count }}</td>
    </tr>
    {% endfor %}
</table>

<div class="footer">
    Generated by Agent Trade Sim v1.0.0 &mdash; Multi-Agent Limit Order Book Market Simulation
</div>

</body>
</html>
""")


def _generate_simple_html(result, output_path: str) -> None:
    """Fallback: generate a simple HTML report without Jinja2 templating."""
    stats_rows = ""
    for s in result.agent_stats:
        pnl_class = "pnl-positive" if s.get("total_pnl", 0) >= 0 else "pnl-negative"
        avg_class = "pnl-positive" if s.get("avg_pnl", 0) >= 0 else "pnl-negative"
        stats_rows += f"""
        <tr>
            <td>{AGENT_LABELS_CN.get(s['agent_type'], s['agent_type'])}</td>
            <td>{s['count']}</td>
            <td class="{pnl_class}">${s['total_pnl']:.2f}</td>
            <td class="{avg_class}">${s['avg_pnl']:.2f}</td>
            <td>{s['total_trades']}</td>
            <td>{s['win_rate']*100:.1f}%</td>
            <td>{s['net_position']:.0f}</td>
        </tr>"""

    agent_rows = ""
    for a in result.agents_detail:
        pnl_class = "pnl-positive" if a.get("total_pnl", 0) >= 0 else "pnl-negative"
        agent_rows += f"""
        <tr>
            <td>{a['agent_id']}</td>
            <td>{AGENT_LABELS_CN.get(a['agent_type'], a['agent_type'])}</td>
            <td>${a['capital']:.0f}</td>
            <td>{a['position']:.0f}</td>
            <td class="{pnl_class}">${a['total_pnl']:.2f}</td>
            <td>${a['equity']:.2f}</td>
            <td>{a['trade_count']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Agent Trade Sim — Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        background: #0d1117; color: #c9d1d9;
        font-family: sans-serif; padding: 40px 60px;
    }}
    h1 {{ color: #58a6ff; font-size: 28px; }}
    h2 {{ color: #58a6ff; margin: 30px 0 15px; border-bottom: 2px solid #21262d; padding-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }}
    th {{ background: #161b22; padding: 12px; text-align: left; border-bottom: 2px solid #21262d; }}
    td {{ padding: 10px; border-bottom: 1px solid #21262d; }}
    .pnl-positive {{ color: #3fb950; }}
    .pnl-negative {{ color: #f85149; }}
    .chart-container {{ margin: 25px 0; }}
    .chart-container img {{ max-width: 100%; border-radius: 6px; }}
    .footer {{ margin-top: 40px; opacity: 0.5; font-size: 12px; }}
    .metrics {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .metric-card {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 20px; min-width: 180px; }}
    .metric-card .label {{ font-size: 12px; opacity: 0.6; }}
    .metric-card .value {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
</style>
</head>
<body>

<h1>Agent Trade Sim — Simulation Report</h1>
<p style="opacity:0.6;font-size:14px;">{result.timestamp} | Duration: {result.duration_seconds:.1f}s</p>

<h2>Key Metrics</h2>
<div class="metrics">
    <div class="metric-card"><div class="label">Total Ticks</div><div class="value">{result.total_ticks}</div></div>
    <div class="metric-card"><div class="label">Total Trades</div><div class="value">{result.total_trades}</div></div>
    <div class="metric-card"><div class="label">Total Volume</div><div class="value">{result.total_volume:,.0f}</div></div>
    <div class="metric-card"><div class="label">Final Price</div><div class="value" style="color:#3fb950">${result.final_price:.2f}</div></div>
</div>

<h2>Price & Volume</h2>
<div class="chart-container"><img src="charts/price_volume.png" alt="Price and Volume"></div>

<h2>Agent Performance (by Type)</h2>
<table>
    <tr><th>Type</th><th>Agents</th><th>Total P&L</th><th>Avg P&L</th><th>Trades</th><th>Win Rate</th><th>Net Position</th></tr>
    {stats_rows}
</table>

<div class="chart-container"><img src="charts/agent_pnl.png" alt="Agent P&L"></div>
<div class="chart-container"><img src="charts/agent_activity.png" alt="Agent Activity"></div>

<h2>Individual Agents</h2>
<table>
    <tr><th>Agent ID</th><th>Type</th><th>Capital</th><th>Position</th><th>Total P&L</th><th>Equity</th><th>Trades</th></tr>
    {agent_rows}
</table>

<div class="footer">Generated by Agent Trade Sim v1.0.0</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Report saved to {output_path}")
