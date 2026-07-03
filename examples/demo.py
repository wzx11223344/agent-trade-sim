"""
demo.py — Example Usage of Agent Trade Sim
============================================
Demonstrates the full workflow:
  1. Create a SimulatedMarket with custom configuration
  2. Run a simulation and collect results
  3. Generate charts and an HTML report
  4. Generate an animated GIF (if PIL is available)
"""

import os
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trade_sim.market import SimulatedMarket, SimulationConfig
from agent_trade_sim.strategies import (
    ValueStrategyParams,
    MomentumStrategyParams,
    NoiseStrategyParams,
    MarketMakerParams,
)
from agent_trade_sim.viz import (
    animate_simulation,
    generate_report,
    price_volume_chart,
    agent_pnl_chart,
    spread_history_chart,
    agent_activity_heatmap,
    order_book_snapshot,
)


def main():
    """Run a demo simulation and generate all output artifacts."""

    # --- 1. Configure the simulation ---
    print("=" * 60)
    print("  Agent Trade Sim — Demo")
    print("=" * 60)
    print()

    # Custom strategy parameters
    value_params = ValueStrategyParams(
        lookback=50,
        noise_std=0.02,
        confidence_threshold=0.005,
        max_position=100,
        order_size=10.0,
    )

    momentum_params = MomentumStrategyParams(
        lookback=20,
        signal_lookback_short=5,
        signal_lookback_long=20,
        trend_threshold=0.003,
        max_position=80,
        order_size=15.0,
        holding_period=40,
    )

    noise_params = NoiseStrategyParams(
        buy_probability=0.5,
        trade_probability=0.25,
        mean_order_size=5.0,
    )

    mm_params = MarketMakerParams(
        base_spread=0.002,
        volatility_multiplier=2.0,
        max_position=200,
        quote_size=25.0,
        requote_interval=3,
    )

    config = SimulationConfig(
        n_value=3,
        n_momentum=5,
        n_noise=15,
        n_mm=2,
        n_ticks=2000,  # Shorter for demo purposes
        initial_price=100.0,
        tick_size=0.01,
        seed=42,
        value_params=value_params,
        momentum_params=momentum_params,
        noise_params=noise_params,
        mm_params=mm_params,
    )

    # --- 2. Run simulation ---
    market = SimulatedMarket(config)
    result = market.run(verbose=True)

    # --- 3. Save simulation data ---
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = str(output_dir / "demo_simulation.json")
    SimulatedMarket.save_result(result, json_path)

    # --- 4. Generate charts ---
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating Charts ---")

    price_volume_chart(
        result.price_history,
        result.volume_profile,
        str(chart_dir / "price_volume.png"),
    )
    print("  [OK] Price + Volume chart")

    agent_pnl_chart(
        result.agents_detail,
        result.price_history,
        str(chart_dir / "agent_pnl.png"),
    )
    print("  [OK] Agent P&L chart")

    spread_history_chart(
        result.spread_history,
        result.price_history,
        str(chart_dir / "spread_history.png"),
    )
    print("  [OK] Spread history chart")

    agent_activity_heatmap(
        result.agents_detail,
        str(chart_dir / "agent_activity.png"),
    )
    print("  [OK] Agent activity chart")

    if result.orderbook_depth_history:
        mid_depth = result.orderbook_depth_history[len(result.orderbook_depth_history) // 2]
        order_book_snapshot(mid_depth, mid_depth["tick"], str(chart_dir / "orderbook_depth.png"))
        print("  [OK] Order book depth chart")

    # --- 5. Generate HTML report ---
    report_path = str(output_dir / "demo_report.html")
    generate_report(result, report_path, str(chart_dir))
    print(f"\n  [OK] Report: {report_path}")

    # --- 6. Generate animation (if PIL available) ---
    print("\n--- Generating Animation ---")
    anim_path = str(output_dir / "demo_animation.gif")
    animate_simulation(
        result.snapshots,
        result.orderbook_depth_history,
        anim_path,
        max_frames=100,
    )
    if os.path.exists(anim_path):
        print(f"  [OK] Animation: {anim_path}")
    else:
        print(f"  [SKIP] Animation not generated (install Pillow: pip install Pillow)")

    # --- 7. Print summary ---
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  - {json_path}")
    print(f"  - {report_path}")
    if os.path.exists(anim_path):
        print(f"  - {anim_path}")
    print()

    # Agent stats summary
    print("  Agent Performance Summary:")
    for s in result.agent_stats:
        label = {
            "value": "Value Investors",
            "momentum": "Momentum Traders",
            "noise": "Noise Traders",
            "market_maker": "Market Makers",
        }.get(s["agent_type"], s["agent_type"])
        pnl_str = f"${s['total_pnl']:+.1f}"
        print(f"    {label:20s} | P&L: {pnl_str:>12s} | Win Rate: {s['win_rate']*100:5.1f}%  | Trades: {s['total_trades']}")


if __name__ == "__main__":
    main()
