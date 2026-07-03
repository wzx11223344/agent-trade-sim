"""
run.py — CLI Entry Point for Agent Trade Sim
==============================================
Usage:
    python run.py simulate                      # Default simulation
    python run.py simulate --ticks 10000        # Longer run
    python run.py simulate --value 5 --momentum 8 --noise 20 --mm 3
    python run.py report --data simulation.json # Generate report
    python run.py animate --data simulation.json # Generate animation
    python run.py benchmark                      # Run 10 and compare
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_trade_sim.market import SimulatedMarket, SimulationConfig, SimulationResult
from agent_trade_sim.strategies import (
    ValueStrategyParams,
    MomentumStrategyParams,
    NoiseStrategyParams,
    MarketMakerParams,
)
from agent_trade_sim.viz import (
    animate_simulation,
    generate_report,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent Trade Sim — Multi-Agent Limit Order Book Market Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py simulate
  python run.py simulate --ticks 10000 --seed 123
  python run.py simulate --value 5 --momentum 8 --noise 20 --mm 3
  python run.py report --data output/simulation.json
  python run.py animate --data output/simulation.json
  python run.py benchmark
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- simulate ---
    sim_parser = subparsers.add_parser("simulate", help="Run a market simulation")
    sim_parser.add_argument("--value", type=int, default=3, help="Number of ValueAgents")
    sim_parser.add_argument("--momentum", type=int, default=5, help="Number of MomentumAgents")
    sim_parser.add_argument("--noise", type=int, default=15, help="Number of NoiseAgents")
    sim_parser.add_argument("--mm", type=int, default=2, help="Number of MarketMakerAgents")
    sim_parser.add_argument("--ticks", type=int, default=5000, help="Simulation ticks")
    sim_parser.add_argument("--price", type=float, default=100.0, help="Initial price")
    sim_parser.add_argument("--tick-size", type=float, default=0.01, help="Minimum price increment")
    sim_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    sim_parser.add_argument("--config", type=str, default=None, help="YAML config file to override defaults")
    sim_parser.add_argument("--output", type=str, default="output/simulation.json", help="Output JSON path")
    sim_parser.add_argument("--report", action="store_true", default=True, help="Generate HTML report")
    sim_parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    sim_parser.add_argument("--animate", action="store_true", default=False, help="Generate animation GIF")
    sim_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # --- report ---
    report_parser = subparsers.add_parser("report", help="Generate HTML report from saved simulation")
    report_parser.add_argument("--data", type=str, required=True, help="Path to simulation JSON")
    report_parser.add_argument("--output", type=str, default="output/report.html", help="Output HTML path")
    report_parser.add_argument("--chart-dir", type=str, default="output/charts", help="Chart image directory")

    # --- animate ---
    anim_parser = subparsers.add_parser("animate", help="Generate animation GIF from saved simulation")
    anim_parser.add_argument("--data", type=str, required=True, help="Path to simulation JSON")
    anim_parser.add_argument("--output", type=str, default="output/animation.gif", help="Output GIF path")
    anim_parser.add_argument("--frames", type=int, default=120, help="Maximum frames")

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Run multiple simulations and compare")
    bench_parser.add_argument("--runs", type=int, default=10, help="Number of simulation runs")
    bench_parser.add_argument("--ticks", type=int, default=3000, help="Ticks per run")
    bench_parser.add_argument("--output", type=str, default="output/benchmark.json", help="Output path")

    return parser.parse_args()


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run a simulation and generate output."""
    # Load config from YAML if provided
    value_params = ValueStrategyParams()
    momentum_params = MomentumStrategyParams()
    noise_params = NoiseStrategyParams()
    mm_params = MarketMakerParams()

    if args.config:
        _load_config_from_yaml(args.config)

    config = SimulationConfig(
        n_value=args.value,
        n_momentum=args.momentum,
        n_noise=args.noise,
        n_mm=args.mm,
        n_ticks=args.ticks,
        initial_price=args.price,
        tick_size=args.tick_size,
        seed=args.seed,
        value_params=value_params,
        momentum_params=momentum_params,
        noise_params=noise_params,
        mm_params=mm_params,
    )

    market = SimulatedMarket(config)
    result = market.run(verbose=not args.quiet)

    # Save result
    SimulatedMarket.save_result(result, args.output)

    # Generate report
    if not args.no_report:
        report_path = args.output.replace(".json", ".html").replace("simulation", "report")
        chart_dir = str(Path(report_path).parent / "charts")
        generate_report(result, report_path, chart_dir)

    # Generate animation
    if args.animate:
        anim_path = args.output.replace(".json", ".gif").replace("simulation", "animation")
        animate_simulation(
            result.snapshots,
            result.orderbook_depth_history,
            anim_path,
            args.frames if hasattr(args, 'frames') else 120,
        )
        print(f"  Animation saved to {anim_path}")


def _load_config_from_yaml(config_path: str) -> None:
    """Load and print config from YAML file. Parameter overrides are handled
    via command-line args, but this validates the config file exists."""
    if not os.path.exists(config_path):
        print(f"  [WARN] Config file not found: {config_path}")
        return
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"  Loaded config from {config_path}")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate report from saved simulation data."""
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    result = SimulatedMarket.load_result(args.data)
    generate_report(result, args.output, args.chart_dir)


def cmd_animate(args: argparse.Namespace) -> None:
    """Generate animation from saved simulation data."""
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    result = SimulatedMarket.load_result(args.data)
    animate_simulation(
        result.snapshots,
        result.orderbook_depth_history,
        args.output,
        args.frames,
    )
    print(f"  Animation saved to {args.output}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run multiple simulations and compare results."""
    print(f"\n{'='*60}")
    print(f"  Benchmark — {args.runs} runs x {args.ticks} ticks each")
    print(f"{'='*60}\n")

    results = []
    for run_idx in range(args.runs):
        seed = 42 + run_idx * 100
        config = SimulationConfig(
            n_value=3,
            n_momentum=5,
            n_noise=15,
            n_mm=2,
            n_ticks=args.ticks,
            initial_price=100.0,
            seed=seed,
        )
        market = SimulatedMarket(config)
        result = market.run(verbose=False)

        mid = result.final_price
        vol = result.total_volume
        trades = result.total_trades
        total_pnl = sum(s["total_pnl"] for s in result.agent_stats)

        results.append({
            "run": run_idx + 1,
            "seed": seed,
            "final_price": mid,
            "total_volume": vol,
            "total_trades": trades,
            "duration_s": result.duration_seconds,
            "total_agent_pnl": total_pnl,
            "agent_stats": result.agent_stats,
        })
        print(f"  Run {run_idx + 1:2d} | Price: {mid:8.2f} | Vol: {vol:8.0f} | "
              f"Trades: {trades:5d} | Time: {result.duration_seconds:5.1f}s")

    # Summary statistics
    prices = [r["final_price"] for r in results]
    vols = [r["total_volume"] for r in results]
    trades_list = [r["total_trades"] for r in results]
    times = [r["duration_s"] for r in results]

    import numpy as np
    print(f"\n{'='*60}")
    print(f"  Benchmark Summary")
    print(f"{'='*60}")
    print(f"  Final Price:  ${np.mean(prices):.2f} +/- {np.std(prices):.2f}")
    print(f"  Total Volume: {np.mean(vols):.0f} +/- {np.std(vols):.0f}")
    print(f"  Total Trades: {np.mean(trades_list):.0f} +/- {np.std(trades_list):.0f}")
    print(f"  Avg Runtime:  {np.mean(times):.1f}s\n")

    # Save benchmark data
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "runs": args.runs,
                "ticks_per_run": args.ticks,
            },
            "summary": {
                "avg_price": float(np.mean(prices)),
                "std_price": float(np.std(prices)),
                "avg_volume": float(np.mean(vols)),
                "avg_trades": float(np.mean(trades_list)),
                "avg_duration": float(np.mean(times)),
            },
            "runs": results,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"  Benchmark saved to {args.output}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.command is None:
        # Default: run simulate
        sys.argv = ["run.py", "simulate"]
        args = parse_args()

    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "animate":
        cmd_animate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
