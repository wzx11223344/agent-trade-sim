# Agent Trade Sim

**A multi-agent market laboratory — not another RL trading bot repo. This simulates emergent price dynamics from heterogeneous agent interactions in a real limit order book.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What Makes This Different?

Most "trading agent" repos are wrappers around RL libraries that train a single agent. **Agent Trade Sim** is a market microstructure simulation: four distinct agent types interact through a realistic limit order book, and **prices emerge from their collective behavior** -- not from a pre-programmed random walk.

You can watch in real-time as:
- Value investors push prices toward fair value
- Momentum traders create trends and reversals
- Noise traders supply the liquidity that makes markets possible
- Market makers earn the spread while managing inventory risk

---

## Architecture

```
agent-trade-sim/
├── run.py                    # CLI entry point
├── agent_trade_sim/
│   ├── orderbook.py          # Limit order book matching engine
│   ├── agents.py             # 4 agent types: Value, Momentum, Noise, MarketMaker
│   ├── market.py             # Simulation coordinator
│   ├── strategies.py         # Strategy parameterization
│   └── viz.py                # Visualization + HTML reports
├── config/
│   └── market_params.yaml    # Configurable parameters
├── examples/
│   └── demo.py               # Full workflow demo
└── output/                   # Simulation results
```

---

## The Four Agent Types

| Agent | Archetype | Strategy | Market Role |
|-------|-----------|----------|-------------|
| **ValueAgent** | Fundamental investor | Estimates noisy fair value; buys below, sells above | Mean-reversion force |
| **MomentumAgent** | Trend follower | Dual MA crossover; rides trends, stops out on reversals | Trend amplification |
| **NoiseAgent** | Uninformed trader | Random buy/sell with configurable probability | Liquidity provision |
| **MarketMakerAgent** | Liquidity provider | Two-sided quotes; spread adjusted for volatility and inventory | Price efficiency |

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run a default simulation
python run.py simulate

# Custom agent population
python run.py simulate --value 5 --momentum 8 --noise 20 --mm 3 --ticks 10000

# Generate report from saved run
python run.py report --data output/simulation.json

# Generate animation
python run.py animate --data output/simulation.json

# Benchmark: 10 runs with statistical summary
python run.py benchmark --runs 10
```

Or run the full demo:

```bash
python examples/demo.py
```

---

## What You'll See

After running a simulation, check `output/` for:

- **simulation.json** — Full run data (prices, volumes, trades, agent P&L)
- **report.html** — Interactive HTML report with charts and tables
- **animation.gif** — Animated price chart with live order book
- **charts/** — Individual PNG charts (price, P&L, spread, depth, activity)

---

## How Prices Emerge

The simulation is **not** a random walk. Price dynamics emerge from the interaction of:

1. **Fundamental anchoring**: Value agents pull prices toward their (noisy) fair value estimate
2. **Trend amplification**: Momentum agents push prices further in trending directions
3. **Liquidity friction**: Noise agents provide random order flow that can absorb or amplify moves
4. **Spread compression**: Market makers narrow spreads when confident, widen them when uncertain

The result: realistic price series with trends, reversals, volatility clustering, and spread dynamics -- all from agent interaction rules, not stochastic processes.

---

## Order Book Features

- **Price-time priority** matching with partial fills
- **L2 depth** queries (top N levels)
- **Order imbalance** metric (buy pressure / total pressure)
- **Rolling realized volatility**
- **Trade history** with timestamps, prices, and aggressor side
- **Market orders** that cross the spread

---

## Configuration

All parameters are in `config/market_params.yaml`:

```yaml
market:
  initial_price: 100.0
  tick_size: 0.01
  n_ticks: 5000

agents:
  n_value: 3
  n_momentum: 5
  n_noise: 15
  n_market_maker: 2

value_strategy:
  noise_std: 0.02
  confidence_threshold: 0.005
  max_position: 100
  # ... more parameters
```

Override any parameter via command-line:

```bash
python run.py simulate --price 200.0 --tick-size 0.05 --noise 30
```

---

## Educational Use Cases

- **Market microstructure**: See how order books work -- bids, asks, spreads, depth, matching
- **Agent-based modeling**: Understand emergent phenomena from simple agent rules
- **Trading strategy analysis**: Compare value vs momentum vs market making strategies
- **Risk management**: Observe how inventory risk affects market maker behavior
- **Price discovery**: Watch prices converge to (or diverge from) fundamental value

---

## Requirements

- Python 3.9+
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- pyyaml >= 6.0
- jinja2 >= 3.0
- Pillow (optional, for GIF animation)

---

## License

MIT — Use it, learn from it, build on it.
