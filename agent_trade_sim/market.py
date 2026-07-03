"""
market.py — Market Coordinator and Simulation Loop
====================================================
The SimulatedMarket class orchestrates the multi-agent simulation. It creates
agents, runs the tick-by-tick simulation loop, manages state, and collects
results. This is the core simulation engine that ties together the order book
and agents.
"""

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from agent_trade_sim.agents import (
    BaseAgent,
    ValueAgent,
    MomentumAgent,
    NoiseAgent,
    MarketMakerAgent,
)
from agent_trade_sim.orderbook import LimitOrderBook, Trade
from agent_trade_sim.strategies import (
    ValueStrategyParams,
    MomentumStrategyParams,
    NoiseStrategyParams,
    MarketMakerParams,
)


@dataclass
class AgentStats:
    """Aggregate statistics for a group of agents (by type or individually).

    Attributes:
        agent_type: The type label ('value', 'momentum', 'noise', 'market_maker').
        count: Number of agents of this type.
        total_pnl: Sum of realized P&L across agents.
        total_unrealized: Sum of unrealized P&L across agents.
        total_equity: Sum of capital + unrealized across agents.
        total_trades: Total number of trades executed.
        avg_pnl: Average realized P&L per agent.
        win_rate: Fraction of agents with positive total P&L.
        net_position: Sum of positions (positive = net long).
    """

    agent_type: str
    count: int
    total_pnl: float
    total_unrealized: float
    total_equity: float
    total_trades: int
    avg_pnl: float
    win_rate: float
    net_position: float


@dataclass
class SimulationConfig:
    """Configuration for a simulation run.

    Attributes:
        n_value: Number of ValueAgents.
        n_momentum: Number of MomentumAgents.
        n_noise: Number of NoiseAgents.
        n_mm: Number of MarketMakerAgents.
        n_ticks: Total simulation ticks.
        initial_price: Starting reference price.
        tick_size: Minimum price increment.
        symbol: Trading instrument symbol.
        seed: Random seed for reproducibility.
        value_params: Parameters for ValueAgents.
        momentum_params: Parameters for MomentumAgents.
        noise_params: Parameters for NoiseAgents.
        mm_params: Parameters for MarketMakerAgents.
    """

    n_value: int = 3
    n_momentum: int = 5
    n_noise: int = 15
    n_mm: int = 2
    n_ticks: int = 5000
    initial_price: float = 100.0
    tick_size: float = 0.01
    symbol: str = "SIM"
    seed: int = 42

    value_params: Optional[ValueStrategyParams] = None
    momentum_params: Optional[MomentumStrategyParams] = None
    noise_params: Optional[NoiseStrategyParams] = None
    mm_params: Optional[MarketMakerParams] = None

    def __post_init__(self):
        if self.value_params is None:
            self.value_params = ValueStrategyParams()
        if self.momentum_params is None:
            self.momentum_params = MomentumStrategyParams()
        if self.noise_params is None:
            self.noise_params = NoiseStrategyParams()
        if self.mm_params is None:
            self.mm_params = MarketMakerParams()


@dataclass
class SimulationResult:
    """Complete results from a simulation run.

    Attributes:
        config: The configuration used for this run.
        timestamp: ISO format start time.
        duration_seconds: Wall-clock duration of the simulation.
        total_ticks: Actual ticks completed.
        total_trades: Total number of trades executed.
        total_volume: Total quantity traded.
        final_price: Last transaction price.
        price_history: List of mid prices per tick.
        spread_history: List of spreads per tick.
        volume_profile: List of volume per tick.
        snapshots: List of market state snapshots.
        agent_stats: Per-type aggregate statistics.
        agents_detail: Per-agent detailed stats.
        orderbook_depth_history: Periodic depth snapshots.
    """

    config: dict
    timestamp: str
    duration_seconds: float
    total_ticks: int
    total_trades: int
    total_volume: float
    final_price: float
    price_history: list[float]
    spread_history: list[float]
    volume_profile: list[float]
    snapshots: list[dict]
    agent_stats: list[dict]
    agents_detail: list[dict]
    orderbook_depth_history: list[dict]


class SimulatedMarket:
    """Orchestrates a multi-agent trading simulation.

    Creates agents, runs the tick-by-tick loop, manages the order book,
    tracks state, and produces results.

    Typical usage:
        market = SimulatedMarket(config)
        result = market.run()
        market.save_result(result, "simulation.json")
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._rng = np.random.RandomState(config.seed)
        self._agents: list[BaseAgent] = []
        self._orderbook: Optional[LimitOrderBook] = None
        self._price_history: list[float] = []
        self._spread_history: list[float] = []

        # Per-agent trade tracking
        self._agent_fills: dict[str, list[Trade]] = {}

        self._create_agents()
        self._create_orderbook()

    def _create_agents(self) -> None:
        """Instantiate all agents with their strategy parameters."""
        cfg = self.config
        agent_idx = 0

        for i in range(cfg.n_value):
            agent = ValueAgent(
                agent_id=f"value_{i}",
                initial_capital=100000.0,
                params=cfg.value_params,
                initial_price=cfg.initial_price,
            )
            agent.seed(cfg.seed + agent_idx)
            self._agents.append(agent)
            agent_idx += 1

        for i in range(cfg.n_momentum):
            agent = MomentumAgent(
                agent_id=f"momentum_{i}",
                initial_capital=100000.0,
                params=cfg.momentum_params,
            )
            agent.seed(cfg.seed + agent_idx)
            self._agents.append(agent)
            agent_idx += 1

        for i in range(cfg.n_noise):
            agent = NoiseAgent(
                agent_id=f"noise_{i}",
                initial_capital=100000.0,
                params=cfg.noise_params,
            )
            agent.seed(cfg.seed + agent_idx)
            self._agents.append(agent)
            agent_idx += 1

        for i in range(cfg.n_mm):
            agent = MarketMakerAgent(
                agent_id=f"mm_{i}",
                initial_capital=500000.0,
                params=cfg.mm_params,
                initial_price=cfg.initial_price,
            )
            agent.seed(cfg.seed + agent_idx)
            self._agents.append(agent)
            agent_idx += 1

    def _create_orderbook(self) -> None:
        """Initialize the limit order book."""
        self._orderbook = LimitOrderBook(
            symbol=self.config.symbol,
            tick_size=self.config.tick_size,
            initial_price=self.config.initial_price,
        )

    def get_state(self) -> dict:
        """Return the current full market state for agent observation."""
        if self._orderbook is None:
            return {}

        snapshots = self._orderbook.get_snapshots()
        recent_prices = [s.mid_price for s in snapshots[-100:]] if snapshots else [self.config.initial_price]
        # Remove NaN values
        recent_prices = [p for p in recent_prices if not math.isnan(p)]

        return {
            "tick": self._orderbook.tick,
            "mid_price": self._orderbook.get_mid_price(),
            "best_bid": self._orderbook.get_best_bid(),
            "best_ask": self._orderbook.get_best_ask(),
            "spread": self._orderbook.get_spread(),
            "spread_bps": self._orderbook.get_spread_bps(),
            "order_imbalance": self._orderbook.get_order_imbalance(),
            "volatility": self._orderbook.get_volatility(),
            "last_price": self._orderbook.get_last_price(),
            "price_history": recent_prices,
            "trade_history": self._orderbook.get_trades(),
        }

    def step(self) -> dict:
        """Execute one simulation tick.

        Workflow:
            1. Agents observe market state.
            2. Agents submit orders.
            3. Orderbook matches, records trades.
            4. Update BOTH counterparties' positions and P&L.
            5. Advance tick.

        Returns:
            Dict with tick summary: tick number, mid price, volume, trades.
        """
        market_state = self.get_state()

        # Build ID -> agent map for counterparty lookup
        agent_map = {a.agent_id: a for a in self._agents}

        # Collect orders from all agents
        all_fills: list[Trade] = []
        for agent in self._agents:
            orders = agent.generate_orders(market_state)
            for side, order_type, price, qty in orders:
                fills = self._orderbook.add_order(agent.agent_id, side, price, qty, order_type)
                for fill in fills:
                    # Update BOTH counterparties
                    buyer = agent_map.get(fill.buyer_id)
                    seller = agent_map.get(fill.seller_id)
                    if buyer:
                        buyer.update_position("buy", fill.quantity, fill.price)
                    if seller:
                        seller.update_position("sell", fill.quantity, fill.price)

                    all_fills.append(fill)

        # Advance tick
        self._orderbook.advance_tick()

        # Record price and spread
        mid = self._orderbook.get_mid_price()
        if not math.isnan(mid):
            self._price_history.append(mid)

        spread = self._orderbook.get_spread()
        if not math.isnan(spread):
            self._spread_history.append(spread)
        else:
            self._spread_history.append(0.0)

        return {
            "tick": self._orderbook.tick,
            "mid_price": mid,
            "volume": sum(f.quantity for f in all_fills),
            "trades": len(all_fills),
        }

    def run(self, verbose: bool = True) -> SimulationResult:
        """Run the full simulation.

        Args:
            verbose: If True, print progress every 500 ticks.

        Returns:
            SimulationResult with complete run data.
        """
        print(f"\n{'='*60}")
        print(f"  Agent Trade Sim — Multi-Agent Market Simulation")
        print(f"{'='*60}")
        print(f"  Agents: {self.config.n_value} value, {self.config.n_momentum} momentum, "
              f"{self.config.n_noise} noise, {self.config.n_mm} market makers")
        print(f"  Ticks:  {self.config.n_ticks}")
        print(f"  Price:  {self.config.initial_price:.2f}")
        print(f"  Seed:   {self.config.seed}")
        print(f"{'='*60}\n")

        start_time = time.time()
        timestamp = datetime.now().isoformat()

        for tick in range(self.config.n_ticks):
            self.step()

            if verbose and (tick + 1) % 500 == 0:
                mid = self._orderbook.get_mid_price()
                vol = self._orderbook.get_total_volume()
                trades = self._orderbook.get_trade_count()
                print(f"  [Tick {tick + 1:5d}] Mid: {mid:8.2f}  "
                      f"Vol: {vol:8.0f}  Trades: {trades:5d}")

        end_time = time.time()
        duration = end_time - start_time

        # Final stats
        mid = self._orderbook.get_mid_price()
        total_trades = self._orderbook.get_trade_count()
        total_vol = self._orderbook.get_total_volume()
        print(f"\n  Complete! {total_trades} trades, {total_vol:.0f} volume, "
              f"final price: {mid:.2f}")
        print(f"  Duration: {duration:.1f}s\n")

        return self._build_result(timestamp, duration)

    def get_agent_stats(self) -> list[AgentStats]:
        """Return per-type aggregate agent statistics.

        PnL values include both realized and unrealized components,
        computed at the current mid price.
        """
        mid_price = self._orderbook.get_mid_price() if self._orderbook else 100.0
        if math.isnan(mid_price):
            mid_price = 100.0

        groups: dict[str, list[BaseAgent]] = {}
        for agent in self._agents:
            groups.setdefault(agent.agent_type, []).append(agent)

        stats_list = []
        for atype, agents in groups.items():
            total_realized = sum(a.pnl for a in agents)
            total_unrealized = sum(a.get_unrealized_pnl(mid_price) for a in agents)
            total_pnl = total_realized + total_unrealized
            total_equity = sum(a.get_equity(mid_price) for a in agents)
            total_trades = sum(a.trade_count for a in agents)
            net_position = sum(a.position for a in agents)
            avg_pnl = total_pnl / len(agents) if agents else 0

            winners = sum(1 for a in agents if a.get_total_pnl(mid_price) > 0)
            win_rate = winners / len(agents) if agents else 0

            stats_list.append(AgentStats(
                agent_type=atype,
                count=len(agents),
                total_pnl=total_pnl,
                total_unrealized=total_unrealized,
                total_equity=total_equity,
                total_trades=total_trades,
                avg_pnl=avg_pnl,
                win_rate=win_rate,
                net_position=net_position,
            ))

        return stats_list

    def _build_result(self, timestamp: str, duration: float) -> SimulationResult:
        """Build the SimulationResult from current state."""
        cfg = self.config
        snapshots = self._orderbook.get_snapshots()

        # Price history: use snapshot mid prices (filter NaN)
        price_hist = [s.mid_price for s in snapshots if not math.isnan(s.mid_price)]
        spread_hist = [s.spread if not math.isnan(s.spread) else 0.0 for s in snapshots]
        vol_profile = [s.volume for s in snapshots]

        # Convert snapshots to dicts
        snapshot_dicts = [
            {
                "tick": s.tick,
                "best_bid": s.best_bid if not math.isnan(s.best_bid) else None,
                "best_ask": s.best_ask if not math.isnan(s.best_ask) else None,
                "mid_price": s.mid_price if not math.isnan(s.mid_price) else None,
                "spread": s.spread,
                "spread_bps": s.spread_bps,
                "volume": s.volume,
                "order_imbalance": s.order_imbalance,
            }
            for s in snapshots
        ]

        # Agent stats
        agent_stats = self.get_agent_stats()
        agent_stats_dicts = [
            {
                "agent_type": s.agent_type,
                "count": s.count,
                "total_pnl": s.total_pnl,
                "total_unrealized": s.total_unrealized,
                "total_equity": s.total_equity,
                "total_trades": s.total_trades,
                "avg_pnl": s.avg_pnl,
                "win_rate": s.win_rate,
                "net_position": s.net_position,
            }
            for s in agent_stats
        ]

        # Per-agent details
        mid_price = self._orderbook.get_mid_price() if self._orderbook else 100.0
        if math.isnan(mid_price):
            mid_price = 100.0
        agents_detail = [
            {
                "agent_id": a.agent_id,
                "agent_type": a.agent_type,
                "capital": a.capital,
                "position": a.position,
                "realized_pnl": a.pnl,
                "unrealized_pnl": a.get_unrealized_pnl(mid_price),
                "total_pnl": a.get_total_pnl(mid_price),
                "equity": a.get_equity(mid_price),
                "trade_count": a.trade_count,
            }
            for a in self._agents
        ]

        # Periodic depth snapshots (every 100 ticks)
        depth_history = []
        for s in snapshots:
            if s.tick % 100 == 0:
                depth = self._orderbook.get_depth(5)
                depth_history.append({
                    "tick": s.tick,
                    "bids": depth["bids"],
                    "asks": depth["asks"],
                })

        return SimulationResult(
            config={
                "n_value": cfg.n_value,
                "n_momentum": cfg.n_momentum,
                "n_noise": cfg.n_noise,
                "n_mm": cfg.n_mm,
                "n_ticks": cfg.n_ticks,
                "initial_price": cfg.initial_price,
                "tick_size": cfg.tick_size,
                "symbol": cfg.symbol,
                "seed": cfg.seed,
            },
            timestamp=timestamp,
            duration_seconds=duration,
            total_ticks=len(snapshots),
            total_trades=self._orderbook.get_trade_count(),
            total_volume=self._orderbook.get_total_volume(),
            final_price=mid_price,
            price_history=price_hist,
            spread_history=spread_hist,
            volume_profile=vol_profile,
            snapshots=snapshot_dicts,
            agent_stats=agent_stats_dicts,
            agents_detail=agents_detail,
            orderbook_depth_history=depth_history,
        )

    @staticmethod
    def save_result(result: SimulationResult, filepath: str) -> None:
        """Save simulation result to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert result to serializable dict
        data = {
            "config": result.config,
            "timestamp": result.timestamp,
            "duration_seconds": result.duration_seconds,
            "total_ticks": result.total_ticks,
            "total_trades": result.total_trades,
            "total_volume": result.total_volume,
            "final_price": result.final_price,
            "price_history": result.price_history,
            "spread_history": result.spread_history,
            "volume_profile": result.volume_profile,
            "snapshots": result.snapshots,
            "agent_stats": result.agent_stats,
            "agents_detail": result.agents_detail,
            "orderbook_depth_history": result.orderbook_depth_history,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        print(f"  Results saved to {filepath}")

    @staticmethod
    def load_result(filepath: str) -> "SimulationResult":
        """Load a saved SimulationResult from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return SimulationResult(
            config=data["config"],
            timestamp=data["timestamp"],
            duration_seconds=data["duration_seconds"],
            total_ticks=data["total_ticks"],
            total_trades=data["total_trades"],
            total_volume=data["total_volume"],
            final_price=data["final_price"],
            price_history=data["price_history"],
            spread_history=data["spread_history"],
            volume_profile=data["volume_profile"],
            snapshots=data["snapshots"],
            agent_stats=data["agent_stats"],
            agents_detail=data["agents_detail"],
            orderbook_depth_history=data.get("orderbook_depth_history", []),
        )
