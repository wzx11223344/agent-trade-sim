"""
agent_trade_sim — Multi-Agent Trading Market Simulation
=======================================================
A specialized limit order book market where 4 types of trading agents
interact, prices emerge from their collective behavior, and the system
visualizes order flow dynamics in real-time.

Core modules:
    - orderbook: Limit order book matching engine
    - agents: Four heterogeneous agent types
    - market: Market coordinator and simulation loop
    - strategies: Strategy parameterization and serialization
    - viz: Real-time visualization and report generation
"""

__version__ = "1.0.0"
__author__ = "agent-trade-sim"

from agent_trade_sim.orderbook import LimitOrderBook
from agent_trade_sim.agents import (
    ValueAgent,
    MomentumAgent,
    NoiseAgent,
    MarketMakerAgent,
)
from agent_trade_sim.market import SimulatedMarket
from agent_trade_sim.strategies import (
    ValueStrategyParams,
    MomentumStrategyParams,
    NoiseStrategyParams,
    MarketMakerParams,
)

__all__ = [
    "LimitOrderBook",
    "ValueAgent",
    "MomentumAgent",
    "NoiseAgent",
    "MarketMakerAgent",
    "SimulatedMarket",
    "ValueStrategyParams",
    "MomentumStrategyParams",
    "NoiseStrategyParams",
    "MarketMakerParams",
]
