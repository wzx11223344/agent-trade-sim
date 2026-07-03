"""
strategies.py — Strategy Parameterization for Trading Agents
=============================================================
Defines the parameter sets for each agent type's trading strategy.
All parameter classes are serializable to/from YAML for easy
configuration and experimentation.
"""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ValueStrategyParams:
    """Parameters for the ValueAgent's fundamental-value strategy.

    The agent estimates a noisy fundamental value and trades when
    price deviates sufficiently from that estimate.

    Attributes:
        lookback: Number of recent prices used to estimate value baseline.
        noise_std: Standard deviation of Gaussian noise added to value estimate.
        confidence_threshold: Minimum deviation (as fraction) required to trade.
            e.g., 0.005 means trade when |price - value|/value > 0.5%.
        max_position: Maximum absolute position the agent can hold.
        order_size: Base order quantity per trade.
        order_size_std: Standard deviation of order size noise.
        value_update_rate: Probability of updating value estimate each tick.
        mean_reversion_speed: How aggressively to trade toward value (0-1).
    """

    lookback: int = 50
    noise_std: float = 0.02
    confidence_threshold: float = 0.005
    max_position: int = 100
    order_size: float = 10.0
    order_size_std: float = 3.0
    value_update_rate: float = 0.05
    mean_reversion_speed: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValueStrategyParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MomentumStrategyParams:
    """Parameters for the MomentumAgent's trend-following strategy.

    The agent detects price trends and trades in the direction of the trend,
    with filters for trend strength and volatility.

    Attributes:
        lookback: Number of ticks to use for trend detection.
        signal_lookback_short: Short window for fast signal (e.g., 5 ticks).
        signal_lookback_long: Long window for slow signal (e.g., 20 ticks).
        trend_threshold: Minimum absolute return over lookback to consider a trend.
        volatility_threshold: Above this volatility, reduce position size.
        max_position: Maximum absolute position the agent can hold.
        order_size: Base order quantity per trade.
        order_size_std: Standard deviation of order size noise.
        holding_period: Ticks to hold a position before reversing.
        stop_loss: Fractional loss that triggers position exit.
    """

    lookback: int = 20
    signal_lookback_short: int = 5
    signal_lookback_long: int = 20
    trend_threshold: float = 0.001
    volatility_threshold: float = 1.0
    max_position: int = 80
    order_size: float = 15.0
    order_size_std: float = 5.0
    holding_period: int = 30
    stop_loss: float = 0.03

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MomentumStrategyParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class NoiseStrategyParams:
    """Parameters for the NoiseAgent's random trading strategy.

    Simulates uninformed order flow that provides natural liquidity demand.
    Noise traders are essential for market realism — without them, informed
    agents would have no counterparties.

    Attributes:
        buy_probability: Probability of submitting a buy (vs sell) order each tick.
        trade_probability: Probability of trading at all on a given tick.
        mean_order_size: Mean of the log-normal distribution for order size.
        size_std: Standard deviation of order size.
        price_deviation_std: Std of price deviation from mid (as fraction).
        use_market_orders: Fraction of orders that are market orders (0-1).
    """

    buy_probability: float = 0.5
    trade_probability: float = 0.3
    mean_order_size: float = 5.0
    size_std: float = 3.0
    price_deviation_std: float = 0.005
    use_market_orders: float = 0.2
    max_position: int = 150

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NoiseStrategyParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MarketMakerParams:
    """Parameters for the MarketMakerAgent's two-sided liquidity provision.

    The market maker continuously quotes bid and ask prices, earning the
    spread while managing inventory risk. Key trade-off: wider spread =
    more profit per trade but fewer trades.

    Attributes:
        base_spread: Minimum spread (as fraction of mid price).
        volatility_multiplier: How much to widen spread when volatility is high.
        inventory_aversion: How aggressively to skew quotes to reduce inventory.
            Higher values = tighter inventory control, potentially lower profits.
        max_position: Absolute position limit before refusing to quote.
        quote_size: Quantity quoted on each side.
        requote_interval: Ticks between quote refreshes.
        max_spread: Maximum allowed spread (as fraction), cap on widening.
        position_target: Target inventory (usually 0).
    """

    base_spread: float = 0.002  # 20 bps
    volatility_multiplier: float = 2.0
    inventory_aversion: float = 0.01
    max_position: int = 200
    quote_size: float = 25.0
    requote_interval: int = 3
    max_spread: float = 0.02  # 200 bps cap
    position_target: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MarketMakerParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Registry for easy lookup
STRATEGY_PARAM_CLASSES = {
    "value": ValueStrategyParams,
    "momentum": MomentumStrategyParams,
    "noise": NoiseStrategyParams,
    "market_maker": MarketMakerParams,
}
