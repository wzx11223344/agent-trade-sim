"""
agents.py — Four Heterogeneous Trading Agent Types
====================================================
Implements the four agent archetypes that drive market dynamics:

1. ValueAgent   — Fundamental value estimation, mean-reversion
2. MomentumAgent — Trend-following, price momentum signals
3. NoiseAgent    — Random uninformed trading (liquidity provision)
4. MarketMakerAgent — Two-sided liquidity, spread capture

All agents observe market state each tick and return (optional) orders.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from agent_trade_sim.strategies import (
    ValueStrategyParams,
    MomentumStrategyParams,
    NoiseStrategyParams,
    MarketMakerParams,
)


class BaseAgent(ABC):
    """Abstract base class for all trading agents.

    Tracks agent identity, capital, position, P&L, and provides
    the common interface for strategy execution.

    Attributes:
        agent_id: Unique identifier string.
        agent_type: One of 'value', 'momentum', 'noise', 'market_maker'.
        capital: Available cash for trading.
        position: Current holding (positive = long, negative = short).
        pnl: Cumulative realized profit/loss.
        trade_count: Number of trades completed.
        last_trade_price: Price of most recent trade (for P&L calc).
    """

    def __init__(self, agent_id: str, agent_type: str, initial_capital: float = 100000.0):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capital = initial_capital
        self.position = 0.0
        self.pnl = 0.0
        self.trade_count = 0
        self._last_trade_price = 0.0
        self._avg_entry_price = 0.0
        self._rng = np.random.RandomState()

    def seed(self, seed: int) -> None:
        """Set the random seed for this agent's RNG."""
        self._rng = np.random.RandomState(seed)

    def update_position(self, side: str, quantity: float, price: float) -> None:
        """Update position and P&L after a trade fill.

        Args:
            side: 'buy' or 'sell' — the agent's side.
            quantity: Number of shares filled.
            price: Execution price.
        """
        if side == "buy":
            cost = quantity * price
            # P&L for buying: cover short if any, rest becomes new long
            if self.position < 0:
                covered = min(quantity, -self.position)
                self.pnl += covered * (self._avg_entry_price - price)
                remaining = quantity - covered
                if remaining > 0:
                    self._avg_entry_price = price  # new long at current price
                elif self.position + quantity < 0:
                    pass  # still short, _avg_entry_price tracks short
                else:
                    self._avg_entry_price = 0  # flat
            elif self.position > 0:
                # Adding to long: average up
                self._avg_entry_price = (
                    (self.position * self._avg_entry_price + quantity * price)
                    / (self.position + quantity)
                )
            else:
                self._avg_entry_price = price
            self.position += quantity
            self.capital -= cost
        else:  # sell
            revenue = quantity * price
            if self.position > 0:
                # Selling from long: realize P&L on sold portion
                sold = min(quantity, self.position)
                self.pnl += sold * (price - self._avg_entry_price)
                remaining = quantity - sold
                if remaining > 0:
                    # Went from long to short: track new short entry
                    self._avg_entry_price = price
                elif self.position - quantity > 0:
                    pass  # still long, _avg_entry_price unchanged
            elif self.position < 0:
                # Adding to short: average down
                self._avg_entry_price = (
                    (abs(self.position) * self._avg_entry_price + quantity * price)
                    / (abs(self.position) + quantity)
                )
            else:
                self._avg_entry_price = price  # new short
            self.position -= quantity
            self.capital += revenue

        self.trade_count += 1
        self._last_trade_price = price

    def get_equity(self, current_price: float) -> float:
        """Return total equity = capital + unrealized P&L at current price."""
        unrealized = self.position * (current_price - self._avg_entry_price)
        return self.capital + unrealized

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Return unrealized P&L at current price."""
        if self.position == 0:
            return 0.0
        return self.position * (current_price - self._avg_entry_price)

    def get_total_pnl(self, current_price: float) -> float:
        """Return total P&L = realized + unrealized."""
        return self.pnl + self.get_unrealized_pnl(current_price)

    @abstractmethod
    def generate_orders(self, market_state: dict) -> list[tuple[str, str, float, float]]:
        """Observe market state and generate trading orders.

        Args:
            market_state: Dict with keys like 'mid_price', 'spread', 'best_bid',
                'best_ask', 'order_imbalance', 'volatility', 'price_history',
                'trade_history', 'tick'.

        Returns:
            List of (side, order_type, price, quantity) tuples.
            Empty list means no orders this tick.
        """
        ...


class ValueAgent(BaseAgent):
    """A fundamental-value trader that estimates intrinsic value with noise.

    Strategy:
        1. Periodically update a noisy estimate of fundamental value.
        2. When market price deviates from estimated value beyond a confidence
           threshold, place a mean-reverting order.
        3. Buy when price < value, sell when price > value.
        4. Order aggressiveness scales with deviation magnitude.

    This agent represents informed investors who have a (noisy) view of
    the asset's fair price and trade to profit from mispricing.
    """

    def __init__(
        self,
        agent_id: str,
        initial_capital: float = 100000.0,
        params: Optional[ValueStrategyParams] = None,
        initial_price: float = 100.0,
    ):
        super().__init__(agent_id, "value", initial_capital)
        self.params = params or ValueStrategyParams()
        self._fundamental_value = initial_price
        self._value_confidence = 0.5  # 0-1, grows with consistent observations

    def _update_fundamental_value(self, price_history: list[float]) -> None:
        """Update the noisy fundamental value estimate.

        Uses recent prices as a baseline, adds Gaussian noise. The noise
        represents the agent's imperfect information about true value.
        """
        if not price_history:
            return

        # Baseline: exponential moving average of recent prices
        baseline = price_history[-1]
        if len(price_history) >= 2:
            alpha = 0.02
            for p in reversed(price_history[-self.params.lookback:]):
                baseline = alpha * p + (1 - alpha) * baseline

        # Add noise proportional to recent volatility
        if len(price_history) >= 5:
            recent = price_history[-min(20, len(price_history)):]
            local_vol = np.std(recent) / np.mean(recent) if np.mean(recent) > 0 else 0.01
        else:
            local_vol = 0.01

        noise = self._rng.normal(0, self.params.noise_std * (1 + local_vol * 10))
        self._fundamental_value = baseline * (1 + noise)

    def generate_orders(self, market_state: dict) -> list[tuple[str, str, float, float]]:
        mid_price = market_state.get("mid_price", 100.0)
        price_history = market_state.get("price_history", [])
        volatility = market_state.get("volatility", 0.01)

        # Occasionally update value estimate
        if self._rng.random() < self.params.value_update_rate:
            self._update_fundamental_value(price_history)

        value = self._fundamental_value
        if value <= 0 or math.isnan(mid_price):
            return []

        deviation = (mid_price - value) / value

        # Only trade if deviation exceeds confidence threshold
        if abs(deviation) < self.params.confidence_threshold:
            return []

        # Position limits
        if deviation < 0 and self.position >= self.params.max_position:
            return []
        if deviation > 0 and self.position <= -self.params.max_position:
            return []

        # Determine side and price
        if deviation < 0:
            # Price below value — buy
            side = "buy"
            # Bid slightly above best bid to get filled, but below value
            best_bid = market_state.get("best_bid", mid_price * 0.99)
            best_ask = market_state.get("best_ask", mid_price * 1.01)
            target_price = mid_price
            if not math.isnan(best_ask):
                target_price = min(best_ask, value)
            order_price = target_price
        else:
            # Price above value — sell
            side = "sell"
            best_ask = market_state.get("best_ask", mid_price * 1.01)
            best_bid = market_state.get("best_bid", mid_price * 0.99)
            target_price = mid_price
            if not math.isnan(best_bid):
                target_price = max(best_bid, value)
            order_price = target_price

        # Scale order size with deviation strength and volatility adjustment
        strength = min(abs(deviation) / self.params.confidence_threshold, 5.0)
        vol_adj = 1.0 / max(volatility * 50, 0.5) if not math.isnan(volatility) else 1.0
        base_size = self.params.order_size * strength * self.params.mean_reversion_speed
        qty = max(1.0, base_size * vol_adj)
        qty += self._rng.normal(0, self.params.order_size_std)
        qty = max(1.0, min(qty, 50.0))

        return [(side, "limit", round(order_price, 2), round(qty, 1))]


class MomentumAgent(BaseAgent):
    """A trend-following trader that rides price momentum.

    Strategy:
        1. Detect trend direction using dual moving average crossover.
        2. Assess trend strength via recent return magnitude.
        3. Only trade when trend is strong enough AND volatility is acceptable.
        4. Hold positions for a configurable period, then exit.

    This agent represents CTAs, trend-following hedge funds, and retail
    traders who buy strength and sell weakness.
    """

    def __init__(
        self,
        agent_id: str,
        initial_capital: float = 100000.0,
        params: Optional[MomentumStrategyParams] = None,
    ):
        super().__init__(agent_id, "momentum", initial_capital)
        self.params = params or MomentumStrategyParams()
        self._ticks_held = 0
        self._entry_price = 0.0
        self._last_signal = 0.0

    def _compute_signal(self, price_history: list[float]) -> float:
        """Compute momentum signal from dual moving average crossover.

        Returns a value between -1 (strong downtrend) and +1 (strong uptrend).
        """
        n = len(price_history)
        short_n = min(self.params.signal_lookback_short, n)
        long_n = min(self.params.signal_lookback_long, n)

        if short_n < 2 or long_n < 2:
            return 0.0

        short_ma = np.mean(price_history[-short_n:])
        long_ma = np.mean(price_history[-long_n:])

        if long_ma == 0:
            return 0.0

        # Crossover signal normalized by long MA
        raw_signal = (short_ma - long_ma) / long_ma

        # Scale and clip
        signal = np.clip(raw_signal * 100, -1.0, 1.0)
        return signal

    def generate_orders(self, market_state: dict) -> list[tuple[str, str, float, float]]:
        price_history = market_state.get("price_history", [])
        mid_price = market_state.get("mid_price", 100.0)
        volatility = market_state.get("volatility", 0.01)
        best_bid = market_state.get("best_bid", mid_price * 0.99)
        best_ask = market_state.get("best_ask", mid_price * 1.01)

        if len(price_history) < self.params.signal_lookback_long:
            return []

        # Compute trend signal
        signal = self._compute_signal(price_history)
        trend_direction = 1 if signal > self.params.trend_threshold else (
            -1 if signal < -self.params.trend_threshold else 0
        )

        # Check holding period exit
        if self._ticks_held >= self.params.holding_period and self.position != 0:
            # Exit position
            side = "sell" if self.position > 0 else "buy"
            qty = abs(self.position)
            order_price = best_bid if side == "sell" else best_ask
            if math.isnan(order_price):
                order_price = mid_price
            self._ticks_held = 0
            return [(side, "limit", round(order_price, 2), round(qty, 1))]

        # Stop loss check
        if self.position != 0 and self._entry_price > 0:
            pnl_pct = (mid_price - self._entry_price) / self._entry_price
            if self.position > 0:
                pnl_pct = pnl_pct  # positive = gain for long
            else:
                pnl_pct = -pnl_pct  # positive = gain for short
            if pnl_pct < -self.params.stop_loss:
                side = "sell" if self.position > 0 else "buy"
                qty = abs(self.position)
                order_price = best_bid if side == "sell" else best_ask
                if math.isnan(order_price):
                    order_price = mid_price
                self._ticks_held = 0
                self._entry_price = 0
                return [(side, "limit", round(order_price, 2), round(qty, 1))]

        # No trend or already at max position
        if trend_direction == 0:
            return []
        if trend_direction == 1 and self.position >= self.params.max_position:
            return []
        if trend_direction == -1 and self.position <= -self.params.max_position:
            return []

        # Volatility filter
        vol_ok = math.isnan(volatility) or volatility < self.params.volatility_threshold
        if not vol_ok:
            return []

        # Size: scale with signal strength
        strength = abs(signal)
        base_qty = self.params.order_size * strength
        qty = max(1.0, base_qty + self._rng.normal(0, self.params.order_size_std))
        qty = min(qty, 40.0)

        if trend_direction > 0:
            # Uptrend — buy
            price = mid_price * 1.001  # slight premium to get filled
            if not math.isnan(best_ask):
                price = best_ask
            side = "buy"
        else:
            # Downtrend — sell
            price = mid_price * 0.999
            if not math.isnan(best_bid):
                price = best_bid
            side = "sell"

        self._ticks_held = 0
        self._entry_price = mid_price
        return [(side, "limit", round(price, 2), round(qty, 1))]


class NoiseAgent(BaseAgent):
    """A random uninformed trader that provides natural liquidity demand.

    Strategy:
        Randomly submits buy or sell orders with configurable probability.
        Order sizes are drawn from a log-normal distribution.
        Prices deviate randomly from the mid price.

    Without noise traders, informed agents would have no natural counterparties
    and the market would fail. Noise traders are the "grease" that keeps
    the market mechanism running.
    """

    def __init__(
        self,
        agent_id: str,
        initial_capital: float = 100000.0,
        params: Optional[NoiseStrategyParams] = None,
    ):
        super().__init__(agent_id, "noise", initial_capital)
        self.params = params or NoiseStrategyParams()

    def generate_orders(self, market_state: dict) -> list[tuple[str, str, float, float]]:
        # Decide whether to trade this tick
        if self._rng.random() > self.params.trade_probability:
            return []

        mid_price = market_state.get("mid_price", 100.0)
        if math.isnan(mid_price):
            return []

        # Position limit: force mean-reversion when near limit
        max_pos = self.params.max_position
        if self.position >= max_pos:
            # Force sell to reduce position
            side = "sell"
        elif self.position <= -max_pos:
            # Force buy to cover short
            side = "buy"
        else:
            # Random side, biased toward reducing extreme positions
            if abs(self.position) > max_pos * 0.7:
                # Skew probability to reduce position
                prob_buy = 0.3 if self.position > 0 else 0.7
            else:
                prob_buy = self.params.buy_probability
            side = "buy" if self._rng.random() < prob_buy else "sell"

        # Random size from log-normal
        log_mean = math.log(max(self.params.mean_order_size, 0.01))
        size = self._rng.lognormal(log_mean, self.params.size_std * 0.5)
        size = max(1.0, min(size, 30.0))

        # Random price deviation
        price_dev = self._rng.normal(0, self.params.price_deviation_std)
        if side == "buy":
            price = mid_price * (1 + price_dev)
            best_ask = market_state.get("best_ask", float("nan"))
            if not math.isnan(best_ask):
                price = min(price, best_ask * 1.005)
        else:
            price = mid_price * (1 - abs(price_dev) * 0.5)
            best_bid = market_state.get("best_bid", float("nan"))
            if not math.isnan(best_bid):
                price = max(price, best_bid * 0.995)

        # Decide order type
        order_type = "market" if self._rng.random() < self.params.use_market_orders else "limit"

        return [(side, order_type, round(price, 2), round(size, 1))]


class MarketMakerAgent(BaseAgent):
    """A liquidity provider that quotes two-sided prices continuously.

    Strategy:
        1. Post bid and ask quotes around the mid price with a spread.
        2. Widen spread when volatility is high (risk compensation).
        3. Skew quotes to manage inventory — if long, lower bid and ask to
           attract sells; if short, raise bid and ask to attract buys.
        4. Cancel stale quotes and replace with fresh ones on each requote cycle.

    Market makers earn the spread as profit but take inventory risk when
    the market moves against their position.
    """

    def __init__(
        self,
        agent_id: str,
        initial_capital: float = 100000.0,
        params: Optional[MarketMakerParams] = None,
        initial_price: float = 100.0,
    ):
        super().__init__(agent_id, "market_maker", initial_capital)
        self.params = params or MarketMakerParams()
        self._active_bid_id: Optional[int] = None
        self._active_ask_id: Optional[int] = None
        self._last_requote_tick = -999
        self._last_mid = initial_price

    def generate_orders(self, market_state: dict) -> list[tuple[str, str, float, float]]:
        tick = market_state.get("tick", 0)

        # Only requote on schedule
        if tick - self._last_requote_tick < self.params.requote_interval:
            return []

        self._last_requote_tick = tick

        mid_price = market_state.get("mid_price", self._last_mid)
        volatility = market_state.get("volatility", 0.01)
        best_bid = market_state.get("best_bid", float("nan"))
        best_ask = market_state.get("best_ask", float("nan"))

        if math.isnan(mid_price):
            return []

        self._last_mid = mid_price

        # Check position limit
        if abs(self.position) >= self.params.max_position:
            return []

        # Compute spread
        vol = volatility if not math.isnan(volatility) else 0.01
        spread_pct = self.params.base_spread * (1 + self.params.volatility_multiplier * vol * 30)
        spread_pct = min(spread_pct, self.params.max_spread)

        half_spread = mid_price * spread_pct / 2.0

        # Inventory skew — adjust quotes to manage inventory toward target
        inventory_ratio = (self.position - self.params.position_target) / max(self.params.max_position, 1)
        skew = inventory_ratio * self.params.inventory_aversion * mid_price

        bid_price = mid_price - half_spread - skew
        ask_price = mid_price + half_spread - skew  # same skew direction pushes both

        # Ensure bid < ask
        if bid_price >= ask_price:
            bid_price = mid_price * 0.999
            ask_price = mid_price * 1.001

        # Avoid crossing existing best prices too aggressively
        if not math.isnan(best_bid) and bid_price > best_bid:
            bid_price = max(bid_price * 0.999, mid_price * 0.995)
        if not math.isnan(best_ask) and ask_price < best_ask:
            ask_price = min(ask_price * 1.001, mid_price * 1.005)

        orders = []

        # Only quote if we have capital for buys
        if self.capital > bid_price * self.params.quote_size:
            orders.append(("buy", "limit", round(bid_price, 2), round(self.params.quote_size, 1)))

        # Only quote if we have inventory for sells (or can go short)
        if self.position > -self.params.max_position:
            orders.append(("sell", "limit", round(ask_price, 2), round(self.params.quote_size, 1)))

        return orders
