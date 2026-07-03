"""
orderbook.py — Pure Python Limit Order Book Matching Engine
============================================================
Implements a price-time priority limit order book with support for
partial fills, cancellations, market depth queries, and trade history
tracking. Designed to be efficient for thousands of agents interacting
at each simulation tick.
"""

import bisect
import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Order:
    """Represents a single order in the order book.

    Attributes:
        order_id: Unique identifier for this order.
        agent_id: The agent that submitted this order.
        side: 'buy' or 'sell'.
        price: Limit price (must be positive).
        quantity: Remaining quantity to fill.
        timestamp: Creation tick number.
        order_type: 'limit' or 'market'.
    """

    order_id: int
    agent_id: str
    side: str
    price: float
    quantity: float
    timestamp: int
    order_type: str = "limit"

    def __lt__(self, other: "Order") -> bool:
        """Comparison for heap ordering: buy orders prefer higher price (max-heap),
        sell orders prefer lower price (min-heap). Both break ties by timestamp."""
        if self.side == "buy":
            # Max-heap: higher price = higher priority
            if self.price != other.price:
                return self.price > other.price
        else:
            # Min-heap: lower price = higher priority
            if self.price != other.price:
                return self.price < other.price
        return self.timestamp < other.timestamp


@dataclass
class Trade:
    """Records a completed trade between two agents.

    Attributes:
        timestamp: The tick when the trade occurred.
        price: Execution price (the resting order's price).
        quantity: Number of shares traded.
        buyer_id: The agent that bought.
        seller_id: The agent that sold.
        aggressor_side: 'buy' if the incoming order crossed the spread, 'sell' otherwise.
    """

    timestamp: int
    price: float
    quantity: float
    buyer_id: str
    seller_id: str
    aggressor_side: str


@dataclass
class MarketSnapshot:
    """A point-in-time snapshot of market state.

    Attributes:
        tick: Simulation tick number.
        best_bid: Highest bid price (or NaN).
        best_ask: Lowest ask price (or NaN).
        mid_price: Midpoint of best bid and ask.
        spread: Bid-ask spread.
        spread_bps: Spread in basis points.
        bid_depth: Total volume on bid side (top levels).
        ask_depth: Total volume on ask side (top levels).
        order_imbalance: Bid vol / (bid vol + ask vol).
        last_price: Most recent trade price (or NaN).
        volume: Total volume traded at this tick.
    """

    tick: int
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    spread_bps: float
    bid_depth: float
    ask_depth: float
    order_imbalance: float
    last_price: float
    volume: float


class LimitOrderBook:
    """Price-time priority limit order book matching engine.

    Maintains separate bid (buy) and ask (sell) books. Buy orders are stored
    in a max-heap (highest price first), sell orders in a min-heap (lowest
    price first). Ties are broken by timestamp (earlier first).

    Parameters:
        symbol: The trading instrument symbol.
        tick_size: Minimum price increment (default 0.01).
        initial_price: Starting reference price.
    """

    def __init__(
        self,
        symbol: str = "SIM",
        tick_size: float = 0.01,
        initial_price: float = 100.0,
    ):
        self.symbol = symbol
        self.tick_size = tick_size
        self._initial_price = initial_price

        # Order ID counter
        self._next_order_id = 1

        # Active orders keyed by order_id
        self._orders: dict[int, Order] = {}

        # Bid heap: stores Order objects, sorted by __lt__ (max price, min timestamp)
        self._bids: list[Order] = []
        # Ask heap: stores Order objects, sorted by __lt__ (min price, min timestamp)
        self._asks: list[Order] = []

        # Trade history
        self._trades: list[Trade] = []

        # Price history for volatility calculation
        self._price_history: deque[float] = deque(maxlen=500)

        # Current tick
        self._tick: int = 0

        # Market snapshots
        self._snapshots: list[MarketSnapshot] = []

    # ------------------------------------------------------------------
    # Public API: Order Submission and Cancellation
    # ------------------------------------------------------------------

    def add_order(
        self,
        agent_id: str,
        side: str,
        price: float,
        quantity: float,
        order_type: str = "limit",
    ) -> list[Trade]:
        """Submit an order to the book. Returns list of fills (trades).

        For market orders (order_type='market'), price is ignored and the
        order crosses to the best available counterparty price.

        Args:
            agent_id: Identifier of the submitting agent.
            side: 'buy' or 'sell'.
            price: Limit price (ignored for market orders).
            quantity: Order quantity (must be > 0).
            order_type: 'limit' (default) or 'market'.

        Returns:
            List of Trade objects representing fills from this order.
        """
        if quantity <= 0:
            return []

        if order_type == "market":
            return self._execute_market_order(agent_id, side, quantity)

        # Round price to tick size
        price = self._round_to_tick(price)
        if price <= 0:
            return []

        order = Order(
            order_id=self._next_order_id,
            agent_id=agent_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=self._tick,
            order_type=order_type,
        )
        self._next_order_id += 1

        # Attempt to match immediately
        fills = self._match(order)
        # If residual remains, rest in the book
        if order.quantity > 0:
            self._orders[order.order_id] = order
            if side == "buy":
                heapq.heappush(self._bids, order)
            else:
                heapq.heappush(self._asks, order)

        return fills

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending limit order by its ID.

        Returns True if the order was found and cancelled, False otherwise.
        """
        order = self._orders.get(order_id)
        if order is None:
            return False
        order.quantity = 0  # Mark as zero; lazily cleaned on pop
        del self._orders[order_id]
        return True

    # ------------------------------------------------------------------
    # Public API: Market State Queries
    # ------------------------------------------------------------------

    def get_best_bid(self) -> float:
        """Return the highest bid price, or NaN if no bids."""
        self._clean_top("buy")
        if not self._bids:
            return float("nan")
        return self._bids[0].price

    def get_best_ask(self) -> float:
        """Return the lowest ask price, or NaN if no asks."""
        self._clean_top("sell")
        if not self._asks:
            return float("nan")
        return self._asks[0].price

    def get_mid_price(self) -> float:
        """Return the midpoint of best bid and best ask.

        If one side is missing, returns the available price or the last trade price.
        Falls back to initial_price if no trades have occurred.
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if not math.isnan(best_bid) and not math.isnan(best_ask):
            return (best_bid + best_ask) / 2.0
        if not math.isnan(best_bid):
            return best_bid
        if not math.isnan(best_ask):
            return best_ask
        if self._trades:
            return self._trades[-1].price
        return self._initial_price

    def get_spread(self) -> float:
        """Return the absolute bid-ask spread, or NaN."""
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if math.isnan(bb) or math.isnan(ba):
            return float("nan")
        return ba - bb

    def get_spread_bps(self) -> float:
        """Return the spread in basis points relative to mid price."""
        mid = self.get_mid_price()
        spread = self.get_spread()
        if math.isnan(spread) or mid == 0:
            return float("nan")
        return (spread / mid) * 10000.0

    def get_depth(self, levels: int = 5) -> dict[str, list[tuple[float, float]]]:
        """Return L2 order book depth: top N price levels and aggregate volume.

        Returns:
            dict with 'bids' and 'asks', each a list of (price, volume) tuples.
        """
        self._clean_all()

        bid_levels = self._aggregate_levels(self._bids, "buy", levels)
        ask_levels = self._aggregate_levels(self._asks, "sell", levels)

        return {"bids": bid_levels, "asks": ask_levels}

    def get_order_imbalance(self, depth: int = 3) -> float:
        """Return bid-side volume / total volume ratio for top `depth` levels.

        A value > 0.5 indicates more buying pressure; < 0.5 more selling pressure.
        """
        d = self.get_depth(depth)
        bid_vol = sum(v for _, v in d["bids"])
        ask_vol = sum(v for _, v in d["asks"])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.5
        return bid_vol / total

    def get_volatility(self, window: int = 20) -> float:
        """Return rolling realized volatility over the last `window` trades.

        Uses trade-to-trade log returns. This is tick-level volatility
        (not annualized). Returns NaN if insufficient data.
        """
        if len(self._price_history) < 2:
            return float("nan")
        prices = list(self._price_history)[-window:]
        if len(prices) < 2:
            return float("nan")
        log_returns = np.diff(np.log(prices))
        std = np.std(log_returns, ddof=1) if len(log_returns) > 1 else 0.0
        # Return raw tick-level volatility (non-annualized)
        return float(std)

    def get_trades(self) -> list[Trade]:
        """Return the complete trade history."""
        return list(self._trades)

    def get_trade_count(self) -> int:
        """Return the total number of trades executed."""
        return len(self._trades)

    def get_total_volume(self) -> float:
        """Return the total traded volume."""
        return sum(t.quantity for t in self._trades)

    def get_last_price(self) -> float:
        """Return the most recent trade price, or NaN."""
        if not self._trades:
            return float("nan")
        return self._trades[-1].price

    def get_snapshots(self) -> list[MarketSnapshot]:
        """Return the list of per-tick market snapshots."""
        return list(self._snapshots)

    def get_orders(self) -> dict[int, Order]:
        """Return a copy of active orders dict."""
        return dict(self._orders)

    # ------------------------------------------------------------------
    # Internal: Tick Lifecycle
    # ------------------------------------------------------------------

    def advance_tick(self) -> None:
        """Advance the simulation tick counter and record a snapshot."""
        self._tick += 1
        self._record_snapshot()

    @property
    def tick(self) -> int:
        return self._tick

    # ------------------------------------------------------------------
    # Private: Matching Engine
    # ------------------------------------------------------------------

    def _match(self, incoming: Order) -> list[Trade]:
        """Match an incoming order against the resting book.

        For a buy order, match against resting asks (lowest first).
        For a sell order, match against resting bids (highest first).
        Partial fills leave residual quantity on the incoming order.
        """
        fills: list[Trade] = []

        if incoming.side == "buy":
            resting = self._asks
            # Buy matches when bid >= ask
            can_match = lambda inc, rest: inc.price >= rest.price
        else:
            resting = self._bids
            # Sell matches when ask <= bid
            can_match = lambda inc, rest: inc.price <= rest.price

        while incoming.quantity > 0 and resting:
            self._clean_top("sell" if incoming.side == "buy" else "buy")
            if not resting:
                break

            top = resting[0]
            if not can_match(incoming, top):
                break

            # Pop the top order
            heapq.heappop(resting)
            if top.order_id in self._orders:
                del self._orders[top.order_id]

            fill_qty = min(incoming.quantity, top.quantity)
            incoming.quantity -= fill_qty
            top.quantity -= fill_qty

            # Trade executes at the resting order's price
            trade = Trade(
                timestamp=self._tick,
                price=top.price,
                quantity=fill_qty,
                buyer_id=incoming.agent_id if incoming.side == "buy" else top.agent_id,
                seller_id=incoming.agent_id if incoming.side == "sell" else top.agent_id,
                aggressor_side=incoming.side,
            )
            fills.append(trade)
            self._trades.append(trade)
            self._price_history.append(top.price)

            # If resting order still has quantity, push it back
            if top.quantity > 0:
                heapq.heappush(resting, top)
                self._orders[top.order_id] = top

        return fills

    def _execute_market_order(
        self, agent_id: str, side: str, quantity: float
    ) -> list[Trade]:
        """Execute a market order by crossing the spread to available liquidity."""
        order = Order(
            order_id=self._next_order_id,
            agent_id=agent_id,
            side=side,
            price=float("inf") if side == "buy" else 0.0,  # matches anything
            quantity=quantity,
            timestamp=self._tick,
            order_type="market",
        )
        self._next_order_id += 1
        return self._match(order)

    # ------------------------------------------------------------------
    # Private: Book Maintenance
    # ------------------------------------------------------------------

    def _clean_top(self, side: str) -> None:
        """Remove zero-quantity orders from the top of the specified side."""
        heap = self._bids if side == "buy" else self._asks
        while heap and heap[0].quantity <= 0:
            heapq.heappop(heap)

    def _clean_all(self) -> None:
        """Remove all zero-quantity orders from both sides."""
        self._clean_top("buy")
        self._clean_top("sell")
        # Additional sweep for stale entries
        self._bids = [o for o in self._bids if o.quantity > 0]
        heapq.heapify(self._bids)
        self._asks = [o for o in self._asks if o.quantity > 0]
        heapq.heapify(self._asks)

    def _aggregate_levels(
        self, orders: list[Order], side: str, levels: int
    ) -> list[tuple[float, float]]:
        """Aggregate orders at the same price level and return top N."""
        price_vol: dict[float, float] = {}
        for o in orders:
            if o.quantity > 0:
                price_vol[o.price] = price_vol.get(o.price, 0.0) + o.quantity

        if side == "buy":
            sorted_prices = sorted(price_vol.keys(), reverse=True)
        else:
            sorted_prices = sorted(price_vol.keys())

        return [(p, price_vol[p]) for p in sorted_prices[:levels]]

    def _record_snapshot(self) -> None:
        """Record a market snapshot for the current tick."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        mid_price = self.get_mid_price()
        spread = self.get_spread()
        spread_bps = self.get_spread_bps()

        depth = self.get_depth(5)
        bid_depth = sum(v for _, v in depth["bids"])
        ask_depth = sum(v for _, v in depth["asks"])
        order_imbalance = self.get_order_imbalance(5)
        last_price = self.get_last_price()

        # Volume for current tick
        tick_trades = [t for t in self._trades if t.timestamp == self._tick]
        volume = sum(t.quantity for t in tick_trades)

        snapshot = MarketSnapshot(
            tick=self._tick,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread if not math.isnan(spread) else 0.0,
            spread_bps=spread_bps if not math.isnan(spread_bps) else 0.0,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            order_imbalance=order_imbalance,
            last_price=last_price,
            volume=volume,
        )
        self._snapshots.append(snapshot)

    def _round_to_tick(self, price: float) -> float:
        """Round a price to the nearest tick increment."""
        return round(price / self.tick_size) * self.tick_size
