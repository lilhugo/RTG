# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
import collections
import numpy as np
from math import ceil, floor
from scipy.optimize import curve_fit

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


# about time
SPEED = 1
TICK_INTERVAL = 0.25
END_TIME = 900

# about price
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS

# about volume
LOT_SIZE = 10
LOT_SIZE_MULTIPLIER = 5
POSITION_LIMIT = 100

# about volatility indicator
BUFFER_VOLATILITY_SIZE = 200
MIN_TO_COMPUTE_VOLATILITY = 150

# about kappa indicator
BUFFER_LAST_QUOTED_SIZE = 200
MIN_TO_COMPUTE_KAPPA = 50
TRESHOLD_SIGMA_MAX = 0.4
TRESHOLD_SIGMA_MIN = 0.2

# Parameters to improve the performance of the strategy
SIGMA = 0.25 # volatility
GAMMA = 0.01 # risk aversion
ALPHA = 0.9 # order book intensity parameter
KAPPA = 0.4 # order book depth parameter

ORDER_OPTIMIZATION_ENABLED = False #Allows the bid and ask order prices to be adjusted based on the current top bid and ask prices in the market
PRICE_QUANTUM = 1 #The minimum price increment that the bid and ask prices can be adjusted by above and below the current top bid and ask prices in the market
MIN_SPREAD_PCT = 0.4 # minimum spread in function of the mid price as a percentage
#MAX_SPREAD = 5 # maximum spread in function of the mid price as a percentage


class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        
        # to compute time to maturity
        self.start = self.event_loop.time()
        self.now = self.start
        
        # about indicators
        self.volatility_indicator = VolatilityIndicator()
        self.kappa_indicator = KappaIndicator()
        self.treshold_sigma_max = TRESHOLD_SIGMA_MAX
        self.treshold_sigma_min = TRESHOLD_SIGMA_MIN
        
        # for Avellaneda-Stoikov
        self.sigma = SIGMA
        self.gamma = GAMMA
        self.kappa = KAPPA
        
        # about orders
        self.order_optimization_enabled = ORDER_OPTIMIZATION_ENABLED
        self.last_etf_best_bid = 0
        self.last_etf_best_ask = 10**9
        self.min_spread_pct = MIN_SPREAD_PCT
        
        
    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        #self.logger.info("received order book for instrument %d with sequence number %d", instrument, sequence_number)
        self.logger.info(f'instrument: {instrument}, sequence number: {sequence_number}, mid price: {(ask_prices[0] + bid_prices[0]) / 2} \n bid prices: {bid_prices} \n bid volumes: {bid_volumes}  \n ask prices: {ask_prices} \n ask volumes: {ask_volumes}')

        
        if instrument == Instrument.FUTURE and sequence_number == 1:
            self.start = self.event_loop.time()
            
        if instrument == Instrument.FUTURE and sequence_number > 1:
            
            # new mid price
            self.mid_price = (ask_prices[0] + bid_prices[0]) / 2
            
            # we compute the time to maturity
            self.now = (self.event_loop.time()-self.start)*SPEED
            T_minus_t = (END_TIME - self.now) / END_TIME
            
            self.logger.info("mid price, volatility, kappa, time to maturity, position: %d, %f, %f, %f, %d", self.mid_price, self.sigma, self.kappa, T_minus_t, self.position)
            
            # Avellaneda-Stoikov
            reservation_price = self.mid_price - self.position * self.gamma * self.sigma**2 * T_minus_t * TICK_SIZE_IN_CENTS
            optimal_spread = self.gamma * self.sigma**2 * T_minus_t + 2 * np.log(1 + self.gamma/self.kappa)/self.gamma
            optimal_spread *= TICK_SIZE_IN_CENTS
            min_spread = self.mid_price / 100 * self.min_spread_pct
            #self.logger.info("reservation price: %d, optimal spread: %d, min spread: %d", reservation_price, optimal_spread, min_spread)
            
            # we compute the new bid and ask prices
            bid_avellaneda = int((reservation_price - optimal_spread/2) // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
            ask_avellaneda = int((reservation_price + optimal_spread/2) // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
            #self.logger.info("bid avellaneda: %d, ask avellaneda: %d", bid_avellaneda, ask_avellaneda)
            
            # we compute the max bid and ask prices
            max_bid_price = int((self.mid_price - min_spread/2) // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
            min_ask_price = int((self.mid_price + min_spread/2) // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
            #self.logger.info("max bid price: %d, min ask price: %d", max_bid_price, min_ask_price)
            
            # new bid and ask prices
            new_bid_price = min(max_bid_price, bid_avellaneda)
            new_ask_price = max(min_ask_price, ask_avellaneda)
            #self.logger.info("new bid price: %d, new ask price: %d", new_bid_price, new_ask_price)
            
            # we adjust the bid and ask prices
            if self.order_optimization_enabled:
                price_above_bid = bid_prices[0] + PRICE_QUANTUM * TICK_SIZE_IN_CENTS
                price_below_ask = ask_prices[0] - PRICE_QUANTUM * TICK_SIZE_IN_CENTS
                if new_bid_price > price_above_bid:
                    new_bid_price = price_above_bid
                if new_ask_price < price_below_ask:
                    new_ask_price = price_below_ask
                #self.logger.info("after optimization bid: %d, ask: %d", new_bid_price, new_ask_price)

            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0) or self.bid_price < bid_prices[-1] - TICK_SIZE_IN_CENTS:
                self.send_cancel_order(self.bid_id)
                self.logger.info("bid %d to cancel", self.bid_id)
            if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0) or self.ask_price > ask_prices[-1] + TICK_SIZE_IN_CENTS:
                self.send_cancel_order(self.ask_id)
                self.logger.info("ask %d to cancel", self.ask_id)

            if self.bid_id == 0 and new_bid_price != 0 and self.position < POSITION_LIMIT:
                self.bid_id = next(self.order_ids)
                self.bid_price = new_bid_price
                
                if self.position == -POSITION_LIMIT:
                    volume_bid = 175
                    self.send_insert_order(self.bid_id, Side.BUY, self.bid_price + TICK_SIZE_IN_CENTS, volume_bid, Lifespan.GOOD_FOR_DAY)
                    self.logger.info(f"bid order {self.bid_id} inserted at price {self.bid_price} for {volume_bid} lots")
                else:
                    volume_bid = min(LOT_SIZE*LOT_SIZE_MULTIPLIER, POSITION_LIMIT - self.position)
                    self.send_insert_order(self.bid_id, Side.BUY, self.bid_price, volume_bid, Lifespan.GOOD_FOR_DAY)
                    self.logger.info(f"bid order {self.bid_id} inserted at price {self.bid_price} for {volume_bid} lots")
                
                self.bids.add(self.bid_id)

            if self.ask_id == 0 and new_ask_price != 0 and self.position > -POSITION_LIMIT:
                self.ask_id = next(self.order_ids)
                self.ask_price = new_ask_price
                
                if self.position == POSITION_LIMIT:
                    volume_ask = 175
                    self.send_insert_order(self.ask_id, Side.SELL, self.ask_price - TICK_SIZE_IN_CENTS, volume_ask, Lifespan.GOOD_FOR_DAY)
                    self.logger.info(f"ask order {self.ask_id} inserted at price {self.ask_price} for {volume_ask} lots")
                else:
                    volume_ask = min(LOT_SIZE*LOT_SIZE_MULTIPLIER, POSITION_LIMIT + self.position)
                    self.send_insert_order(self.ask_id, Side.SELL, self.ask_price, volume_ask, Lifespan.GOOD_FOR_DAY)
                    self.logger.info(f"ask order {self.ask_id} inserted at price {self.ask_price} for {volume_ask} lots")
                    
                self.asks.add(self.ask_id)
        
        if instrument == Instrument.ETF and sequence_number > 1:
            #self.kappa_indicator.update(ask_prices, ask_volumes, bid_prices, bid_volumes)
            pass
            
        

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
            self.send_cancel_order(client_order_id)
            self.logger.info("ask order %d cancelled after being filled at price %d and volume %d", client_order_id, price, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume)
            #self.send_cancel_order(client_order_id)
            self.logger.info("ask order %d cancelled after being filled at price %d and volume %d", client_order_id, price, volume)


    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0 and fill_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
                self.logger.info("bid order %d was cancelled or rejected", client_order_id)
                self.bids.discard(client_order_id)
            elif client_order_id == self.ask_id:
                self.ask_id = 0
                self.logger.info("ask order %d was cancelled or rejected", client_order_id)
                self.asks.discard(client_order_id)
        elif remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
                self.bids.discard(client_order_id)
            elif client_order_id == self.ask_id:
                self.ask_id = 0
                self.bids.discard(client_order_id)            
            

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)
        
        if instrument == Instrument.ETF and sequence_number > 1:
            #self.kappa_indicator.update(ask_prices, ask_volumes, bid_prices, bid_volumes)
            pass
        
        if instrument == Instrument.FUTURE and sequence_number > 1:
            self.mid_price = (ask_prices[0] + bid_prices[0]) / 2
            self.volatility_indicator.add_sample(self.mid_price)
            self.sigma = self.volatility_indicator.current_volatility()
            
            # we adjust kappa and sigma values depends on the volatility and the trend
            if self.sigma > self.treshold_sigma_max:
                #self.kappa = self.kappa_indicator.current_kappa()
                self.kappa = KAPPA + 0.05
                self.sigma = SIGMA - 0.005
            elif self.sigma < self.treshold_sigma_min:
                #self.kappa = self.kappa_indicator.current_kappa()
                self.kappa = KAPPA - 0.05
                self.sigma = SIGMA + 0.005
            elif self.sigma > self.treshold_sigma_min and self.sigma < self.treshold_sigma_max:
                self.kappa = KAPPA
                self.sigma = SIGMA


#### Custom Class/functions ####

class VolatilityIndicator():
    """ Use a custom Ring Buffer to calculate the volatility of the last N ticks. """
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_VOLATILITY_SIZE)
        self.sigma = SIGMA
        self.min_size = MIN_TO_COMPUTE_VOLATILITY

    def add_sample(self, sample: float):
        self.buffer.append(sample)
            
    def calculation(self) -> None:
        """ Calculate the volatility"""
        if len(self.buffer) > self.min_size: 
            self.sigma = np.sqrt(np.sum(np.square(np.log(np.array(self.buffer)[:-1]) - np.log(np.array(self.buffer)[1:])))/(self.size-1))
    
    def current_volatility(self) -> float:
        self.calculation()
        return self.sigma
    
class KappaIndicator():
    """ Determine the kappa value """
        
    def __init__(self):
        self.__ask_prices: List[int] = [0]*5
        self.__bid_prices: List[int] = [0]*5
        
        self.price_levels = collections.deque(maxlen=BUFFER_LAST_QUOTED_SIZE)
        self.volume_levels = collections.deque(maxlen=BUFFER_LAST_QUOTED_SIZE)
        self.min_size = MIN_TO_COMPUTE_KAPPA
        self._alpha = ALPHA
        self._kappa = KAPPA
        self.params = 0

    def update(self, ask_prices: List[int], ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]):
        """Update the order book with the latest information
        It also computes the last traded price by comparing the current book with the previous one and give the volume traded
        """
        # determine the last quoted price knowing that bid prices are sorted in descending order and ask prices in ascending order by comparing the new book with the previous one
        mid_price = (ask_prices[0] + bid_prices[0]) // 2
        
        if self.__bid_prices[0] != bid_prices[0] and bid_volumes[0] != 0:
            self.price_levels.append(bid_prices[0])
            self.volume_levels.append(bid_volumes[0])
        if self.__ask_prices[0] != ask_prices[0] and ask_volumes[0] != 0:
            self.price_levels.append(ask_prices[0])
            self.volume_levels.append(ask_volumes[0])
        
        # update the order book
        self.__ask_prices = ask_prices
        self.__bid_prices = bid_prices
        
    def calculation(self) -> None:
        """ Calculate the kappa"""
        if len(self.price_levels) > self.min_size: 
            self.params = curve_fit(lambda t, a, b: a*np.exp(-b*t),
                               np.array(self.price_levels),
                               np.array(self.volume_levels),
                               p0=(self._alpha, self._kappa),
                               method='dogbox',
                               bounds=([0, 0], [np.inf, np.inf]))
            self._alpha = self.params[0][0]
            self._kappa = self.params[0][1]
    
    def current_kappa(self) -> float:
        self.calculation()
        return self._kappa