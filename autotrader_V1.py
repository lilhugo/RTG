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
import math
import numpy as np

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side

LOT_SIZE = 10
POSITION_LIMIT = 100

SPEED = 1
TICK_INTERVAL = 0.25
END_TIME = 900

TICK_SIZE = 1.00
TICK_SIZE_IN_CENTS = 100

MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS

# fees
MAKER_FEES =-0.0001
TAKER_FEES = 0.0002
ETF_CLAMP = 0.002

# about volatility parameters
BUFFER_VOLATILITY_SIZE = 200
MIN_TO_COMPUTE_VOLATILITY = 20

# Parameters to improve the performance of the strategy
GAMMA = 0.01 # risk aversion
KAPPA = 0.2 # order book liquidity parameter

MAX_AGE_ORDER = 30 # maximum age of an order before it is cancelled

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
        self.position = 0
        
        # about the current order books
        self.order_book_ETF = OrderBookCustom()
        self.order_book_Future = OrderBookCustom()
        
        # about outstanding orders
        self.outstanding_orders = OutsandingOrders()
        
        # about volatility
        self.volatility_indicator = VolatilityIndicator()
        self.vol_is_calculated = False
        
        # to compute time to maturity
        self.start = self.event_loop.time()
        self.now = self.start
        
        # for Avellaneda-Stoikov
        self.sigma = 0.2
        self.gamma = GAMMA
        self.kappa = KAPPA

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.info("on_error_message")
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        self.logger.info("position: %d", self.position)
        self.logger.info("theorical position: %d", self.outstanding_orders.get_theorical_position())
        self.logger.info("real position: %d", self.outstanding_orders.get_real_position())
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
        self.logger.info("on_order_book_update_message")
        self.logger.info(f'instrument: {instrument}, sequence number: {sequence_number}, mid price: {(ask_prices[0] + bid_prices[0]) / 2} \n bid prices: {bid_prices} \n bid volumes: {bid_volumes}  \n ask prices: {ask_prices} \n ask volumes: {ask_volumes}')
        self.logger.info("position: %d", self.position)
        self.logger.info("theorical position: %d", self.outstanding_orders.get_theorical_position())
        self.logger.info("real position: %d", self.outstanding_orders.get_real_position())
        if instrument == 0 and sequence_number == 1:
            self.start = self.event_loop.time()
        
        if instrument == Instrument.FUTURE and sequence_number > 1:
            
            #update the order book
            #self.order_book_Future.update(ask_prices, ask_volumes, bid_prices, bid_volumes)               
                   
            self.now = (self.event_loop.time()-self.start)*SPEED
            T_minus_t = (END_TIME - self.now) / END_TIME
            self.logger.info("volatility: %f, now: %f, T_minus_t: %f", self.sigma, self.now, T_minus_t)
            position = self.outstanding_orders.get_real_position()
            
            # avellaneda-stoikov
            mid_price = (ask_prices[0] + bid_prices[0]) / 2
            gamma_sigma2_T_minus_t = self.gamma * self.sigma**2 * T_minus_t
            reservation_price = mid_price - position * gamma_sigma2_T_minus_t * TICK_SIZE_IN_CENTS
            optimal_spread = gamma_sigma2_T_minus_t + TICK_SIZE_IN_CENTS * 2 * np.log(1 + self.gamma/self.kappa)/self.gamma
            self.logger.info("mid price:%d, reservation price:%d, optimal spread:%d", mid_price, reservation_price, optimal_spread)
            
            # compute new bid and ask prices
            #new_bid_price = math.floor((reservation_price - optimal_spread / 2) / TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
            #new_ask_price = math.ceil((reservation_price + optimal_spread / 2) / TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
            max_bid = int((mid_price - TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
            bid_price = int((reservation_price - optimal_spread / 2) // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
            new_bid_price = np.minimum(bid_price, max_bid)
            lot_size_bid = np.minimum(LOT_SIZE, POSITION_LIMIT - self.outstanding_orders.get_real_position())
            
            min_ask = int((mid_price + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
            ask_prices = int((reservation_price + optimal_spread / 2) // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
            new_ask_price = np.maximum(ask_prices, min_ask)
            lot_size_ask = np.minimum(LOT_SIZE, self.outstanding_orders.get_real_position() + POSITION_LIMIT)
                        
            if self.outstanding_orders.can_we_bid():
                bid_id = next(self.order_ids)                                
                self.outstanding_orders.add_order(self.now, bid_id, Side.BUY, new_bid_price, lot_size_bid)
                self.send_insert_order(bid_id, Side.BUY, new_bid_price, lot_size_bid, Lifespan.GOOD_FOR_DAY)
                
                self.bids.add(bid_id)
                
                self.logger.info("new bid %d with price %d, volume: %d", bid_id, new_bid_price, lot_size_bid)
            
            if self.outstanding_orders.can_we_ask():
                ask_id = next(self.order_ids)               
                
                self.outstanding_orders.add_order(self.now, ask_id, Side.SELL, new_ask_price, lot_size_ask)
                self.send_insert_order(ask_id, Side.SELL, new_ask_price, lot_size_ask, Lifespan.GOOD_FOR_DAY)
                
                self.asks.add(ask_id)
                
                self.logger.info("new ask %d with price %d, volume: %d", ask_id, new_ask_price, lot_size_ask)
        
        elif instrument == Instrument.ETF and sequence_number > 1:
            #self.order_book_ETF.update(ask_prices, ask_volumes, bid_prices, bid_volumes)
            pass
        self.logger.info("-------------------")


    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("on_order_filled_message")
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        self.send_cancel_order(client_order_id)
        
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume)

        
        self.logger.info(str(self.outstanding_orders))
        self.outstanding_orders.is_filled(client_order_id,volume)
        self.logger.info("position: %d", self.position)
        self.logger.info("theorical class: %d", self.outstanding_orders.get_theorical_position())
        self.logger.info("real class: %d", self.outstanding_orders.get_real_position())
        self.logger.info("-------------------")
        
    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("on_order_status_message")
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        self.logger.info("position: %d", self.position)
        self.logger.info("theorical position: %d", self.outstanding_orders.get_theorical_position())
        self.logger.info("real position: %d", self.outstanding_orders.get_real_position())
        self.outstanding_orders.update_order(client_order_id, fill_volume, remaining_volume)
        self.logger.info(str(self.outstanding_orders))
        self.logger.info("position: %d", self.position)
        self.logger.info("theorical position: %d", self.outstanding_orders.get_theorical_position())
        self.logger.info("real position: %d", self.outstanding_orders.get_real_position())
        self.logger.info("-------------------")

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("on_trade_ticks_message")
        self.logger.info(f'instrument: {instrument}, sequence number: {sequence_number}, mid price: {(ask_prices[0] + bid_prices[0]) / 2} \n bid prices: {bid_prices} \n bid volumes: {bid_volumes}  \n ask prices: {ask_prices} \n ask volumes: {ask_volumes}')
        
        if instrument == Instrument.ETF:      
            to_cancel = self.outstanding_orders.to_cancel(bid_prices, ask_prices)
            for order_id in to_cancel:
                self.send_cancel_order(order_id)
        elif instrument == Instrument.FUTURE:
            mid_price = (ask_prices[0] + bid_prices[0]) / 2
            self.volatility_indicator.add_sample(mid_price)
            self.sigma = self.volatility_indicator.current_vol()  

            
                



### Custom functions

class OrderBookCustom:
    """A custom order book class."""
    
    def __init__(self):
        self.__ask_prices: List[int] = []
        self.__ask_volumes: List[int] = []
        self.__bid_prices: List[int] = []
        self.__bid_volumes: List[int] = []
        
        self.__last_traded_price: int = 0
        self.__mid_price: int = 0

        
    def __str__(self):
        """Return a string representation of this order book."""
        return f'last traded price: {self.__last_traded_price}, mid price: {self.__mid_price} \n bid prices: {self.__bid_prices} \n bid volumes: {self.__bid_volumes} \n  \n ask prices: {self.__ask_prices} \n ask volumes: {self.__ask_volumes}'
    
    def update(self, ask_prices: List[int], ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]):
        """Update the order book with the latest information
        It also computes the last traded price by comparing the current book with the previous one and give the volume traded
        """
        
        # determine the last traded price knowing that bid prices are sorted in descending order and ask prices in ascending order
        # it determines if a price disappear or a volume change
        for i in range(len(self.__ask_prices)):
            if self.__ask_prices[i] not in ask_prices:
                self.__last_traded_price = self.__ask_prices[i]
                break
            if self.__ask_volumes[i] != ask_volumes[ask_prices.index(self.__ask_prices[i])]:
                self.__last_traded_price = self.__ask_prices[i]
                break
        for i in range(len(self.__bid_prices)):
            if self.__bid_prices[i] not in bid_prices:
                self.__last_traded_price = self.__bid_prices[i]
                break
            if self.__bid_volumes[i] != bid_volumes[bid_prices.index(self.__bid_prices[i])]:
                self.__last_traded_price = self.__bid_prices[i]
                break
        
        self.__mid_price = (ask_prices[0] + bid_prices[0]) / 2
        
        # update the order book
        self.__ask_prices = ask_prices
        self.__ask_volumes = ask_volumes
        self.__bid_prices = bid_prices
        self.__bid_volumes = bid_volumes
    
    def last_traded_price(self):
        """Return the last traded price."""
        return self.__last_traded_price
    
    def mid_price(self):
        """Return the mid price."""
        return self.__mid_price


class VolatilityIndicator():
    """ Use a custom Ring Buffer to calculate the volatility of the last N ticks. """
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_VOLATILITY_SIZE)
        self.volatility = 0.25
        self.min_size = MIN_TO_COMPUTE_VOLATILITY

    def add_sample(self, sample: float):
        if sample != 0: self.buffer.append(sample)
            
    def calculation(self) -> None:
        """ Calculate the volatility"""
        if len(self.buffer) > self.min_size: 
            self.volatility = np.sqrt(np.sum(np.square(np.log(np.array(self.buffer)[:-1]) - np.log(np.array(self.buffer)[1:])))/(len(self.buffer)-1))
    
    def current_vol(self) -> float:
        self.calculation()
        return self.volatility
    
        
class OutsandingOrders():
    """Keep track of my outstanding orders."""    
    def __init__(self) -> None:
        self.theorical_position = 0 # theorical curent position should not exceed 100 or -100
        self.real_position = 0 # real position
        self.active_orders : int = 0 # number of active orders should no be more than 10
        self.orders =  collections.defaultdict(int) # dictionary of all orders
    
    def add_order(self, time: float, order_id: int, side: int, price: int, volume: int) -> None:
        """Add an order to the dictionary of orders.
        if side is 1 it's a bid and if it's 0 it's an ask."""
        if side == Side.BID:
            self.theorical_position += volume
        else:
            self.theorical_position -= volume
        self.orders[order_id] = [time, side, price, volume]
        self.active_orders += 1
    
    def is_filled(self, order_id: int, volume: int) -> bool:
        """Change the postiton when an order is filled."""
        if self.orders[order_id][1] == Side.BID:
            self.real_position += volume
        else:
            self.real_position -= volume            

    def update_order(self, client_order_id: int, fill_volume: int, remaining_volume: int) -> None:
        """Update the volume of an order.
         The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded
        """
        if fill_volume == 0 and remaining_volume == 0: # the order was rejected
            self.active_orders -= 1
            if self.orders[client_order_id][1] == Side.BID:
                self.theorical_position -= self.orders[client_order_id][3]
            else:
                self.theorical_position += self.orders[client_order_id][3]
            del self.orders[client_order_id]
        elif fill_volume == 0: # we already add this order when it was created
            pass 
        elif remaining_volume == 0: # we add to delete this order
            self.active_orders -= 1
            del self.orders[client_order_id]
        else: # we update the volume of the position
            self.orders[client_order_id][3] = remaining_volume
    
    def can_we_bid(self):
        """Return True if we can bid, False otherwise and return the volume we can bid."""
        if self.active_orders < 4 and self.real_position < 100:
            return True
        else:
            return False
        
    def can_we_ask(self):
        """Return True if we can ask, False otherwise and return the volume we can ask."""
        if self.active_orders < 4 and self.real_position > -100:
            return True
        else:
            return False
        
    def get_theorical_position(self):
        """Return the current position."""
        return self.theorical_position
    
    def get_real_position(self):
        """Return the real position."""
        return self.real_position
    
    def get_active_orders(self):
        """Return the number of active orders."""
        return self.active_orders
    
    def to_cancel(self, bid_prices, ask_prices) -> List[int]:
        """ If order are not in the top 5 bids or top 5 asks we return a list of order ids to cancel"""
        to_cancel = []
        for order_id in self.orders:
            if (self.orders[order_id][2] not in bid_prices) or (self.orders[order_id][2] not in ask_prices):
                to_cancel.append(order_id)
        return to_cancel
    
    def __str__(self) -> str:
        """Return a string representation of the outstanding orders."""
        return f'theorical positions: {self.theorical_position}, real positions: {self.theorical_position} \n active orders: {self.active_orders} \n orders: {self.orders} \n'
        
                 
        
        