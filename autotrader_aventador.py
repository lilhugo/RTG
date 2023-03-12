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

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side

from ready_trader_go import timer, account, order_book


LOT_SIZE = 10
POSITION_LIMIT = 100

SPEED = 1
TICK_INTERVAL = 0.25
END_TIME = 900

TICK_SIZE = 1.00
TICK_SIZE_IN_CENTS = 100


MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS

MAKER_FEES =-0.0001
TAKER_FEES = 0.0002
ETF_CLAMP = 0.002

# about volatility parameters
BUFFER_SIZE = 200
MIN_SIZE = 10

# Parameters to improve the performance of the strategy
GAMMA = 0.015 # risk aversion
KAPPA = 0.3 # order book liquidity parameter
SKEW = 0
TIME_TO_CANCEL = 3 # time to cancel the order if it is not filled
MIN_SPREAD = 0.3 # as a percentage of the mid price


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
        
        # custom init
        self.vol_is_calculated = False
        self.is_started = False
        self.can_trade = False
        
        self.volatility_indicator = Volatility(BUFFER_SIZE,MIN_SIZE)
        self.gamma = GAMMA
        self.kappa = KAPPA
        self.skew = SKEW
        self.min_spread = MIN_SPREAD
        
        self.start = self.event_loop.time()
        self.now = self.start
        
        self.outsanding_orders = Outstanding_orders()
        '''self.my_account = account.CompetitorAccount(TICK_SIZE_IN_CENTS, ETF_CLAMP)'''
        
        self.future_price = 0
        self.etf_price = 0
        

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
        '''self.my_account.transact(Instrument.FUTURE, Side.BUY if client_order_id in self.bids else Side.SELL, price, volume, TAKER_FEES)
        self.my_account.update(self.future_price, self.mid_price)
        self.logger.info("my account: %d", self.my_account.profit_or_loss)'''

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        
        if instrument == Instrument.FUTURE:
            
            # remove too old orders
            list_to_cancel = self.outsanding_orders.get_orders_to_cancel(self.now)
            self.logger.info("list_to_cancel: %s", list_to_cancel)
            for order_id in list_to_cancel:
                self.send_cancel_order(order_id)
                self.outsanding_orders.remove_order(order_id)
            self.can_trade = self.outsanding_orders.can_trade()
            
            # compute max bid and min ask
            mid_price = (ask_prices[0] + bid_prices[0]) / 2
            min_spread = mid_price * self.min_spread / 100
            max_limit_bid = int((mid_price - min_spread / 2) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS)
            min_limit_ask = int((mid_price + min_spread / 2) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS)
            
            if self.vol_is_calculated:
                # Avellaneda-Stoikov                           
                vol = self.volatility_indicator.current_vol() 
                self.now = (self.event_loop.time()-self.start)*SPEED
                T_minus_t = (END_TIME - self.now) / END_TIME
                self.logger.info("vol: %f, now: %f, T_minus_t: %f", vol, self.now, T_minus_t)

                reservation_price = mid_price - self.position * TICK_SIZE_IN_CENTS * self.gamma * vol**2 * T_minus_t
                optimal_spread = self.gamma * vol**2 * T_minus_t + 2 * np.log(1 + self.gamma/self.kappa)/self.gamma
                optimal_spread *= TICK_SIZE_IN_CENTS
                        
                new_bid_price = int((reservation_price - optimal_spread / 2) * ( 1 + self.skew) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS)
                new_ask_price = int((reservation_price + optimal_spread / 2) * ( 1 + self.skew) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS)
            
            else :
                new_bid_price = max_limit_bid
                new_ask_price = min_limit_ask     
                                   
            
            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                self.send_cancel_order(self.bid_id)
                self.bid_id = 0
                #self.logger.info("cancel bid_id: %d", self.bid_id)
            if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
                self.send_cancel_order(self.ask_id)
                self.ask_id = 0
                #self.logger.info("cancel ask_id: %d", self.ask_id)
                
            if self.bid_id == 0 and new_bid_price != 0 and self.position < POSITION_LIMIT and self.can_trade:
                self.bid_id = next(self.order_ids)
                self.bid_price = np.minimum(new_bid_price, max_limit_bid)
                volume = np.minimum(LOT_SIZE, POSITION_LIMIT - self.position)
                self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, volume, Lifespan.GOOD_FOR_DAY)
                
                self.bids.add(self.bid_id)
                self.logger.info("bid_id: %d, bid_price: %d, volume: %d, position: %d", self.bid_id, self.bid_price, volume, self.position)
                self.outsanding_orders.add_order(self.now, self.bid_id, Side.BUY, new_bid_price, volume)
                #self.logger.info(str(self.outsanding_orders))
                
            if self.ask_id == 0 and new_ask_price != 0 and self.position > -POSITION_LIMIT and self.can_trade:
                self.ask_id = next(self.order_ids)
                self.ask_price = np.maximum(new_ask_price, min_limit_ask)
                volume = np.minimum(LOT_SIZE, POSITION_LIMIT + self.position)
                self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, volume, Lifespan.GOOD_FOR_DAY)
                self.asks.add(self.ask_id)
                self.logger.info("bid_id: %d, bid_price: %d, volume: %d, position: %d", self.ask_id, self.ask_price, volume, self.position)
                self.outsanding_orders.add_order(self.now, self.ask_id, Side.SELL, new_ask_price, volume)
                #self.logger.info(str(self.outsanding_orders))
               

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        '''self.my_account.transact(Instrument.ETF, Side.BUY if client_order_id in self.bids else Side.SELL, price, volume, TAKER_FEES)
        self.my_account.update(self.future_price, self.mid_price)
        self.logger.info("my account: %d", self.my_account.profit_or_loss)'''

        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
            self.outsanding_orders.update_order(client_order_id, volume)
            
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume)
            self.outsanding_orders.update_order(client_order_id, volume)
        

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
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.outsanding_orders.remove_order(client_order_id)
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.outsanding_orders.remove_order(client_order_id)
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """

        self.now = (self.event_loop.time()-self.start)*SPEED            
        
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)     
        
        self.logger.info("5 best bid prices: %s / 5 best ask prices: %s", bid_prices, ask_prices)
        
        self.logger.info(str(self.outsanding_orders))
        
        # add the mid price to the ring buffer and calculate the volatility if we have enough data
        if instrument == Instrument.FUTURE:
            mid_price = (ask_prices[0] + bid_prices[0]) / 2
            self.future_price = mid_price
            self.volatility_indicator.add_sample(mid_price)
            if not self.vol_is_calculated:
                self.vol_is_calculated = self.volatility_indicator.is_more_than_min_size()  
                
        '''elif instrument == Instrument.ETF:   
            self.mid_price = (ask_prices[0] + bid_prices[0]) / 2'''
        

#######  Custom functions/classes #######

class Volatility():
    """ Use a custom Ring Buffer to calculate the volatility of the last N ticks. """
    def __init__(self, size: int, MIN_SIZE):
        self.size = size
        self.buffer = collections.deque(maxlen=size)
        self.vol = 0   
        self.min_size = MIN_SIZE

    def is_more_than_min_size(self) -> bool:
        return len(self.buffer) > self.min_size
    
    def is_full(self) -> bool:
        return len(self.buffer) == self.size
    
    def add_sample(self, sample: float):
        self.buffer.append(sample)
            
    def partial_calculation(self) -> None:
        """ Calculate the volatility of the available ticks. """
        if len(self.buffer) > self.min_size: 
            self.vol = np.sqrt(np.sum(np.square(np.log(np.array(self.buffer)[:-1]) - np.log(np.array(self.buffer)[1:])))/(len(self.buffer)-1))
    
    def calculation(self) -> None:
        """ Calculate the volatility of the last N ticks. """
        self.vol = np.sqrt(np.sum(np.square(np.log(np.array(self.buffer)[:-1]) - np.log(np.array(self.buffer)[1:])))/(len(self.buffer)-1))
    
    def current_vol(self) -> float:
        if self.is_full():
            self.calculation()
        else:
            self.partial_calculation()
        return self.vol
    
    

class Outstanding_orders():
    """ Keep track of the outstanding orders. """
    def __init__(self) -> None:
        self.positions = 0 # curent position should not exceed 100 or -100
        self.active_orders : int = 0 # number of active orders should no be more than 10
        self.orders =  collections.defaultdict(int) # dictionary of all orders
        self.time_to_cancel = TIME_TO_CANCEL # time to cancel an order in seconds
        
    def can_trade(self) -> bool:
        return self.active_orders < 10 and abs(self.positions) < 100

    def get_orders_to_cancel(self, time) -> List[int]:
        """ Return a list of orders id to cancel.
        We should cancel if the order is older than TIME_TO_CANCEL seconds. """
        
        to_cancel = []
        for order_id in self.orders:
            if time - self.orders[order_id][0] > self.time_to_cancel:
                to_cancel.append(order_id)
        return to_cancel       
    
    def add_order(self, time: float, order_id: int, side: Side, price: int, volume: int) -> None:
        self.orders[order_id] = [time, side, price, volume]
        self.active_orders += 1
        if side == Side.BID:
            self.positions += volume
        else:
            self.positions -= volume
    
    def update_order(self, order_id: int, filled_volume: int) -> None:
        self.orders[order_id][3] -= filled_volume
        if self.orders[order_id][1] == Side.BID:
            self.positions -= filled_volume
        else:
            self.positions += filled_volume
    
    def remove_order(self, order_id: int) -> None:
        if self.orders[order_id][1] == Side.BID:
            self.positions -= self.orders[order_id][3]
        else:
            self.positions += self.orders[order_id][3]
        del self.orders[order_id]
        self.active_orders -= 1
    
    def current_positions(self) -> int:
        return self.positions
    
    def number_of_active_orders(self) -> int:
        return self.active_orders
    
    def __str__(self) -> str:
        return f"Outstanding_orders(positions={self.positions}, active_orders={self.active_orders}, orders={self.orders})"
