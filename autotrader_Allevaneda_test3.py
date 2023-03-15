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

LOT_SIZE = 10
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
END_TIME = 900
TICK_INTERVAL = 0.25
SPEED = 1
BUFFER_SIZE = 100
MIN_SIZE = 10

class AutoTrader(BaseAutoTrader):
    """Auto trader that implements the Allevaneda and Stoikov (2007) algorithm.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = self.filled_position = 0

        # Initialize the variables for the time control
        self.start = 0
        self.time = 0
        self.now = 0

        # Initialize the variables for the volatility and bid/ask control 
        self.mid_prices_future = self.returns_future = self.bid_prices = self.ask_prices =  np.array([])
        self.volatility_indicator = Volatility(BUFFER_SIZE)

        # Position limit
        self.position_limit = 100

        # Allevaneda and Stoikov (2007) parameters
        self.gamma = 0.01
        self.min_spread_percent = 0.02

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
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,sequence_number)
        self.logger.info(f"current position is {self.position}")
        if instrument == Instrument.FUTURE:
            # Calculate the time now
            if self.start == 0:
                self.start = self.event_loop.time()
                self.now = TICK_INTERVAL
        
            # Update the time
            else:
                self.time = self.event_loop.time()
                self.now += (self.time - self.start) * SPEED
                self.start = self.time

            self.logger.info(f"Time now: {self.now}")

            # Calculate the mid price for the Future
            mid_price_future = (ask_prices[0] + bid_prices[0]) // 2

            # Add the mid price to the array
            self.mid_prices_future = np.append(self.mid_prices_future,mid_price_future)

            # Calculate the return for the Future if there are more than 1 mid prices
            if self.mid_prices_future.shape[0] > 1:
                return_future = np.log(self.mid_prices_future[-1]/self.mid_prices_future[-2])
                self.volatility_indicator.buffer.append(return_future)
                self.logger.info(f"mid price for the Future: {self.mid_prices_future[-1]}")
            
            # Calculate the volatility for the Future if there are more than 20 returns
            if len(self.volatility_indicator.buffer) >= MIN_SIZE:
                # Vol type I
                # vol_future = np.std(self.returns_future) * np.sqrt(END_TIME * 4)
                
                # Vol type II
                vol_future = self.volatility_indicator.current_vol() 
                self.logger.info(f"volatility indicator: {vol_future}")

                # Calculate the reservation price for the Future
                T_minus_t = (END_TIME - np.minimum(self.now,END_TIME)) / END_TIME
                reservation_price = mid_price_future - self.position * self.gamma * (vol_future ** 2) * T_minus_t
                self.logger.info(f"reservation price for the Future: {reservation_price}")
                
                # Optimal spread given by the theoretical model
                optimal_spread = self.gamma * (vol_future ** 2) * T_minus_t + 2 * np.log(1 + self.gamma / 0.3) / self.gamma
                min_spread = mid_price_future / 100 * self.min_spread_percent

                # Calculate the bid price given by the theoretical model
                max_limit_bid = mid_price_future - min_spread / 2
                new_bid_price = min(int(max_limit_bid // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS),int((reservation_price - optimal_spread * TICK_SIZE_IN_CENTS  / 2) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS))
                self.bid_prices = np.append(self.bid_prices,new_bid_price)
                # self.logger.info(f"max_limit_bid: {max_limit_bid}")
                # self.logger.info(f"new bid price: {new_bid_price}")

                # Calculate the ask price given by the theoretical model 
                min_limit_ask = mid_price_future + min_spread / 2         
                new_ask_price = max(int(min_limit_ask // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS),int((reservation_price + optimal_spread * TICK_SIZE_IN_CENTS / 2) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS))  
                self.ask_prices = np.append(self.ask_prices,new_ask_price)
                # self.logger.info(f"min_limit_ask: {min_limit_ask}")
                # self.logger.info(f"new ask price: {new_ask_price}")
                
                if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                    self.send_cancel_order(self.bid_id)
                    self.logger.info(f"bid order {self.bid_id} cancelled")

                if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
                    self.send_cancel_order(self.ask_id)
                    self.logger.info(f"ask order {self.ask_id} cancelled")


                # Condition to send a new bid order
                if self.bid_id == 0 and new_bid_price != 0 and self.position < self.position_limit:
                    self.bid_id = next(self.order_ids)
                    self.bid_price = new_bid_price
                    
                    if self.position == -self.position_limit:
                        self.send_insert_order(self.bid_id, Side.BUY, new_bid_price,self.position_limit+50, Lifespan.GOOD_FOR_DAY)
                        self.logger.info(f"bid order {self.bid_id} inserted at price {new_bid_price} for {self.position_limit+50} lots")
                    else:
                        self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, np.minimum(LOT_SIZE*5,self.position_limit - self.position), Lifespan.GOOD_FOR_DAY)
                        self.logger.info(f"bid order {self.bid_id} inserted at price {new_bid_price} for {np.minimum(LOT_SIZE*5,self.position_limit - self.position)} lots")
                    self.bids.add(self.bid_id)

                # Condition to send a new ask order
                if self.ask_id == 0 and new_ask_price != 0 and self.position > -self.position_limit:
                    self.ask_id = next(self.order_ids)
                    self.ask_price = new_ask_price
                    
                    if self.position == self.position_limit:
                        self.send_insert_order(self.ask_id, Side.SELL, new_ask_price,self.position_limit+50,Lifespan.GOOD_FOR_DAY)
                        self.logger.info(f"ask order {self.ask_id} inserted at price {new_ask_price} for {self.position_limit+50} lots")

                    else:
                        self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, np.minimum(LOT_SIZE*5,self.position + self.position_limit), Lifespan.GOOD_FOR_DAY)
                        self.logger.info(f"ask order {self.ask_id} inserted at price {new_ask_price} for {np.minimum(LOT_SIZE*5,self.position + self.position_limit)} lots")
                    self.asks.add(self.ask_id)

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,price, volume)

        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK,MIN_BID_NEAREST_TICK,volume)
            self.send_cancel_order(client_order_id)
            self.logger.info(f"ask order {client_order_id} cancelled")
            

        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID,MAX_ASK_NEAREST_TICK,volume)
            self.send_cancel_order(client_order_id)
            self.logger.info(f"ask order {client_order_id} cancelled")

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",client_order_id, fill_volume, remaining_volume, fees)
        
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.logger.info(f"cancel order {client_order_id} received (bid)")
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.logger.info(f"cancel order {client_order_id} received (ask)")
                self.ask_id = 0
            
            new_bid_price, new_ask_price = int(self.bid_prices[-1]), int(self.ask_prices[-1])
            # Condition to send a new bid order
            if self.bid_id == 0 and new_bid_price != 0 and self.position < self.position_limit:
                self.bid_id = next(self.order_ids)
                self.bid_price = new_bid_price
                
                if self.position == -self.position_limit:
                    self.send_insert_order(self.bid_id, Side.BUY, new_bid_price,self.position_limit+50, Lifespan.GOOD_FOR_DAY)
                    self.logger.info(f"bid order {self.bid_id} inserted at price {new_bid_price} for {self.position_limit+50} lots")
                else:
                    self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, np.minimum(LOT_SIZE*5,self.position_limit - self.position), Lifespan.GOOD_FOR_DAY)
                    self.logger.info(f"bid order {self.bid_id} inserted at price {new_bid_price} for {np.minimum(LOT_SIZE*5,self.position_limit - self.position)} lots")
                self.bids.add(self.bid_id)

            # Condition to send a new ask order
            if self.ask_id == 0 and new_ask_price != 0 and self.position > -self.position_limit:
                self.ask_id = next(self.order_ids)
                self.ask_price = new_ask_price
                
                if self.position == self.position_limit:
                    self.send_insert_order(self.ask_id, Side.SELL, new_ask_price,self.position_limit+50,Lifespan.GOOD_FOR_DAY)
                    self.logger.info(f"ask order {self.ask_id} inserted at price {new_ask_price} for {self.position_limit+50} lots")

                else:
                    self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, np.minimum(LOT_SIZE*5,self.position + self.position_limit), Lifespan.GOOD_FOR_DAY)
                    self.logger.info(f"ask order {self.ask_id} inserted at price {new_ask_price} for {np.minimum(LOT_SIZE*5,self.position + self.position_limit)} lots")
                self.asks.add(self.ask_id)

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
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)




#######  Custom functions/classes #######

class Volatility():
    """ Use a custom Ring Buffer to calculate the volatility of the last N ticks. """
    def __init__(self, size: int):
        self.size = size
        self.buffer = collections.deque(maxlen=size)
        self.vol = 0   
        
    def calculation(self) -> None:
        """ Calculate the volatility of the last N ticks. """
        self.vol = np.sqrt(252 * 24 * 3600 * 4 * np.sum(np.square(self.buffer))/(len(self.buffer)-1))
    
    def current_vol(self) -> float:
        self.calculation()
        return self.vol