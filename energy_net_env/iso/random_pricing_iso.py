# iso/random_pricing_iso.py

import random
from typing import Callable, Dict
from energy_net_env.iso.iso_base import ISOBase


class RandomPricingISO(ISOBase):
    """
    ISO implementation that generates random prices within a specified range.
    """

    def __init__(self, min_price: float = 40.0, max_price: float = 60.0):
        """
        Initializes the RandomPricingISO with minimum and maximum price bounds.

        Args:
            min_price (float): Minimum possible price ($/MWh).
            max_price (float): Maximum possible price ($/MWh).
        """
        self.min_price = min_price
        self.max_price = max_price

    def reset(self) -> None:
        """
        Resets the ISO's internal state.
        For RandomPricingISO, no internal state is maintained.
        """
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float, float], float]:
        """
        Returns a pricing function with random buy and sell prices for each call.

        Args:
            observation (Dict): Not utilized in this pricing strategy.

        Returns:
            Callable[[float, float], float]: Pricing function calculating reward.
        """
        # Generate random prices within the specified range
        price_buy = random.uniform(self.min_price, self.max_price)
        price_sell = price_buy * random.uniform(0.8, 0.95)  # Sell price between 80% and 95% of buy price

        def pricing(buy: float, sell: float) -> float:
            """
            Calculates reward based on buy and sell amounts.

            Args:
                buy (float): Amount of energy bought (MWh).
                sell (float): Amount of energy sold (MWh).

            Returns:
                float: Calculated reward.
            """
            return (buy * price_buy) - (sell * price_sell)

        return pricing
