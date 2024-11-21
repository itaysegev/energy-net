# iso/quadratic_pricing_iso.py

from typing import Callable, Dict
from energy_net_env.iso.iso_base import ISOBase


class QuadraticPricingISO(ISOBase):
    """
    ISO implementation that uses a quadratic function to determine prices based on demand or other factors.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, c: float = 50.0):
        """
        Initializes the QuadraticPricingISO with quadratic coefficients.

        Args:
            a (float): Quadratic coefficient.
            b (float): Linear coefficient.
            c (float): Constant term (base price).

        Raises:
            TypeError: If any of the parameters are not floats or integers.
        """
        # Type checking for 'a'
        if not isinstance(a, (float, int)):
            raise TypeError(f"a must be a float, got {type(a).__name__}")
        # Type checking for 'b'
        if not isinstance(b, (float, int)):
            raise TypeError(f"b must be a float, got {type(b).__name__}")
        # Type checking for 'c'
        if not isinstance(c, (float, int)):
            raise TypeError(f"c must be a float, got {type(c).__name__}")

        # Convert to float in case integers are provided
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def reset(self) -> None:
        """
        Resets the ISO's internal state.
        For QuadraticPricingISO, no internal state is maintained.
        """
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float, float], float]:
        """
        Returns a pricing function based on a quadratic relation to a variable (e.g., demand).

        Args:
            observation (Dict): Should contain a variable influencing the price (e.g., 'demand').

        Returns:
            Callable[[float, float], float]: Pricing function calculating reward.
        """
        # Example: Price influenced by demand
        demand = observation.get('demand', 1.0)  # Default demand factor

        # Quadratic pricing formula: price = a * demand^2 + b * demand + c
        price_buy = self.a * (demand ** 2) + self.b * demand + self.c
        price_sell = price_buy * 0.85  # Example: sell price is 85% of buy price

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
