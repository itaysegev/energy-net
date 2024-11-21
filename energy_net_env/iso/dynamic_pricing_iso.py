# iso/dynamic_pricing_iso.py

from typing import Callable, Dict
from energy_net_env.iso.iso_base import ISOBase


class DynamicPricingISO(ISOBase):
    """
    ISO implementation that adjusts prices dynamically based on real-time factors.
    """

    def __init__(
        self,
        base_price: float = 50.0,
        demand_factor: float = 1.0,
        supply_factor: float = 1.0,
        elasticity: float = 0.5
    ):
        """
        Initializes the DynamicPricingISO with parameters influencing dynamic pricing.

        Args:
            base_price (float): Base price ($/MWh).
            demand_factor (float): Multiplier for demand influence.
            supply_factor (float): Multiplier for supply influence.
            elasticity (float): Elasticity coefficient to determine price sensitivity.
        """
        self.base_price = base_price
        self.demand_factor = demand_factor
        self.supply_factor = supply_factor
        self.elasticity = elasticity

    def reset(self) -> None:
        """
        Resets the ISO's internal state.
        For DynamicPricingISO, no internal state is maintained.
        """
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float, float], float]:
        """
        Returns a pricing function that adjusts prices based on demand and supply.

        Args:
            observation (Dict): Should contain 'demand' and 'supply' values.

        Returns:
            Callable[[float, float], float]: Pricing function calculating reward.
        """
        demand = observation.get('demand', 1.0)  # Example: demand factor
        supply = observation.get('supply', 1.0)  # Example: supply factor

        # Dynamic price calculation using elasticity
        price_buy = self.base_price * (1 + self.elasticity * (demand - supply))
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
