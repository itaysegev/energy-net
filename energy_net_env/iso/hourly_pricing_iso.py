# iso/hourly_pricing_iso.py

from typing import Callable, Dict
from energy_net_env.iso.iso_base import ISOBase


class HourlyPricingISO(ISOBase):
    """
    ISO implementation that sets prices based on predefined hourly rates.
    """

    def __init__(self, hourly_rates: Dict[int, float]):
        """
        Initializes the HourlyPricingISO with a dictionary of hourly rates.

        Args:
            hourly_rates (Dict[int, float]): Mapping from hour of the day (0-23) to price ($/MWh).
        """
        self.hourly_rates = hourly_rates

    def reset(self) -> None:
        """
        Resets the ISO's internal state.
        For HourlyPricingISO, no internal state is maintained.
        """
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float, float], float]:
        """
        Returns a pricing function based on the current time in the observation.

        Args:
            observation (Dict): Should contain 'time' as a fraction of the day (0 to 1).

        Returns:
            Callable[[float, float], float]: Pricing function calculating reward.
        """
        current_time_fraction = observation.get('time', 0.0)
        current_hour = int(current_time_fraction * 24) % 24  # Convert fraction to hour

        # Fetch the price for the current hour
        price_buy = self.hourly_rates.get(current_hour, 50.0)  # Default buy price
        price_sell = price_buy * 0.9  # Example: sell price is 90% of buy price

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
