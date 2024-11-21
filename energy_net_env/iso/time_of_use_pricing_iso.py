# iso/time_of_use_pricing_iso.py

from typing import Callable, Dict
from energy_net_env.iso.iso_base import ISOBase


class TimeOfUsePricingISO(ISOBase):
    """
    ISO implementation that sets prices based on time-of-use (TOU) periods.
    """

    def __init__(
        self,
        peak_hours: list,
        off_peak_hours: list,
        peak_price: float = 60.0,
        off_peak_price: float = 30.0
    ):
        """
        Initializes the TimeOfUsePricingISO with specified TOU periods and prices.

        Args:
            peak_hours (list): List of hours (0-23) considered as peak.
            off_peak_hours (list): List of hours (0-23) considered as off-peak.
            peak_price (float): Price during peak hours ($/MWh).
            off_peak_price (float): Price during off-peak hours ($/MWh).
        """
        self.peak_hours = peak_hours
        self.off_peak_hours = off_peak_hours
        self.peak_price = peak_price
        self.off_peak_price = off_peak_price

    def reset(self) -> None:
        """
        Resets the ISO's internal state.
        For TimeOfUsePricingISO, no internal state is maintained.
        """
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float, float], float]:
        """
        Returns a pricing function based on the current hour.

        Args:
            observation (Dict): Should contain 'time' as a fraction of the day (0 to 1).

        Returns:
            Callable[[float, float], float]: Pricing function calculating reward.
        """
        current_time_fraction = observation.get('time', 0.0)
        current_hour = int(current_time_fraction * 24) % 24  # Convert fraction to hour

        if current_hour in self.peak_hours:
            price_buy = self.peak_price
            price_sell = self.peak_price * 0.9  # Example: sell price is 90% of buy price
        elif current_hour in self.off_peak_hours:
            price_buy = self.off_peak_price
            price_sell = self.off_peak_price * 0.9
        else:
            # Default pricing for hours not classified as peak or off-peak
            price_buy = (self.peak_price + self.off_peak_price) / 2
            price_sell = price_buy * 0.9

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
