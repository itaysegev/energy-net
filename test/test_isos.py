# tests/test_isos.py

import unittest

from energy_net_env.iso.hourly_pricing_iso import HourlyPricingISO
from energy_net_env.iso.random_pricing_iso import RandomPricingISO
from energy_net_env.iso.quadratic_pricing_iso import QuadraticPricingISO
from energy_net_env.iso.time_of_use_pricing_iso import TimeOfUsePricingISO
from energy_net_env.iso.dynamic_pricing_iso import DynamicPricingISO


class TestHourlyPricingISO(unittest.TestCase):
    def setUp(self):
        self.hourly_rates = {hour: 50.0 for hour in range(24)}
        self.iso = HourlyPricingISO(hourly_rates=self.hourly_rates)

    def test_pricing_function_peak_hour(self):
        observation = {'time': 0.5}  # 12:00 PM
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.hourly_rates[12]
        expected_price_sell = expected_price_buy * 0.9
        reward = pricing_func(10, 5)
        expected_reward = (10 * expected_price_buy) - (5 * expected_price_sell)
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_default_price(self):
        observation = {'time': 1.0}  # 24:00 or 0:00 AM
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.hourly_rates[0]
        expected_price_sell = expected_price_buy * 0.9
        reward = pricing_func(0, 0)
        expected_reward = 0.0
        self.assertEqual(reward, expected_reward)


class TestRandomPricingISO(unittest.TestCase):
    def setUp(self):
        self.min_price = 40.0
        self.max_price = 60.0
        self.iso = RandomPricingISO(min_price=self.min_price, max_price=self.max_price)

    def test_pricing_function_range(self):
        observation = {}
        pricing_func = self.iso.get_pricing_function(observation)
        # Since prices are random, we check if they fall within the expected range
        # For unit tests, it's better to mock randomness to have deterministic tests
        # However, for simplicity, we'll perform a basic range check here
        buy = 10
        sell = 5
        reward = pricing_func(buy, sell)
        # Assuming price_buy is between min_price and max_price
        # And price_sell is between 80% and 95% of price_buy
        # So reward should be between:
        min_reward = (buy * self.min_price) - (sell * self.min_price * 0.95)
        max_reward = (buy * self.max_price) - (sell * self.min_price * 0.8)  # Since sell_price could be as low as 80% of buy
        self.assertTrue(min_reward <= reward <= (buy * self.max_price) - (sell * self.min_price * 0.8))


class TestQuadraticPricingISO(unittest.TestCase):
    def setUp(self):
        self.a = 1.0
        self.b = 0.0
        self.c = 50.0
        self.iso = QuadraticPricingISO(a=self.a, b=self.b, c=self.c)

    def test_pricing_function_zero_demand(self):
        observation = {'demand': 0.0}
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.c  # 50.0
        expected_price_sell = expected_price_buy * 0.85
        reward = pricing_func(10, 5)
        expected_reward = (10 * expected_price_buy) - (5 * expected_price_sell)
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_positive_demand(self):
        observation = {'demand': 2.0}
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.a * (2.0 ** 2) + self.b * 2.0 + self.c  # 1*4 + 0*2 + 50 = 54
        expected_price_sell = expected_price_buy * 0.85  # 54 * 0.85 = 45.9
        reward = pricing_func(10, 5)
        expected_reward = (10 * 54) - (5 * 45.9)  # 540 - 229.5 = 310.5
        self.assertAlmostEqual(reward, 310.5, places=2)


class TestTimeOfUsePricingISO(unittest.TestCase):
    def setUp(self):
        self.peak_hours = [17, 18, 19]
        self.off_peak_hours = [0, 1, 2, 3, 4, 5]
        self.peak_price = 60.0
        self.off_peak_price = 30.0
        self.iso = TimeOfUsePricingISO(
            peak_hours=self.peak_hours,
            off_peak_hours=self.off_peak_hours,
            peak_price=self.peak_price,
            off_peak_price=self.off_peak_price
        )

    def test_pricing_function_peak_hour(self):
        observation = {'time': 17 / 24}  # 17:00 or 5:00 PM
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.peak_price
        expected_price_sell = self.peak_price * 0.9
        reward = pricing_func(10, 5)
        expected_reward = (10 * self.peak_price) - (5 * self.peak_price * 0.9)
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_off_peak_hour(self):
        observation = {'time': 4 / 24}  # 4:00 AM
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.off_peak_price
        expected_price_sell = self.off_peak_price * 0.9
        reward = pricing_func(10, 5)
        expected_reward = (10 * self.off_peak_price) - (5 * self.off_peak_price * 0.9)
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_default_hour(self):
        observation = {'time': 12 / 24}  # 12:00 PM
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = (self.peak_price + self.off_peak_price) / 2  # 45.0
        expected_price_sell = expected_price_buy * 0.9  # 40.5
        reward = pricing_func(10, 5)
        expected_reward = (10 * 45.0) - (5 * 40.5)  # 450 - 202.5 = 247.5
        self.assertEqual(reward, 247.5)


class TestDynamicPricingISO(unittest.TestCase):
    def setUp(self):
        self.base_price = 50.0
        self.demand_factor = 1.0
        self.supply_factor = 1.0
        self.elasticity = 0.5
        self.iso = DynamicPricingISO(
            base_price=self.base_price,
            demand_factor=self.demand_factor,
            supply_factor=self.supply_factor,
            elasticity=self.elasticity
        )

    def test_pricing_function_equal_demand_supply(self):
        observation = {'demand': 1.0, 'supply': 1.0}
        pricing_func = self.iso.get_pricing_function(observation)
        # price_buy = 50 * (1 + 0.5*(1 - 1)) = 50
        expected_price_buy = self.base_price
        expected_price_sell = expected_price_buy * 0.85
        reward = pricing_func(10, 5)
        expected_reward = (10 * 50.0) - (5 * 42.5)  # 500 - 212.5 = 287.5
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_high_demand(self):
        observation = {'demand': 2.0, 'supply': 1.0}
        pricing_func = self.iso.get_pricing_function(observation)
        # price_buy = 50 * (1 + 0.5*(2 - 1)) = 50 * 1.5 = 75
        expected_price_buy = 75.0
        expected_price_sell = expected_price_buy * 0.85  # 63.75
        reward = pricing_func(10, 5)
        expected_reward = (10 * 75.0) - (5 * 63.75)  # 750 - 318.75 = 431.25
        self.assertEqual(reward, 431.25)

    def test_pricing_function_low_demand(self):
        observation = {'demand': 0.5, 'supply': 1.0}
        pricing_func = self.iso.get_pricing_function(observation)
        # price_buy = 50 * (1 + 0.5*(0.5 - 1)) = 50 * 0.75 = 37.5
        expected_price_buy = 37.5
        expected_price_sell = expected_price_buy * 0.85  # 31.875
        reward = pricing_func(10, 5)
        expected_reward = (10 * 37.5) - (5 * 31.875)  # 375 - 159.375 = 215.625
        self.assertEqual(reward, 215.625)


if __name__ == '__main__':
    unittest.main()
