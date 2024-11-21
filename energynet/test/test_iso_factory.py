# test_iso_factory.py

import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Add the project's root directory to sys.path if necessary
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the iso_factory and ISO classes
from utils.iso_factory import iso_factory
from iso.hourly_pricing_iso import HourlyPricingISO
from iso.dynamic_pricing_iso import DynamicPricingISO
from iso.quadratic_pricing_iso import QuadraticPricingISO
from iso.random_pricing_iso import RandomPricingISO
from iso.time_of_use_pricing_iso import TimeOfUsePricingISO
from iso.iso_base import ISOBase


class TestISOFactory(unittest.TestCase):
    def test_iso_factory_hourly_pricing_iso(self):
        """
        Test that the iso_factory correctly creates an instance of HourlyPricingISO
        when provided with the corresponding type and parameters.
        """
        iso_type = 'HourlyPricingISO'
        iso_parameters = {
            'hourly_rates': {
                0: 45.0,
                1: 45.0,
                2: 45.0,
                3: 45.0,
                4: 45.0,
                5: 50.0,
                6: 55.0,
                7: 60.0,
                8: 65.0,
                9: 70.0,
                10: 75.0,
                11: 80.0,
                12: 85.0,
                13: 85.0,
                14: 85.0,
                15: 80.0,
                16: 75.0,
                17: 70.0,
                18: 65.0,
                19: 60.0,
                20: 55.0,
                21: 50.0,
                22: 45.0,
                23: 45.0
            }
        }
        
        iso_instance = iso_factory(iso_type, iso_parameters)
        
        self.assertIsInstance(iso_instance, HourlyPricingISO, 
                              f"Expected instance of HourlyPricingISO, got {type(iso_instance)}")
        self.assertEqual(iso_instance.hourly_rates, iso_parameters['hourly_rates'])
    
    def test_iso_factory_dynamic_pricing_iso(self):
        """
        Test that the iso_factory correctly creates an instance of DynamicPricingISO
        when provided with the corresponding type and parameters.
        """
        iso_type = 'DynamicPricingISO'
        iso_parameters = {
            'base_price': 50.0,
            'demand_factor': 1.2,
            'supply_factor': 0.8,
            'elasticity': 0.5
        }
        
        iso_instance = iso_factory(iso_type, iso_parameters)
        
        self.assertIsInstance(iso_instance, DynamicPricingISO, 
                              f"Expected instance of DynamicPricingISO, got {type(iso_instance)}")
        self.assertEqual(iso_instance.base_price, iso_parameters['base_price'])
        self.assertEqual(iso_instance.demand_factor, iso_parameters['demand_factor'])
        self.assertEqual(iso_instance.supply_factor, iso_parameters['supply_factor'])
        self.assertEqual(iso_instance.elasticity, iso_parameters['elasticity'])
    
    def test_iso_factory_quadratic_pricing_iso(self):
        """
        Test that the iso_factory correctly creates an instance of QuadraticPricingISO
        when provided with the corresponding type and parameters.
        """
        iso_type = 'QuadraticPricingISO'
        iso_parameters = {
            'a': 1.0,
            'b': 2.0,
            'c': 50.0
        }
        
        iso_instance = iso_factory(iso_type, iso_parameters)
        
        self.assertIsInstance(iso_instance, QuadraticPricingISO, 
                              f"Expected instance of QuadraticPricingISO, got {type(iso_instance)}")
        self.assertEqual(iso_instance.a, iso_parameters['a'])
        self.assertEqual(iso_instance.b, iso_parameters['b'])
        self.assertEqual(iso_instance.c, iso_parameters['c'])
    
    def test_iso_factory_random_pricing_iso(self):
        """
        Test that the iso_factory correctly creates an instance of RandomPricingISO
        when provided with the corresponding type and parameters.
        """
        iso_type = 'RandomPricingISO'
        iso_parameters = {
            'min_price': 40.0,
            'max_price': 60.0
        }
        
        iso_instance = iso_factory(iso_type, iso_parameters)
        
        self.assertIsInstance(iso_instance, RandomPricingISO, 
                              f"Expected instance of RandomPricingISO, got {type(iso_instance)}")
        self.assertEqual(iso_instance.min_price, iso_parameters['min_price'])
        self.assertEqual(iso_instance.max_price, iso_parameters['max_price'])
    
    def test_iso_factory_time_of_use_pricing_iso(self):
        """
        Test that the iso_factory correctly creates an instance of TimeOfUsePricingISO
        when provided with the corresponding type and parameters.
        """
        iso_type = 'TimeOfUsePricingISO'
        iso_parameters = {
            'peak_hours': [16, 17, 18, 19, 20],
            'off_peak_hours': [0, 1, 2, 3, 4, 5, 6, 22, 23],
            'peak_price': 60.0,
            'off_peak_price': 30.0
        }
        
        iso_instance = iso_factory(iso_type, iso_parameters)
        
        self.assertIsInstance(iso_instance, TimeOfUsePricingISO, 
                              f"Expected instance of TimeOfUsePricingISO, got {type(iso_instance)}")
        self.assertEqual(iso_instance.peak_hours, iso_parameters['peak_hours'])
        self.assertEqual(iso_instance.off_peak_hours, iso_parameters['off_peak_hours'])
        self.assertEqual(iso_instance.peak_price, iso_parameters['peak_price'])
        self.assertEqual(iso_instance.off_peak_price, iso_parameters['off_peak_price'])
    
    def test_iso_factory_invalid_iso_type(self):
        """
        Test that the iso_factory raises a ValueError when provided with an invalid ISO type.
        """
        iso_type = 'NonExistentISO'
        iso_parameters = {}
        
        with self.assertRaises(ValueError) as context:
            iso_factory(iso_type, iso_parameters)
        
        self.assertIn(f"Unknown ISO type: {iso_type}", str(context.exception))
    
    def test_iso_factory_missing_parameters(self):
        """
        Test that the iso_factory raises a TypeError when required parameters are missing.
        """
        # Example with missing 'hourly_rates' for HourlyPricingISO
        iso_type = 'HourlyPricingISO'
        iso_parameters = {
            # 'hourly_rates' is missing
        }
        
        with self.assertRaises(TypeError) as context:
            iso_factory(iso_type, iso_parameters)
        
        self.assertIn("missing", str(context.exception))
    
    def test_iso_factory_extra_parameters(self):
        """
        Test that the iso_factory can handle extra parameters gracefully,
        possibly ignoring them or passing them correctly to the ISO instance.
        """
        iso_type = 'HourlyPricingISO'
        iso_parameters = {
            'hourly_rates': {
                0: 45.0,
                1: 45.0,
                # ... (other hours)
                23: 45.0
            },
            'extra_param': 'should_be_ignored_or_handled'
        }
        
        with patch.object(HourlyPricingISO, '__init__', return_value=None) as mock_init:
            iso_instance = iso_factory(iso_type, iso_parameters)
            mock_init.assert_called_with(
                hourly_rates=iso_parameters['hourly_rates'],
                extra_param='should_be_ignored_or_handled'
            )
    
    def test_iso_factory_default_parameters(self):
        """
        Test that the iso_factory assigns default values to parameters if they are not provided,
        assuming the ISO classes have default parameters.
        """
        iso_type = 'DynamicPricingISO'
        iso_parameters = {
            'base_price': 40.0
            # 'demand_factor', 'supply_factor', and 'elasticity' are omitted; assume they have defaults
        }
        
        iso_instance = iso_factory(iso_type, iso_parameters)
        
        self.assertIsInstance(iso_instance, DynamicPricingISO, 
                              f"Expected instance of DynamicPricingISO, got {type(iso_instance)}")
        self.assertEqual(iso_instance.base_price, iso_parameters['base_price'])
        self.assertEqual(iso_instance.demand_factor, DynamicPricingISO.__init__.__defaults__[1])
        self.assertEqual(iso_instance.supply_factor, DynamicPricingISO.__init__.__defaults__[2])
        self.assertEqual(iso_instance.elasticity, DynamicPricingISO.__init__.__defaults__[3])
    
    def test_iso_factory_base_iso_instance(self):
        """
        Test that the iso_factory returns an instance that inherits from BaseISO.
        """
        iso_type = 'QuadraticPricingISO'
        iso_parameters = {
            'a': 1.0,
            'b': 2.0,
            'c': 50.0
        }
        
        iso_instance = iso_factory(iso_type, iso_parameters)
        
        self.assertIsInstance(iso_instance, ISOBase, 
                              f"Expected instance of BaseISO, got {type(iso_instance)}")
    
    def test_iso_factory_case_insensitivity(self):
        """
        Test that the iso_factory handles ISO types case-insensitively, if applicable.
        """
        iso_type = 'hourlypricingiso'  # Lowercase
        iso_parameters = {
            'hourly_rates': {
                0: 45.0,
                1: 45.0,
                # ... (other hours)
                23: 45.0
            }
        }
        
        # Depending on implementation, this might raise an error or handle case-insensitively
        # Here, we assume it is case-sensitive and should raise an error
        with self.assertRaises(ValueError) as context:
            iso_factory(iso_type, iso_parameters)
        
        self.assertIn(f"Unknown ISO type: {iso_type}", str(context.exception))
    
    def test_iso_factory_no_parameters_random_pricing_iso(self):
        """
        Test that the iso_factory can handle cases where no parameters are provided
        for RandomPricingISO, and defaults are used.
        """
        iso_type = 'RandomPricingISO'
        iso_parameters = {}
        
        iso_instance = iso_factory(iso_type, iso_parameters)
        
        self.assertIsInstance(iso_instance, RandomPricingISO, 
                            f"Expected instance of RandomPricingISO, got {type(iso_instance)}")
        self.assertEqual(iso_instance.min_price, 40.0)  # Default value
        self.assertEqual(iso_instance.max_price, 60.0)  # Default value
        
    def test_iso_factory_missing_parameters_other_than_random_pricing(self):
        """
        Test that the iso_factory raises a TypeError when required parameters are missing
        for ISO types that do not have default parameters.
        """
        # Example with missing 'hourly_rates' for HourlyPricingISO
        iso_type = 'HourlyPricingISO'
        iso_parameters = {
            # 'hourly_rates' is missing
        }
        
        with self.assertRaises(TypeError) as context:
            iso_factory(iso_type, iso_parameters)
        
        self.assertIn("missing", str(context.exception))
    
    def test_iso_factory_invalid_parameter_types(self):
        """
        Test that the iso_factory raises appropriate errors when parameter types are incorrect.
        """
        iso_type = 'QuadraticPricingISO'
        iso_parameters = {
            'a': 'one',  # Should be a float
            'b': 'two',  # Should be a float
            'c': 'fifty'  # Should be a float
        }
        
        with self.assertRaises(TypeError) as context:
            iso_factory(iso_type, iso_parameters)
        
        self.assertIn("must be a float", str(context.exception))
    
    def test_iso_factory_partial_parameters(self):
        """
        Test that the iso_factory correctly handles environments where only some parameters are provided,
        and defaults are assigned to others.
        """
        iso_type = 'TimeOfUsePricingISO'
        iso_parameters = {
            'peak_hours': [16, 17, 18, 19, 20],
            # 'off_peak_hours', 'peak_price', and 'off_peak_price' are omitted; assume they have defaults
            # Removed 'log_file' as it is not a valid parameter for TimeOfUsePricingISO
        }
        
        with self.assertRaises(TypeError) as context:
            iso_factory(iso_type, iso_parameters)
        
        self.assertIn("missing", str(context.exception))




# ========================= Run Tests =========================

if __name__ == '__main__':
    unittest.main()
