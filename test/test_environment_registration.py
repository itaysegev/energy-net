# tests/test_environment_registration.py

import unittest
import gymnasium as gym

class TestEnvironmentRegistration(unittest.TestCase):
    def test_environment_registration(self):
        """
        Test that EnergyNetEnv is correctly registered and can be instantiated.
        """
        try:
            env = gym.make('EnergyNetEnv-v0/EnergyNetEnv-v0')
            self.assertIsNotNone(env, "Failed to instantiate EnergyNetEnv-v0.")
            self.assertEqual(env.spec.id, 'EnergyNetEnv-v0/EnergyNetEnv-v0', "Environment ID does not match.")
            env.close()
        except Exception as e:
            self.fail(f"Failed to make EnergyNetEnv-v0: {e}")

if __name__ == '__main__':
    unittest.main()
