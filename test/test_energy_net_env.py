# test_energy_net_env.py

import warnings
from typing import Any, Dict, Union, Optional, List
import unittest
import numpy as np

import gymnasium as gym
from gymnasium import spaces


from energy_net_env.energy_net_env import EnergyNetEnv
from unittest.mock import MagicMock

NUM_SEEDS = 5

# ========================= Helper Functions =========================


def _is_numpy_array_space(space: spaces.Space) -> bool:
    """
    Returns False if provided space is not representable as a single numpy array
    (e.g. Dict and Tuple spaces return False)
    """
    return not isinstance(space, (spaces.Dict, spaces.Tuple))


def _check_unsupported_spaces(env: gym.Env, observation_space: spaces.Space, action_space: spaces.Space) -> None:
    """Emit warnings when the observation space or action space used is not supported by Stable-Baselines."""
    if isinstance(observation_space, spaces.Dict):
        nested_dict = False
        for key, space in observation_space.spaces.items():
            if isinstance(space, spaces.Dict):
                nested_dict = True
            if isinstance(space, spaces.Discrete) and space.start != 0:
                warnings.warn(
                    f"Discrete observation space (key '{key}') with a non-zero start is not supported by Stable-Baselines3. "
                    "You can use a wrapper or update your observation space."
                )

        if nested_dict:
            warnings.warn(
                "Nested observation spaces are not supported by Stable Baselines3 "
                "(Dict spaces inside Dict space). "
                "You should flatten it to have only one level of keys. "
                "For example, `dict(space1=dict(space2=Box(), space3=Box()), spaces4=Discrete())` "
                "is not supported but `dict(space2=Box(), spaces3=Box(), spaces4=Discrete())` is."
            )

    if isinstance(observation_space, spaces.Tuple):
        warnings.warn(
            "The observation space is a Tuple,"
            "this is currently not supported by Stable Baselines3. "
            "However, you can convert it to a Dict observation space "
            "(cf. https://github.com/openai/gym/blob/master/gym/spaces/dict.py). "
            "which is supported by SB3."
        )

    if isinstance(observation_space, spaces.Discrete) and observation_space.start != 0:
        warnings.warn(
            "Discrete observation space with a non-zero start is not supported by Stable-Baselines3. "
            "You can use a wrapper or update your observation space."
        )

    if isinstance(action_space, spaces.Discrete) and action_space.start != 0:
        warnings.warn(
            "Discrete action space with a non-zero start is not supported by Stable-Baselines3. "
            "You can use a wrapper or update your action space."
        )

    if not _is_numpy_array_space(action_space):
        warnings.warn(
            "The action space is not based off a numpy array. Typically this means it's either a Dict or Tuple space. "
            "This type of action space is currently not supported by Stable Baselines 3. You should try to flatten the "
            "action using a wrapper."
        )


def _is_goal_env(env: gym.Env) -> bool:
    """
    Check if the env uses the convention for goal-conditioned envs (previously, the gym.GoalEnv interface)
    """
    # We need to unwrap the env since gym.Wrapper has the compute_reward method
    return hasattr(env.unwrapped, "compute_reward")


def _check_goal_env_obs(obs: dict, observation_space: spaces.Dict, method_name: str) -> None:
    """
    Check that an environment implementing the `compute_rewards()` method
    (previously known as GoalEnv in gym) contains three elements,
    namely `observation`, `desired_goal`, and `achieved_goal`.
    """
    assert len(observation_space.spaces) == 3, (
        "A goal conditioned env must contain 3 observation keys: `observation`, `desired_goal`, and `achieved_goal`."
        f"The current observation contains {len(observation_space.spaces)} keys: {list(observation_space.spaces.keys())}"
    )

    for key in ["observation", "achieved_goal", "desired_goal"]:
        if key not in observation_space.spaces:
            raise AssertionError(
                f"The observation returned by the `{method_name}()` method of a goal-conditioned env requires the '{key}' "
                "key to be part of the observation dictionary. "
                f"Current keys are {list(observation_space.spaces.keys())}"
            )


def _check_goal_env_compute_reward(
        obs: Dict[str, Union[np.ndarray, int]],
        env: gym.Env,
        reward: float,
        info: Dict[str, Any],
) -> None:
    """
    Check that reward is computed with `compute_reward`
    and that the implementation is vectorized.
    """
    achieved_goal, desired_goal = obs["achieved_goal"], obs["desired_goal"]
    assert reward == env.compute_reward(  # type: ignore[attr-defined]
        achieved_goal, desired_goal, info
    ), "The reward was not computed with `compute_reward()`"

    achieved_goal, desired_goal = np.array(achieved_goal), np.array(desired_goal)
    batch_achieved_goals = np.array([achieved_goal, achieved_goal])
    batch_desired_goals = np.array([desired_goal, desired_goal])
    if isinstance(achieved_goal, int) or len(achieved_goal.shape) == 0:
        batch_achieved_goals = batch_achieved_goals.reshape(2, 1)
        batch_desired_goals = batch_desired_goals.reshape(2, 1)
    batch_infos = np.array([info, info])
    rewards = env.compute_reward(batch_achieved_goals, batch_desired_goals, batch_infos)  # type: ignore[attr-defined]
    assert rewards.shape == (2,), f"Unexpected shape for vectorized computation of reward: {rewards.shape} != (2,)"
    assert rewards[
               0] == reward, f"Vectorized computation of reward differs from single computation: {rewards[0]} != {reward}"


def _check_obs(obs: Union[tuple, dict, np.ndarray, int], observation_space: spaces.Space, method_name: str) -> None:
    """
    Check that the observation returned by the environment
    correspond to the declared one.
    """
    if not isinstance(observation_space, spaces.Tuple):
        assert not isinstance(
            obs, tuple
        ), f"The observation returned by the `{method_name}()` method should be a single value, not a tuple"

    # The check for a GoalEnv is done by the base class
    if isinstance(observation_space, spaces.Discrete):
        # Since https://github.com/Farama-Foundation/Gymnasium/pull/141,
        # `sample()` will return a np.int64 instead of an int
        assert np.issubdtype(type(obs),
                             np.integer), f"The observation returned by `{method_name}()` method must be an int"
    elif _is_numpy_array_space(observation_space):
        assert isinstance(obs,
                          np.ndarray), f"The observation returned by `{method_name}()` method must be a numpy array"

    assert observation_space.contains(
        obs
    ), f"The observation returned by the `{method_name}()` method does not match the given observation space"

def _check_spaces(env: gym.Env) -> None:
    """
    Check that the observation and action spaces are defined and inherit from spaces.Space. For
    envs that follow the goal-conditioned standard (previously, the gym.GoalEnv interface) we check
    the observation space is gym.spaces.Dict
    """
    # Helper to link to the code, because gym has no proper documentation
    gym_spaces = " cf https://github.com/openai/gym/blob/master/gym/spaces/"

    assert hasattr(env, "observation_space"), "You must specify an observation space (cf gym.spaces)" + gym_spaces
    assert hasattr(env, "action_space"), "You must specify an action space (cf gym.spaces)" + gym_spaces

    assert isinstance(env.observation_space, spaces.Space), (
            "The observation space must inherit from gymnasium.spaces" + gym_spaces
    )
    assert isinstance(env.action_space,
                      spaces.Space), "The action space must inherit from gymnasium.spaces" + gym_spaces

    if _is_goal_env(env):
        assert isinstance(
            env.observation_space, spaces.Dict
        ), "Goal conditioned envs (previously gym.GoalEnv) require the observation space to be gym.spaces.Dict"


def _check_box_obs(observation_space: spaces.Box, key: str = "") -> None:
    """
    Check that the observation space is correctly formatted
    when dealing with a ``Box()`` space. In particular, it checks:
    - that the dimensions are big enough when it is an image, and that the type matches
    - that the observation has an expected shape (warn the user if not)
    """
    # If image, check the low and high values, the type and the number of channels
    # and the shape (minimal value)
    if len(observation_space.shape) == 3:
        _check_image_input(observation_space, key)

    if len(observation_space.shape) not in [1, 3]:
        warnings.warn(
            f"Your observation {key} has an unconventional shape (neither an image, nor a 1D vector). "
            "We recommend you to flatten the observation "
            "to have only a 1D vector or use a custom policy to properly process the data."
        )


def _check_returned_values(env: gym.Env, observation_space: spaces.Space, action_space: spaces.Space) -> None:
    """
    Check the returned values by the env when calling `.reset()` or `.step()` methods.
    """
    # because env inherits from gymnasium.Env, we assume that `reset()` and `step()` methods exists
    reset_returns = env.reset()
    assert isinstance(reset_returns, tuple), "`reset()` must return a tuple (obs, info)"
    assert len(reset_returns) == 2, f"`reset()` must return a tuple of size 2 (obs, info), not {len(reset_returns)}"
    obs, info = reset_returns
    assert isinstance(info, dict), "The second element of the tuple return by `reset()` must be a dictionary"

    if _is_goal_env(env):
        # Make mypy happy, already checked
        assert isinstance(observation_space, spaces.Dict)
        _check_goal_env_obs(obs, observation_space, "reset")
    elif isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict), "The observation returned by `reset()` must be a dictionary"

        if not obs.keys() == observation_space.spaces.keys():
            raise AssertionError(
                "The observation keys returned by `reset()` must match the observation "
                f"space keys: {obs.keys()} != {observation_space.spaces.keys()}"
            )

        for key in observation_space.spaces.keys():
            try:
                _check_obs(obs[key], observation_space.spaces[key], "reset")
            except AssertionError as e:
                raise AssertionError(f"Error while checking key={key}: " + str(e)) from e
    else:
        _check_obs(obs, observation_space, "reset")

    # Sample a random action
    action = action_space.sample()
    data = env.step(action)

    assert len(data) == 5, "The `step()` method must return five values: obs, reward, terminated, truncated, info"

    # Unpack
    obs, reward, terminated, truncated, info = data

    if _is_goal_env(env):
        # Make mypy happy, already checked
        assert isinstance(observation_space, spaces.Dict)
        _check_goal_env_obs(obs, observation_space, "step")
        _check_goal_env_compute_reward(obs, env, reward, info)  # type: ignore[arg-type]
    elif isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict), "The observation returned by `step()` must be a dictionary"

        if not obs.keys() == observation_space.spaces.keys():
            raise AssertionError(
                "The observation keys returned by `step()` must match the observation "
                f"space keys: {obs.keys()} != {observation_space.spaces.keys()}"
            )

        for key in observation_space.spaces.keys():
            try:
                _check_obs(obs[key], observation_space.spaces[key], "step")
            except AssertionError as e:
                raise AssertionError(f"Error while checking key={key}: " + str(e)) from e

    else:
        _check_obs(obs, observation_space, "step")

    # We also allow int because the reward will be cast to float
    assert isinstance(reward, (float, int)), "The reward returned by `step()` must be a float"
    assert isinstance(terminated, bool), "The `terminated` signal must be a boolean"
    assert isinstance(truncated, bool), "The `truncated` signal must be a boolean"
    assert isinstance(info, dict), "The `info` returned by `step()` must be a python dictionary"


def _check_image_input(observation_space: spaces.Box, key: str = "") -> None:
    """
    Placeholder for image input checks. Implement as needed.
    """
    # Implement image-specific checks if necessary
    pass


def _check_env(env: gym.Env, warn: bool = True, skip_render_check: bool = True) -> None:
    """
    Check that an environment follows Gym API.
    This is particularly useful when using a custom environment.
    Please take a look at https://github.com/openai/gym/blob/master/gym/core.py
    for more information about the API.

    It also optionally checks that the environment is compatible with Stable-Baselines.

    Parameters
    ----------
    env : gym.Env
        The Gym environment that will be checked
    warn : bool, optional
        Whether to output additional warnings
        mainly related to the interaction with Stable Baselines
    skip_render_check : bool, optional
        Whether to skip the checks for the render method.
        True by default (useful for the CI)
    """
    assert isinstance(
        env, gym.Env
    ), "Your environment must inherit from the gym.Env class cf https://github.com/openai/gym/blob/master/gym/core.py"

    # ============= Check the spaces (observation and action) ================
    _check_spaces(env)

    # Define aliases for convenience
    observation_space = env.observation_space
    action_space = env.action_space

    # Warn the user if needed.
    # A warning means that the environment may run but not work properly with Stable Baselines algorithms
    if warn:
        _check_unsupported_spaces(env, observation_space, action_space)

        if isinstance(observation_space, spaces.Dict):
            obs_spaces = observation_space.spaces
        else:
            obs_spaces = {"": observation_space}

        for key, space in obs_spaces.items():
            if isinstance(space, spaces.Box):
                _check_box_obs(space, key)

        # Check for the action space, it may lead to hard-to-debug issues
        if isinstance(action_space, spaces.Box) and (
                np.any(np.abs(action_space.low) != np.abs(action_space.high))
                or np.any(action_space.low != -1)
                or np.any(action_space.high != 1)
        ):
            warnings.warn(
                "We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) "
                "cf https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html"
            )

        if isinstance(action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([action_space.low, action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        if isinstance(action_space, spaces.Box) and action_space.dtype != np.dtype(np.float32):
            warnings.warn(
                f"Your action space has dtype {action_space.dtype}, we recommend using np.float32 to avoid cast errors."
            )

    # ============ Check the returned values ===============
    _check_returned_values(env, observation_space, action_space)

    # # ==== Check the render method and the declared render modes ====
    # if not skip_render_check:
    #     _check_render(env, warn)  # pragma: no cover

    try:
        _check_for_nested_spaces(env.observation_space)
    except NotImplementedError:
        pass


def _check_for_nested_spaces(obs_space: spaces.Space) -> None:
    """
    Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    If so, raise an Exception informing that there is no support for this.
    
    Parameters
    ----------
    obs_space : spaces.Space
        An observation space
    """
    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = obs_space.spaces.values() if isinstance(obs_space, spaces.Dict) else obs_space.spaces
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)."
                )

# ========================= Test Classes =========================


class TestEnergyNetEnv(unittest.TestCase):
    def setUp(self):
        """
        Set up the EnergyNetEnv instance with mock configurations for testing.
        """
        
        # Initialize EnergyNetEnv with mock configurations
        self.env = EnergyNetEnv(
            render_mode=None,
            env_config_path='configs/environment_config.yaml',
            iso_config_path='configs/iso_config.yaml',
            pcs_unit_config_path='configs/pcs_unit_config.yaml',
            log_file='logs/test_environment.log'
        )

        self.env_config = self.env.load_config('configs/environment_config.yaml')
        # Reset the environment after mocking
        self.env.reset()

    def tearDown(self):
        """
        Clean up after tests.
        """
        self.env.close()

    def test_initialization(self):
        """
        Test that the environment initializes correctly with given configurations.
        """
        # Check initial energy level
        self.assertEqual(self.env.energy_lvl, self.env_config['energy']['init'])

        # Check action space bounds
        expected_low = -self.env_config['energy']['discharge_rate_max']
        expected_high = self.env_config['energy']['charge_rate_max']
        np.testing.assert_array_equal(self.env.action_space.low, np.array([expected_low]))
        np.testing.assert_array_equal(self.env.action_space.high, np.array([expected_high]))

        # Check observation space bounds
        expected_obs_low = np.array([
            self.env_config['energy']['min'],
            0.0,
            0.0,
            0.0
        ], dtype=np.float32)
        expected_obs_high = np.array([
            self.env_config['energy']['max'],
            1.0,
            np.inf,
            np.inf
        ], dtype=np.float32)
        np.testing.assert_array_equal(self.env.observation_space.low, expected_obs_low)
        np.testing.assert_array_equal(self.env.observation_space.high, expected_obs_high)

    def test_reset(self):
        """
        Test that resetting the environment restores it to the initial state.
        """
        # Perform some steps
        action = 1.0  # Charge
        self.env.step(action)
        self.env.step(-1.0)  # Discharge

        # Reset the environment
        observation, info = self.env.reset()

        # Check that energy level is reset
        self.assertEqual(self.env.energy_lvl, self.env_config['energy']['init'])

        # Check that step count is reset
        self.assertEqual(self.env.count, 0)

        # Check initial observation
        expected_observation = np.array([
            self.env_config['energy']['init'],
            0.0,  # current_time after reset
            self.env.PCSUnit.get_self_production(),
            self.env.PCSUnit.get_self_consumption()
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(observation, expected_observation, decimal=5)

    def test_step_with_valid_action_charge(self):
        """
        Test stepping the environment with a valid charge action.
        """
        action = 2.0  # Charge at 2 MW
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Check that the energy level has increased appropriately
        expected_energy = self.env_config['energy']['init'] + action * self.env.PCSUnit.battery.charge_efficiency
        self.assertAlmostEqual(self.env.energy_lvl, expected_energy, places=5)

        # Check that the step count has incremented
        self.assertEqual(self.env.count, 1)

        # Check that done flags are False
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_step_with_valid_action_discharge(self):
        """
        Test stepping the environment with a valid discharge action.
        """
        action = -1.5  # Discharge at 1.5 MW
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Check that the energy level has decreased appropriately
        expected_energy = self.env_config['energy']['init'] - abs(action) * self.env.PCSUnit.battery.discharge_efficiency
        self.assertAlmostEqual(self.env.energy_lvl, expected_energy, places=5)

        # Check that the step count has incremented
        self.assertEqual(self.env.count, 1)

        # Check that done flags are False
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_step_exceeds_max_steps(self):
        """
        Test that the environment correctly terminates after reaching max steps.
        """
        # Set max_steps_per_episode to 2 for testing
        self.env.max_steps_per_episode = 2

        # Perform two steps
        self.env.step(1.0)
        observation, reward, terminated, truncated, info = self.env.step(1.0)

        # Check that terminated is True on the second step
        self.assertTrue(terminated)
        self.assertFalse(truncated)


    def test_observation_structure(self):
        """
        Test that the observation has the correct structure and data types.
        """
        action = 0.0  # No action
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Check observation shape
        self.assertEqual(observation.shape, (4,))

        # Check data types
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.dtype, np.float32)

    def test_reward_calculation(self):
        """
        Test that the reward is calculated correctly based on the ISO pricing function.
        """
        # Mock the ISO's get_pricing_function to return a fixed reward calculation
        mock_pricing_function = MagicMock(return_value=100.0)
        self.env.ISO.get_pricing_function = MagicMock(return_value=mock_pricing_function)

        action = 1.0  # Charge
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Check that the pricing function was called with correct parameters
        self.env.ISO.get_pricing_function.assert_called_once()
        mock_pricing_function.assert_called_once_with(
            1.0 + max(0.0, self.env.PCSUnit.get_self_consumption() - self.env.PCSUnit.get_self_production()),
            0.0 + max(0.0, self.env.PCSUnit.get_self_production() - self.env.PCSUnit.get_self_consumption())
        )

        # Check that reward is as expected
        self.assertEqual(reward, 100.0)

    def test_info_structure(self):
        """
        Test that the info dictionary has the correct structure.
        """
        action = 0.0
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Check that 'running_avg' key exists and is a float
        self.assertIn('running_avg', info)
        self.assertIsInstance(info['running_avg'], float)
        
    def test_step_with_numpy_array_action(self):
        """
        Test stepping the environment with a valid numpy array action.
        """
        action_array = np.array([1.0], dtype=np.float32)  # Valid charge action
        observation, reward, terminated, truncated, info = self.env.step(action_array)

        # Check that the energy level has increased appropriately
        expected_energy = self.env_config['energy']['init'] + action_array.item() * self.env.PCSUnit.battery.charge_efficiency
        self.assertAlmostEqual(self.env.energy_lvl, expected_energy, places=5)

        # Check that the step count has incremented
        self.assertEqual(self.env.count, 1)

        # Check that done flags are False
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        # Check observation structure
        self.assertEqual(observation.shape, (4,))
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.dtype, np.float32)
        
        
    def test_step_with_boundary_actions(self):
        """
        Test stepping the environment with actions at the boundary of the action space.
        """
        # Test maximum charge action
        max_charge_action = self.env.action_space.high
        observation, reward, terminated, truncated, info = self.env.step(max_charge_action)
        expected_energy = self.env_config['energy']['init'] + max_charge_action.item() * self.env.PCSUnit.battery.charge_efficiency
        self.assertAlmostEqual(self.env.energy_lvl, expected_energy, places=5)

        # Test maximum discharge action
        self.env.reset()
        max_discharge_action = self.env.action_space.low
        observation, reward, terminated, truncated, info = self.env.step(max_discharge_action)
        expected_energy = self.env_config['energy']['init'] - abs(max_discharge_action.item()) * self.env.PCSUnit.battery.discharge_efficiency
        self.assertAlmostEqual(self.env.energy_lvl, expected_energy, places=5)
        
    def test_gym_api(self):
        
        seed = hash('EnergyNetEnv')
        for _ in range(NUM_SEEDS):
            seed = abs(hash(str(seed)))
            env = EnergyNetEnv(
                render_mode=None,
                env_config_path='configs/environment_config.yaml',
                iso_config_path='configs/iso_config.yaml',
                pcs_unit_config_path='configs/pcs_unit_config.yaml',
                log_file='logs/test_environment.log'
            )
            env.reset(seed=seed)
            _check_env(env, skip_render_check=False)

    

    # ========================= Run Tests =========================

if __name__ == '__main__':
    unittest.main()
