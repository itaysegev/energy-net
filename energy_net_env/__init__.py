# energy_net_env/__init__.py

from gymnasium.envs.registration import register

print("Registering EnergyNetEnv-v0 environment...")

register(
    id='EnergyNetEnv-v0',
    entry_point='energy_net_env.energy_net_env:EnergyNetEnv',
)