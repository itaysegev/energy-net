# main.py

from energy_net_env.energy_net_env import EnergyNetEnv
import numpy as np

def main():
    """
    Main function to interact with the EnergyNetEnv.
    Runs a simulation loop with random actions.
    """
    # Define configuration paths (update paths as necessary)
    env_config_path = 'configs/environment_config.yaml'
    iso_config_path = 'configs/iso_config.yaml'
    pcs_unit_config_path = 'configs/pcs_unit_config.yaml'
    log_file = 'logs/environment.log'

    # Initialize the environment
    env = EnergyNetEnv(
        render_mode='human',  # Assuming render is implemented
        env_config_path=env_config_path,
        iso_config_path=iso_config_path,
        pcs_unit_config_path=pcs_unit_config_path,
        log_file=log_file
    )

    # Reset the environment
    observation, info = env.reset()

    done = False
    truncated = False

    print("Starting EnergyNet Simulation...")

    while not done and not truncated:
        # Sample a random action from the action space
        action = env.action_space.sample()
        
        # Take a step in the environment
        observation, reward, done, truncated, info = env.step(action)
        
        # Render the current state
        try:
            env.render()
        except NotImplementedError:
            pass  # Render not implemented; skip

        # Print observation and reward
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print("-" * 50)

    print("Simulation completed.")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
