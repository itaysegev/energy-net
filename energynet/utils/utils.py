import os
import yaml

def load_config(self, config_path: str) -> dict:
    """
    Loads and validates a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Example validation
    required_energy_params = ['min', 'max', 'init', 'charge_rate_max', 'discharge_rate_max', 'charge_efficiency', 'discharge_efficiency']
    for param in required_energy_params:
        if param not in config.get('energy', {}):
            raise ValueError(f"Missing energy parameter in config: {param}")
    
    # Add more validations as needed
    
    return config


def dict_level_alingment(d, key1, key2):
    return d[key1] if key2 not in d[key1] else d[key1][key2]
