# dynamics/data_driven_dynamics.py

import pandas as pd
from typing import Any, Dict
from dynamics.energy_dynamics import EnergyDynamics


class DataDrivenDynamics(EnergyDynamics):
    """
    Data-Driven Dynamics implementation.
    Defines behavior based on data from external sources.
    """

    def __init__(self, data_file: str, value_column: str):
        """
        Initializes the DataDrivenDynamics with data from a file.

        Args:
            data_file (str): Path to the data file (e.g., CSV).
            value_column (str): Name of the column to retrieve values from.
        """
        self.data = pd.read_csv(data_file)
        self.value_column = value_column
        self.current_index = 0

    def reset(self) -> None:
        """
        Resets the internal state of the data-driven dynamics.
        """
        self.current_index = 0

    def get_value(self, **kwargs) -> Any:
        """
        Retrieves the value from the data corresponding to the current time.

        Args:
            **kwargs: Should contain 'time' as a fraction of the day (0 to 1).

        Returns:
            Any: The value from the data corresponding to the current time.
        """
        time = kwargs.get('time', 0.0)
        total_steps = len(self.data)
        index = int(time * total_steps) % total_steps
        self.current_index = index
        value = self.data.iloc[index][self.value_column]
        return value
