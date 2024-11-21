# components/grid_entity.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

from utils.logger import setup_logger  # Import the logger setup utility


class GridEntity(ABC):
    """
    Abstract Base Class for all grid entities such as Battery, ProductionUnit, and ConsumptionUnit.
    
    This class defines the interface that all grid entities must implement, ensuring consistency
    across different components within the smart grid simulation.
    """

    def __init__(self, dynamics: Any, log_file: str = 'logs/grid_entity.log'):
        """
        Initializes the GridEntity with specified dynamics and sets up logging.
        
        Args:
            dynamics (Any): The dynamics model associated with the grid entity.
            log_file (str): Path to the log file for the grid entity.
        """
        self.logger = setup_logger(self.__class__.__name__, log_file)
        self.dynamics = dynamics
        self.logger.info(f"Initialized {self.__class__.__name__} with dynamics: {self.dynamics}")

    @abstractmethod
    def perform_action(self, action: float) -> None:
        """
        Performs an action (e.g., charging or discharging) on the grid entity.
        
        This method must be implemented by all subclasses, defining how the entity responds
        to a given action.
        
        Args:
            action (float): The action to perform. The meaning of the action depends on the entity.
                             For example, positive values might indicate charging, while negative
                             values indicate discharging for a Battery.
        """
        pass

    @abstractmethod
    def get_state(self) -> float:
        """
        Retrieves the current state of the grid entity.
        
        This method must be implemented by all subclasses, providing a way to access the
        entity's current state (e.g., energy level for a Battery).
        
        Returns:
            float: The current state of the entity.
        """
        pass

    @abstractmethod
    def update(self, time: float, action: float = 0.0) -> None:
        """
        Updates the state of the grid entity based on the current time and action.
        
        This method must be implemented by all subclasses, defining how the entity's state
        evolves over time and in response to actions.
        
        Args:
            time (float): The current time as a fraction of the day (0 to 1).
            action (float, optional): The action to perform (default is 0.0).
                                       The meaning of the action depends on the entity.
        """
        pass

    def reset(self) -> None:
        """
        Resets the grid entity to its initial state.
        
        Subclasses can override this method to define specific reset behaviors.
        """
        self.logger.info(f"Resetting {self.__class__.__name__} to initial state.")
        # Default implementation does nothing. Subclasses should override as needed.
        pass
