# pcs_unit/pcs_unit.py

from typing import Any, Dict, Optional
from components.battery import Battery
from components.production_unit import ProductionUnit
from components.consumption_unit import ConsumptionUnit

from dynamics.energy_dynamics import EnergyDynamics

from utils.logger import setup_logger  # Import the logger setup
from utils.utils import dict_level_alingment

class PCSUnit:
    """
    Power Conversion System Unit (PCSUnit) managing Battery, ProductionUnit, and ConsumptionUnit.

    This class integrates the battery, production, and consumption components, allowing for
    coordinated updates and state management within the smart grid simulation.
    """

    def __init__(
        self,
        battery_dynamics: EnergyDynamics,
        production_dynamics: EnergyDynamics,
        consumption_dynamics: EnergyDynamics,
        config: Dict[str, Any],
        log_file: Optional[str] = 'logs/pcs_unit.log'  # Path to the PCSUnit log file
    ):
        """
        Initializes the PCSUnit with its components and configuration.

        Args:
            battery_dynamics (EnergyDynamics): Dynamics defining the battery's behavior.
            production_dynamics (EnergyDynamics): Dynamics defining the production unit's behavior.
            consumption_dynamics (EnergyDynamics): Dynamics defining the consumption unit's behavior.
            config (Dict[str, Any]): Configuration parameters for the PCSUnit components.
            log_file (str, optional): Path to the PCSUnit log file.
        """
        # Set up logger
        self.logger = setup_logger('PCSUnit', log_file)
        self.logger.info("Initializing PCSUnit.")

        # Initialize Battery
        self.battery = Battery(dynamics=battery_dynamics, config=dict_level_alingment(config, 'battery', 'model_parameters'))
        self.logger.info(f"Initialized Battery with energy level: {self.battery.energy_level} MWh")

        # Initialize ProductionUnit
        self.production_unit = ProductionUnit(dynamics=production_dynamics, config=dict_level_alingment(config, 'production_unit', 'model_parameters'))
        self.logger.info(f"Initialized ProductionUnit with current production: {self.production_unit.get_state()} MWh")

        # Initialize ConsumptionUnit
        self.consumption_unit = ConsumptionUnit(dynamics=consumption_dynamics, config=dict_level_alingment(config, 'consumption_unit', 'model_parameters'))
        self.logger.info(f"Initialized ConsumptionUnit with current consumption: {self.consumption_unit.get_state()} MWh")

    def reset(self) -> None:
        """
        Resets all components to their initial states.
        """
        self.logger.info("Resetting PCSUnit components.")
        # Reset Battery
        self.battery.reset()
        self.logger.debug(f"Battery reset to energy level: {self.battery.energy_level} MWh")

        # Reset ProductionUnit
        self.production_unit.reset()
        self.logger.debug(f"ProductionUnit reset to production: {self.production_unit.get_state()} MWh")

        # Reset ConsumptionUnit
        self.consumption_unit.reset()
        self.logger.debug(f"ConsumptionUnit reset to consumption: {self.consumption_unit.get_state()} MWh")

    def update(self, time: float, battery_action: float) -> None:
        """
        Updates the state of all components based on the current time and battery action.

        Args:
            time (float): Current time as a fraction of the day (0 to 1).
            battery_action (float): Charging (+) or discharging (-) power (MW).
        """
        self.logger.info(f"Updating PCSUnit at time: {time}, with battery_action: {battery_action} MW")

        # Update Battery with the action
        self.battery.update(time=time, action=battery_action)
        self.logger.debug(f"Battery updated to energy level: {self.battery.get_state()} MWh")

        # Update ProductionUnit (no action required)
        self.production_unit.update(time=time, action=0.0)
        self.logger.debug(f"ProductionUnit updated to production: {self.production_unit.get_state()} MWh")

        # Update ConsumptionUnit (no action required)
        self.consumption_unit.update(time=time, action=0.0)
        self.logger.debug(f"ConsumptionUnit updated to consumption: {self.consumption_unit.get_state()} MWh")

    def get_self_production(self) -> float:
        """
        Retrieves the current self-production value.

        Returns:
            float: Current production in MWh.
        """
        production = self.production_unit.get_state()
        self.logger.debug(f"Retrieved self-production: {production} MWh")
        return production

    def get_self_consumption(self) -> float:
        """
        Retrieves the current self-consumption value.

        Returns:
            float: Current consumption in MWh.
        """
        consumption = self.consumption_unit.get_state()
        self.logger.debug(f"Retrieved self-consumption: {consumption} MWh")
        return consumption
