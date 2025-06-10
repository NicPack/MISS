import os
import os.path

from interfaces.behaviors import BasicReproduction
from simulation.simulation import (
    EncounterManager,
    SimulationEngine,
    WorldState,
)
from utils.logger import SimulationLogger

DIR = "results"
FILE_NR = len(
    [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
)
BOARD_SIZE = (40, 40)
POPULATION_SIZE = 1000


def setup_simulation():
    world = WorldState(BOARD_SIZE)
    logger = SimulationLogger(log_all_encounters=True)

    # Explicitly provide a working reproduction system
    reproduction_system = BasicReproduction()
    encounter_manager = EncounterManager(
        reproduction_system=reproduction_system,
        logger=logger,
    )

    engine = SimulationEngine(
        world_state=world,
        encounter_manager=encounter_manager,
        logger=logger,
        population_size=POPULATION_SIZE,
    )

    # Properly initialize population and add agents to the world
    engine.initialize_population()

    return engine, logger


def main():
    # Main simulation loop
    engine, logger = setup_simulation()

    for _ in range(1000):  # Run 1000 timesteps
        engine.step()

    # Save results after simulation
    logger.save_data(f"reimplementation/results/simulation_results{FILE_NR}.json")

    # Generate analysis report
    analysis = logger.get_analysis_summary()
    print("Simulation Report:")
    print(f"Final Population: {analysis['final_population']}")
    print(f"Mating Success Rate: {analysis['mating_success_rate']:.2%}")
    # Example: Get education mixing trends
    education_data = logger._analyze_education_mixing()
    print(f"{education_data}")


if __name__ == "__main__":
    main()
