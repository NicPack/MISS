from __future__ import annotations

from typing import Dict, List, Tuple

from reimplementation.interfaces.behaviors import BasicReproduction
from reimplementation.utils.logger import SimulationLogger

# =============================================================================
# WORLD AND SIMULATION MANAGEMENT
# =============================================================================


class WorldState:
    """Centralized world state"""

    def __init__(self, map_size: Tuple[int, int]):
        self.map_size = map_size
        self.agents: Dict[int, "Agent"] = {}
        self.spatial_index: Dict[Tuple[int, int], List[int]] = {}
        self.time_step = 0

    def add_agent(self, agent: "Agent"):
        self.agents[agent.id] = agent
        self._update_spatial_index(agent)

    def remove_agent(self, agent_id: int):
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            self._remove_from_spatial_index(agent)
            del self.agents[agent_id]

    def get_agents_at_position(self, position: Tuple[int, int]) -> List["Agent"]:
        agent_ids = self.spatial_index.get(position, [])
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]

    def _update_spatial_index(self, agent: "Agent"):
        # Remove from old position
        self._remove_from_spatial_index(agent)
        # Add to new position
        if agent.position not in self.spatial_index:
            self.spatial_index[agent.position] = []
        self.spatial_index[agent.position].append(agent.id)

    def _remove_from_spatial_index(self, agent: "Agent"):
        for pos, agent_list in self.spatial_index.items():
            if agent.id in agent_list:
                agent_list.remove(agent.id)
                if not agent_list:
                    del self.spatial_index[pos]
                break


class EncounterManager:
    """Manages agent encounters and interactions"""

    def __init__(
        self,
        reproduction_system: "ReproductionSystem" = None,
        logger: SimulationLogger = None,
    ):
        self.reproduction_system = BasicReproduction()
        self.encounter_history: Dict[Tuple[int, int], int] = {}
        self.logger = logger

    def process_encounters(self, world_state: WorldState) -> List["Agent"]:
        """Process all encounters and return new offspring"""
        new_agents = []

        for position, agent_ids in world_state.spatial_index.items():
            if len(agent_ids) > 1:
                # Process all pairs at this position
                for i in range(len(agent_ids)):
                    for j in range(i + 1, len(agent_ids)):
                        agent1 = world_state.agents[agent_ids[i]]
                        agent2 = world_state.agents[agent_ids[j]]

                        # Only process opposite sex encounters
                        if agent1.sex == agent2.sex:
                            continue
                        if agent1.sex == "M":
                            male = agent1
                            female = agent2
                        else:
                            male = agent2
                            female = agent1

                        # Check if they haven't met before
                        encounter_key = tuple(sorted([male.id, female.id]))
                        if encounter_key not in self.encounter_history:
                            self.encounter_history[encounter_key] = (
                                world_state.time_step
                            )

                            # Record the meeting
                            if hasattr(male, "females_met") and female.sex == "F":
                                male.females_met.add(female.id)
                            if hasattr(female, "males_met") and male.sex == "M":
                                female.males_met.add(male.id)

                            # Attempt reproduction
                            offspring = self.reproduction_system.attempt_reproduction(
                                male, female
                            )

                            # Only log if logger exists
                            if self.logger:
                                # LOG THE ENCOUNTER
                                self.logger.log_encounter(
                                    male,
                                    female,
                                    world_state,
                                    reproduction_result=offspring,
                                )

                            if offspring:
                                # Add to new agents
                                new_agents.append(offspring)

                                # Update parent stats
                                male.offspring_count += 1
                                female.offspring_count += 1

                                # Record mating data
                                if hasattr(male, "mating_pairs_ages"):
                                    male.mating_pairs_ages.append(
                                        (int(male.age), int(female.age))
                                    )
                                if hasattr(female, "mating_pairs_ages"):
                                    female.mating_pairs_ages.append(
                                        (int(female.age), int(male.age))
                                    )

                                # LOG REPRODUCTION
                                if self.logger:
                                    self.logger.log_reproduction(
                                        male, female, offspring, world_state
                                    )

        return new_agents


class SimulationEngine:
    """Main simulation controller"""

    def __init__(
        self,
        world_state: WorldState,
        encounter_manager: EncounterManager,
        logger: SimulationLogger,
        population_size: int = 100,
    ):
        self.world_state = world_state
        self.encounter_manager = encounter_manager
        self.logger = logger
        self.population_size = population_size

    def step(self):
        """Execute one simulation step"""
        # 1. Update all agents (movement, aging, etc.)
        agents_to_remove = []
        for agent in self.world_state.agents.values():
            if agent.is_alive:
                agent.update(self.world_state)
                self.world_state._update_spatial_index(agent)
            else:
                agents_to_remove.append(agent.id)

        # 2. Remove dead agents
        for agent_id in agents_to_remove:
            self.world_state.remove_agent(agent_id)

        # PROCESS ENCOUNTERS (logging happens inside)
        new_agents = self.encounter_manager.process_encounters(self.world_state)

        # 4. Add new agents (they already have proper IDs)
        for agent in new_agents:
            self.world_state.add_agent(agent)

        # LOG TIMESTEP DATA
        self.logger.log_timestep(
            world_state=self.world_state,
            encounters=None,  # Already logged individually
            reproductions=None,  # Already logged individually
        )

        self.world_state.time_step += 1

    def initialize_population(self):
        """Initialize the population with a given size"""
        population = self.encounter_manager.reproduction_system.initialize_population(
            population_size=self.population_size
        )
        for agent in population:
            self.world_state.add_agent(agent)


# Create logger instance (customize parameters as needed)
logger = SimulationLogger(
    log_agent_snapshots=True, log_all_encounters=True, snapshot_interval=10
)
