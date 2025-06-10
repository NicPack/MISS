from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from protocols.agents import Female, Male

# =============================================================================
# PROTOCOLS
# =============================================================================


class MovementBehavior(ABC):
    """Abstract base for movement strategies"""

    @abstractmethod
    def move(self, agent: "Agent", world_state: "WorldState") -> Tuple[int, int]:
        pass


class MatingBehavior(ABC):
    """Abstract base for mating strategies"""

    @abstractmethod
    def rate_partner(self, agent: "Agent", potential_partner: "Agent") -> float:
        pass

    @abstractmethod
    def is_compatible(self, agent: "Agent", potential_partner: "Agent") -> bool:
        pass


class ReproductionSystem(ABC):
    """Abstract base for reproduction mechanics"""

    @abstractmethod
    def attempt_reproduction(
        self, parent1: "Agent", parent2: "Agent"
    ) -> Optional["Agent"]:
        pass


# =============================================================================
# CONCRETE BEHAVIOR IMPLEMENTATIONS
# =============================================================================


class SeekingMovement(MovementBehavior):
    """Males seek closest compatible female"""

    def move(self, agent: "Agent", world_state: "WorldState") -> Tuple[int, int]:
        # Find targets (females not yet met)
        targets = self._find_targets(agent, world_state)

        if not targets:
            return self._random_walk(agent, world_state.map_size)

        # Move toward closest target
        closest = min(targets, key=lambda t: self._distance(agent.position, t.position))
        return self._move_toward(agent.position, closest.position, world_state.map_size)

    def _find_targets(self, agent, world_state):
        # This would be implemented based on your specific logic
        return []  # Placeholder

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _move_toward(
        self,
        current_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        map_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        cx, cy = current_pos
        tx, ty = target_pos
        dx, dy = np.sign(tx - cx), np.sign(ty - cy)
        new_x = max(0, min(map_size[0] - 1, cx + dx))
        new_y = max(0, min(map_size[1] - 1, cy + dy))
        return (new_x, new_y)

    def _random_walk(self, agent, map_size):
        cx, cy = agent.position
        dx, dy = np.random.randint(-1, 2), np.random.randint(-1, 2)
        new_x = max(0, min(map_size[0] - 1, cx + dx))
        new_y = max(0, min(map_size[1] - 1, cy + dy))
        return (new_x, new_y)


class AgeFocusedMating(MatingBehavior):
    """Males prefer younger females"""

    def __init__(self, ideal_age_diff: float = 3.0, max_age_diff: float = 12.0):
        self.ideal_age_diff = ideal_age_diff
        self.max_age_diff = max_age_diff

    def rate_partner(self, agent: "Agent", potential_partner: "Agent") -> float:
        age_diff = agent.age - potential_partner.age

        if age_diff < 0 or age_diff > self.max_age_diff:
            age_score = 0.1
        else:
            age_score = max(0, 1 - abs(age_diff - self.ideal_age_diff) / 10)

        return potential_partner.attractiveness * age_score

    def is_compatible(self, agent: "Agent", potential_partner: "Agent") -> bool:
        rating = self.rate_partner(agent, potential_partner)
        return agent.selectivity[0] <= rating <= agent.selectivity[1]


class BasicReproduction(ReproductionSystem):
    """Basic reproduction with genetic inheritance"""

    def __init__(self):
        """
        Args:
            id_generator: Callable that returns next available ID, or None to use timestamp-based IDs
        """
        self.next_id = 0

    def get_next_id(self):
        """Get the next available ID"""
        current_id = self.next_id
        self.next_id += 1
        return current_id

    def initialize_population(
        self, population_size: int = 100
    ):  # Initialize population using sim's ID generator
        population = []
        for i in range(population_size):
            pos = (np.random.randint(0, 50), np.random.randint(0, 50))
            if i < population_size // 2:
                agent = Male(
                    self.get_next_id(),
                    pos,
                    movement_behavior=SeekingMovement(),
                    mating_behavior=AgeFocusedMating(),
                )
            else:
                agent = Female(
                    self.get_next_id(),
                    pos,
                    movement_behavior=RandomWalk(),
                    mating_behavior=ResourceFocusedMating(),
                )

            population.append(agent)
        return population

    def attempt_reproduction(
        self, parent1: "Agent", parent2: "Agent"
    ) -> Optional["Agent"]:
        # Mutual attraction check
        if not (
            parent1.mating_behavior.is_compatible(parent1, parent2)
            and parent2.mating_behavior.is_compatible(parent2, parent1)
        ):
            return None

        # Probability-based matching
        if not self._calculate_match_probability(parent1, parent2):
            return None

        return self._create_offspring(parent1, parent2)

    def _calculate_match_probability(self, parent1: "Agent", parent2: "Agent") -> bool:
        def closeness(attractiveness, selectivity):
            lower, upper = selectivity
            if not lower <= attractiveness <= upper:
                return 0.0
            center = (lower + upper) / 2
            radius = (upper - lower) / 2
            return 1 - abs(attractiveness - center) / radius

        c1 = closeness(parent2.attractiveness, parent1.selectivity)
        c2 = closeness(parent1.attractiveness, parent2.selectivity)

        match_probability = (c1 + c2) / 2
        match_probability = np.clip(match_probability, 0.0, 1.0)  # Ensure within bounds

        return np.random.choice(
            [True, False], p=[match_probability, 1 - match_probability]
        )

    def _create_offspring(self, parent1: "Agent", parent2: "Agent") -> "Agent":
        # Determine gender and traits
        child_gender = np.random.choice(["M", "F"])
        child_attractiveness = np.clip(
            np.random.normal(
                (parent1.attractiveness + parent2.attractiveness) / 2, 0.1
            ),
            0,
            1,
        )

        # Child appears at one of the parents' positions
        child_pos = parent1.position

        # Create appropriate child
        if child_gender == "M":
            return Male(
                self.get_next_id(),
                child_pos,
                attractiveness=child_attractiveness,
                mating_behavior=AgeFocusedMating(),
                movement_behavior=SeekingMovement(),
            )
        else:
            return Female(
                self.get_next_id(),
                child_pos,
                attractiveness=child_attractiveness,
                mating_behavior=ResourceFocusedMating(),
                movement_behavior=RandomWalk(),
            )


class RandomWalk(MovementBehavior):
    """Random movement behavior"""

    def __init__(self, step_probability: float = 0.8):
        self.step_probability = step_probability

    def move(self, agent: "Agent", world_state: "WorldState") -> Tuple[int, int]:
        # Sometimes stay in place
        if np.random.random() > self.step_probability:
            return agent.position

        cx, cy = agent.position
        # Random direction (-1, 0, 1) for each axis
        dx, dy = np.random.randint(-1, 2), np.random.randint(-1, 2)

        # Ensure within bounds
        new_x = max(0, min(world_state.map_size[0] - 1, cx + dx))
        new_y = max(0, min(world_state.map_size[1] - 1, cy + dy))

        return (new_x, new_y)


class ResourceFocusedMating(MatingBehavior):
    """Females focus on resources (income, education) and stability"""

    def __init__(
        self,
        income_weight: float = 0.4,
        education_weight: float = 0.3,
        attractiveness_weight: float = 0.2,
        age_stability_weight: float = 0.1,
    ):
        self.income_weight = income_weight
        self.education_weight = education_weight
        self.attractiveness_weight = attractiveness_weight
        self.age_stability_weight = age_stability_weight

        # Education value mapping
        self.education_values = {"basic": 0.2, "high school": 0.5, "higher": 0.8}

    def rate_partner(self, agent: "Agent", potential_partner: "Agent") -> float:
        # Income score (normalized)
        income_score = min(1.0, potential_partner.income / 100000)  # Cap at 100k

        # Education score
        education_score = self.education_values.get(potential_partner.education, 0.2)

        # Attractiveness score (already 0-1)
        attractiveness_score = potential_partner.attractiveness

        # Age stability score - prefer men slightly older but not too much
        age_diff = potential_partner.age - agent.age
        if -5 <= age_diff <= 8:  # 5 years younger to 8 years older is acceptable
            age_stability_score = 1.0 - abs(age_diff - 2) / 10  # Ideal is 2 years older
        else:
            age_stability_score = 0.1

        age_stability_score = max(0, age_stability_score)

        # Weighted combination
        total_score = (
            income_score * self.income_weight
            + education_score * self.education_weight
            + attractiveness_score * self.attractiveness_weight
            + age_stability_score * self.age_stability_weight
        )

        return np.clip(total_score, 0, 1)

    def is_compatible(self, agent: "Agent", potential_partner: "Agent") -> bool:
        rating = self.rate_partner(agent, potential_partner)
        return agent.selectivity[0] <= rating <= agent.selectivity[1]
