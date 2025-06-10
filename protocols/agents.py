from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

# =============================================================================
# INTERFACES
# =============================================================================


class Agent(Protocol):
    """Protocol for all simulation agents"""

    id: int
    position: Tuple[int, int]
    age: float
    is_alive: bool

    def update(self, world_state: "WorldState") -> None: ...


class Male:
    """Simplified Male class focused on attributes"""

    def __init__(
        self,
        unique_id: int,
        position: Tuple[int, int],
        age: int = None,
        attractiveness: float = None,
        movement_behavior=None,
        mating_behavior=None,
    ):
        self.id = unique_id
        self.position = position
        self.age = age if age is not None else np.random.randint(18, 40)
        self.attractiveness = (
            attractiveness if attractiveness is not None else np.random.uniform(0, 1)
        )
        self.is_alive = True
        self.longevity = 80  # Example
        self.sex = "M"

        # Attributes
        self.education = self._set_education()
        self.income = self._set_income()
        self.fitness = self._calculate_fitness()
        self.selectivity = (self.fitness * 0.9, min(self.fitness * 1.1, 1))

        # Behaviors (injected)
        self.movement_behavior = movement_behavior
        self.mating_behavior = mating_behavior

        # Stats
        self.offspring_count = 0
        self.females_met = set()

    def update(self, world_state: "WorldState"):
        """Update agent state"""
        # Move
        new_position = self.movement_behavior.move(self, world_state)
        self.position = new_position

        # Age
        self.age += 0.1
        self.is_alive = self.age < self.longevity

    def _set_education(self):
        return np.random.choice(
            ["basic", "high school", "higher"], p=[0.075, 0.615, 0.31]
        )

    def _set_income(self):
        income_map = {
            "basic": (35000, 10000),
            "high school": (45000, 15000),
            "higher": (70000, 20000),
        }
        mean, std = income_map[self.education]
        return np.random.normal(mean, std)

    def _calculate_fitness(self):
        degree_mapping = {"basic": 0.2, "high school": 0.5, "higher": 0.8}
        return (
            degree_mapping[self.education]
            + self.attractiveness
            + min(1, self.income / 80000)
        ) / 3


class Female:
    """Female class with resource-focused mating and random walk movement"""

    def __init__(
        self,
        unique_id: int,
        position: Tuple[int, int],
        age: int = None,
        attractiveness: float = None,
        movement_behavior=None,
        mating_behavior=None,
    ):
        self.id = unique_id
        self.position = position
        self.age = age if age is not None else np.random.randint(18, 40)
        self.attractiveness = (
            attractiveness if attractiveness is not None else np.random.uniform(0, 1)
        )
        self.is_alive = True
        self.longevity = 85  # Women typically live longer
        self.sex = "F"

        # Attributes
        self.education = self._set_education()
        self.income = self._set_income()
        self.fitness = self._calculate_fitness()
        # Females are generally more selective
        self.selectivity = (self.fitness * 0.7, min(self.fitness * 1.0, 1))

        # Behaviors (injected) - different from males
        self.movement_behavior = movement_behavior
        self.mating_behavior = mating_behavior

        # Stats
        self.offspring_count = 0
        self.males_met = set()
        self.mating_pairs_ages = []
        self.mating_preferences = []

    def update(self, world_state: "WorldState"):
        """Update agent state"""
        # Move
        new_position = self.movement_behavior.move(self, world_state)
        self.position = new_position

        # Age
        self.age += 0.1
        self.is_alive = self.age < self.longevity

    def _set_education(self):
        # Females have slightly higher education rates in modern times
        return np.random.choice(["basic", "high school", "higher"], p=[0.05, 0.55, 0.4])

    def _set_income(self):
        # Generally lower income due to wage gap, but varies by education
        income_map = {
            "basic": (28000, 8000),
            "high school": (38000, 12000),
            "higher": (60000, 18000),
        }
        mean, std = income_map[self.education]
        return np.random.normal(mean, std)

    def _calculate_fitness(self):
        """Fitness calculation focusing on different factors for females"""
        degree_mapping = {"basic": 0.2, "high school": 0.5, "higher": 0.8}
        # For females, education and attractiveness matter more than income in fitness
        return (
            degree_mapping[self.education] * 1.2
            + self.attractiveness * 1.3
            + min(1, self.income / 80000) * 0.5
        ) / 3
