from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AgentSnapshot:
    """Snapshot of an agent's state at a specific time"""

    agent_id: int
    age: float
    position: Tuple[int, int]
    attractiveness: float
    education: str
    income: float
    fitness: float
    selectivity: Tuple[float, float]
    offspring_count: int
    is_alive: bool
    gender: str  # 'M' or 'F'

    # Behavioral metrics
    partners_encountered: int
    successful_matings: int
    rejection_count: int

    # For detailed analysis
    recent_positions: List[Tuple[int, int]] = field(default_factory=list)
    partner_preferences: List[float] = field(default_factory=list)


@dataclass
class EncounterEvent:
    """Records details of agent encounters"""

    timestep: int
    male_id: int
    female_id: int
    position: Tuple[int, int]

    # Agent states at encounter
    male_age: float
    female_age: float
    male_attractiveness: float
    female_attractiveness: float
    male_income: float
    female_income: float
    male_education: str
    female_education: str

    # Encounter outcome
    mutual_attraction: bool
    reproduction_attempted: bool
    reproduction_successful: bool

    # Behavioral analysis
    male_rating_of_female: float
    female_rating_of_male: float
    compatibility_score: float


@dataclass
class PopulationMetrics:
    """Population-level statistics at each timestep"""

    timestep: int
    total_population: int
    male_count: int
    female_count: int

    # Age demographics
    avg_age: float
    age_distribution: Dict[str, int]  # '18-25', '26-35', etc.

    # Socioeconomic metrics
    avg_income: float
    income_inequality_gini: float
    education_distribution: Dict[str, int]

    # Mating market dynamics
    single_males: int
    single_females: int
    avg_male_selectivity: float
    avg_female_selectivity: float

    # Spatial distribution
    population_density_map: Dict[Tuple[int, int], int]
    clustering_coefficient: float


@dataclass
class ReproductionEvent:
    """Details of successful reproductions"""

    timestep: int
    male_id: int
    female_id: int
    child_id: int
    child_gender: str

    # Parent characteristics
    male_age: float
    female_age: float
    age_gap: float

    # Socioeconomic mixing
    male_income: float
    female_income: float
    income_ratio: float

    male_education: str
    female_education: str
    education_match: bool

    # Genetic/trait inheritance
    male_attractiveness: float
    female_attractiveness: float
    child_attractiveness: float

    # Market dynamics
    male_fitness: float
    female_fitness: float
    assortative_mating_score: float  # How similar are the parents


class SimulationLogger:
    """Comprehensive logging system for the simulation"""

    def __init__(
        self,
        log_agent_snapshots: bool = True,
        log_all_encounters: bool = False,
        snapshot_interval: int = 10,
    ):
        """
        Args:
            log_agent_snapshots: Whether to log detailed agent states
            log_all_encounters: Whether to log all encounters (can be memory intensive)
            snapshot_interval: How often to take agent snapshots
        """
        self.log_agent_snapshots = log_agent_snapshots
        self.log_all_encounters = log_all_encounters
        self.snapshot_interval = snapshot_interval

        # Data storage
        self.population_metrics: List[PopulationMetrics] = []
        self.encounter_events: List[EncounterEvent] = []
        self.reproduction_events: List[ReproductionEvent] = []
        self.agent_snapshots: Dict[int, List[AgentSnapshot]] = defaultdict(list)

        # Tracking state
        self.agent_encounter_counts: Dict[int, int] = defaultdict(int)
        self.agent_rejection_counts: Dict[int, int] = defaultdict(int)
        self.agent_success_counts: Dict[int, int] = defaultdict(int)

        # Behavioral tracking
        self.movement_patterns: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.preference_evolution: Dict[int, List[float]] = defaultdict(list)

    def log_timestep(
        self,
        world_state: "WorldState",
        encounters: List[EncounterEvent] = None,
        reproductions: List[ReproductionEvent] = None,
    ):
        """Log data for a complete timestep"""

        timestep = world_state.time_step

        # 1. Population metrics
        pop_metrics = self._calculate_population_metrics(world_state)
        self.population_metrics.append(pop_metrics)

        # 2. Store encounters if provided
        if encounters:
            self.encounter_events.extend(encounters)

        # 3. Store reproductions if provided
        if reproductions:
            self.reproduction_events.extend(reproductions)

        # 4. Agent snapshots (periodic)
        if self.log_agent_snapshots and timestep % self.snapshot_interval == 0:
            self._take_agent_snapshots(world_state)

    def log_encounter(
        self,
        male: "Agent",
        female: "Agent",
        world_state: "WorldState",
        reproduction_result: Optional["Agent"] = None,
    ):
        """Log a specific encounter between two agents"""

        # Calculate ratings
        rating_1_of_2 = male.mating_behavior.rate_partner(male, female)
        rating_2_of_1 = female.mating_behavior.rate_partner(female, male)

        # Check compatibility
        compat_1_2 = male.mating_behavior.is_compatible(male, female)
        compat_2_1 = female.mating_behavior.is_compatible(female, male)
        mutual_attraction = compat_1_2 and compat_2_1

        # Compatibility score (geometric mean of ratings)
        compatibility_score = np.sqrt(rating_1_of_2 * rating_2_of_1)

        encounter = EncounterEvent(
            timestep=world_state.time_step,
            male_id=male.id,
            female_id=female.id,
            position=male.position,
            male_age=male.age,
            female_age=female.age,
            male_attractiveness=male.attractiveness,
            female_attractiveness=female.attractiveness,
            male_income=male.income,
            female_income=female.income,
            male_education=male.education,
            female_education=female.education,
            mutual_attraction=mutual_attraction,
            reproduction_attempted=mutual_attraction,
            reproduction_successful=reproduction_result is not None,
            male_rating_of_female=rating_1_of_2,
            female_rating_of_male=rating_2_of_1,
            compatibility_score=compatibility_score,
        )

        # Update tracking
        self.agent_encounter_counts[male.id] += 1
        self.agent_encounter_counts[female.id] += 1

        if not mutual_attraction:
            if not compat_1_2:
                self.agent_rejection_counts[male.id] += 1
            if not compat_2_1:
                self.agent_rejection_counts[female.id] += 1
        else:
            if reproduction_result:
                self.agent_success_counts[male.id] += 1
                self.agent_success_counts[female.id] += 1

        if self.log_all_encounters:
            self.encounter_events.append(encounter)

        return encounter

    def log_reproduction(
        self,
        male: "Agent",
        female: "Agent",
        child: "Agent",
        world_state: "WorldState",
    ):
        """Log a successful reproduction event"""

        age_gap = abs(male.age - female.age)
        income_ratio = max(male.income, female.income) / (
            min(male.income, female.income) + 1e-6
        )
        education_match = male.education == female.education

        # Assortative mating score (how similar are the parents)
        attractiveness_diff = abs(male.attractiveness - female.attractiveness)
        fitness_diff = abs(male.fitness - female.fitness)
        assortative_score = 1 - (attractiveness_diff + fitness_diff) / 2

        reproduction = ReproductionEvent(
            timestep=world_state.time_step,
            male_id=male.id,
            female_id=female.id,
            child_id=child.id,
            child_gender="M" if hasattr(child, "sex") and child.sex == "M" else "F",
            male_age=male.age,
            female_age=female.age,
            age_gap=age_gap,
            male_income=male.income,
            female_income=female.income,
            income_ratio=income_ratio,
            male_education=male.education,
            female_education=female.education,
            education_match=education_match,
            male_attractiveness=male.attractiveness,
            female_attractiveness=female.attractiveness,
            child_attractiveness=child.attractiveness,
            male_fitness=male.fitness,
            female_fitness=female.fitness,
            assortative_mating_score=assortative_score,
        )

        self.reproduction_events.append(reproduction)
        return reproduction

    def _calculate_population_metrics(
        self, world_state: "WorldState"
    ) -> PopulationMetrics:
        """Calculate comprehensive population statistics"""

        agents = list(world_state.agents.values())

        if not agents:
            return PopulationMetrics(
                timestep=world_state.time_step,
                total_population=0,
                male_count=0,
                female_count=0,
                avg_age=0,
                age_distribution={},
                avg_income=0,
                income_inequality_gini=0,
                education_distribution={},
                single_males=0,
                single_females=0,
                avg_male_selectivity=0,
                avg_female_selectivity=0,
                population_density_map={},
                clustering_coefficient=0,
            )

        # Basic counts
        male_agents = [a for a in agents if hasattr(a, "sex") and a.sex == "M"]
        female_agents = [a for a in agents if hasattr(a, "sex") and a.sex == "F"]

        # Age analysis
        ages = [a.age for a in agents]
        age_bins = {
            "18-25": len([a for a in ages if 18 <= a < 26]),
            "26-35": len([a for a in ages if 26 <= a < 36]),
            "36-45": len([a for a in ages if 36 <= a < 46]),
            "46+": len([a for a in ages if a >= 46]),
        }

        # Income analysis
        incomes = [a.income for a in agents]
        gini = self._calculate_gini(incomes)

        # Education distribution
        education_dist = defaultdict(int)
        for agent in agents:
            education_dist[agent.education] += 1

        # Mating market metrics
        single_males = len([a for a in male_agents if a.offspring_count == 0])
        single_females = len([a for a in female_agents if a.offspring_count == 0])

        # Selectivity
        male_selectivity = (
            np.mean([a.selectivity[1] - a.selectivity[0] for a in male_agents])
            if male_agents
            else 0
        )
        female_selectivity = (
            np.mean([a.selectivity[1] - a.selectivity[0] for a in female_agents])
            if female_agents
            else 0
        )

        # Spatial analysis
        density_map = defaultdict(int)
        for agent in agents:
            density_map[agent.position] += 1

        # Simple clustering coefficient
        clustering = self._calculate_clustering(world_state)

        return PopulationMetrics(
            timestep=world_state.time_step,
            total_population=len(agents),
            male_count=len(male_agents),
            female_count=len(female_agents),
            avg_age=np.mean(ages),
            age_distribution=dict(age_bins),
            avg_income=np.mean(incomes),
            income_inequality_gini=gini,
            education_distribution=dict(education_dist),
            single_males=single_males,
            single_females=single_females,
            avg_male_selectivity=male_selectivity,
            avg_female_selectivity=female_selectivity,
            population_density_map=dict(density_map),
            clustering_coefficient=clustering,
        )

    def _take_agent_snapshots(self, world_state: "WorldState"):
        """Take detailed snapshots of all agents"""

        for agent in world_state.agents.values():
            # Track movement
            if agent.id in self.movement_patterns:
                recent_positions = self.movement_patterns[agent.id][
                    -10:
                ]  # Last 10 positions
            else:
                recent_positions = []

            snapshot = AgentSnapshot(
                agent_id=agent.id,
                age=agent.age,
                position=agent.position,
                attractiveness=agent.attractiveness,
                education=agent.education,
                income=agent.income,
                fitness=agent.fitness,
                selectivity=agent.selectivity,
                offspring_count=agent.offspring_count,
                is_alive=agent.is_alive,
                gender="M" if hasattr(agent, "sex") and agent.sex == "M" else "F",
                partners_encountered=self.agent_encounter_counts[agent.id],
                successful_matings=agent.offspring_count,
                rejection_count=self.agent_rejection_counts[agent.id],
                recent_positions=recent_positions,
                partner_preferences=self.preference_evolution[agent.id].copy(),
            )

            self.agent_snapshots[agent.id].append(snapshot)

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values:
            return 0

        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)

        return (
            n
            + 1
            - 2
            * sum([(n + 1 - i) * y for i, y in enumerate(sorted_values, 1)])
            / cumsum[-1]
        ) / n

    def _calculate_clustering(self, world_state: "WorldState") -> float:
        """Simple spatial clustering coefficient"""
        if not world_state.agents:
            return 0

        # Count agents with neighbors
        agents_with_neighbors = 0
        for pos, agent_ids in world_state.spatial_index.items():
            if len(agent_ids) > 1:
                agents_with_neighbors += len(agent_ids)

        return agents_with_neighbors / len(world_state.agents)

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis summary"""

        if not self.population_metrics:
            return {"status": "No data logged"}

        # Population trends
        initial_pop = self.population_metrics[0].total_population
        final_pop = self.population_metrics[-1].total_population
        growth_rate = (final_pop - initial_pop) / initial_pop if initial_pop > 0 else 0

        # Mating success rates
        total_encounters = len(self.encounter_events)
        successful_matings = len(self.reproduction_events)
        success_rate = (
            successful_matings / total_encounters if total_encounters > 0 else 0
        )

        # Inequality trends
        gini_trend = [m.income_inequality_gini for m in self.population_metrics]

        # Age gap analysis
        age_gaps = [r.age_gap for r in self.reproduction_events]

        # Assortative mating
        assortative_scores = [
            r.assortative_mating_score for r in self.reproduction_events
        ]

        return {
            "simulation_length": len(self.population_metrics),
            "population_growth_rate": growth_rate,
            "final_population": final_pop,
            "total_encounters": total_encounters,
            "successful_matings": successful_matings,
            "mating_success_rate": success_rate,
            "avg_age_gap": np.mean(age_gaps) if age_gaps else 0,
            "avg_assortative_mating": np.mean(assortative_scores)
            if assortative_scores
            else 0,
            "income_inequality_trend": {
                "initial": gini_trend[0] if gini_trend else 0,
                "final": gini_trend[-1] if gini_trend else 0,
                "peak": max(gini_trend) if gini_trend else 0,
            },
            "education_mixing": self._analyze_education_mixing(),
            "spatial_dynamics": self._analyze_spatial_patterns(),
        }

    def _analyze_education_mixing(self) -> Dict[str, float]:
        """Analyze education-based assortative mating"""
        if not self.reproduction_events:
            return {}

        same_education = sum(1 for r in self.reproduction_events if r.education_match)
        total = len(self.reproduction_events)

        return {
            "homogamy_rate": same_education / total,
            "heterogamy_rate": (total - same_education) / total,
        }

    def _analyze_spatial_patterns(self) -> Dict[str, Any]:
        """Analyze spatial movement and clustering patterns"""
        if not self.population_metrics:
            return {}

        clustering_over_time = [
            m.clustering_coefficient for m in self.population_metrics
        ]

        return {
            "avg_clustering": np.mean(clustering_over_time),
            "clustering_trend": "increasing"
            if clustering_over_time[-1] > clustering_over_time[0]
            else "decreasing",
        }

    def convert_keys_to_str(self, obj):
        if isinstance(obj, dict):
            return {str(k): self.convert_keys_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_keys_to_str(i) for i in obj]
        else:
            return obj

    def save_data(self, filepath: str):
        """Save all logged data to file"""
        data = {
            "population_metrics": [m.__dict__ for m in self.population_metrics],
            "encounter_events": [e.__dict__ for e in self.encounter_events],
            "reproduction_events": [r.__dict__ for r in self.reproduction_events],
            "agent_snapshots": {
                agent_id: [s.__dict__ for s in snapshots]
                for agent_id, snapshots in self.agent_snapshots.items()
            },
            "summary": self.get_analysis_summary(),
        }

        if filepath.endswith(".json"):
            with open(filepath, "w") as f:
                json.dump(self.convert_keys_to_str(data), f, indent=2, default=str)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

    def load_data(self, filepath: str):
        """Load previously saved data"""
        if filepath.endswith(".json"):
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        # Reconstruct objects from dictionaries
        self.population_metrics = [
            PopulationMetrics(**m) for m in data["population_metrics"]
        ]
        self.encounter_events = [EncounterEvent(**e) for e in data["encounter_events"]]
        self.reproduction_events = [
            ReproductionEvent(**r) for r in data["reproduction_events"]
        ]

        # Reconstruct agent snapshots
        self.agent_snapshots = {}
        for agent_id, snapshots in data["agent_snapshots"].items():
            self.agent_snapshots[int(agent_id)] = [
                AgentSnapshot(**s) for s in snapshots
            ]
