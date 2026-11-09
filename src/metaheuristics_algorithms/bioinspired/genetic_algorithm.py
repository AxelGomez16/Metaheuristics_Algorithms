"""Genetic Algorithm (GA)"""

import random
from typing import Callable, TypeAlias

Position: TypeAlias = list[float]
ObjectiveFunction: TypeAlias = Callable[[Position], float]


class GeneticAlgorithm:

  def __init__(
    self,
    fitness_function: ObjectiveFunction,
    space_search: list[list[float]],
    mode: str = "minimize",
    population_size: int = 20,
    max_iterations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    seed: int | None = None,
  ) -> None:
    """Initialize a continuous genetic algorithm.

    Args:
      - fitness_function: Objective function evaluated for each position.
      - space_search: Inclusive [lower, upper] bounds for each dimension.
      - mode: Whether to minimize or maximize the objective function.
      - population_size: Number of individuals in each generation.
      - max_iterations: Maximum number of generations to evaluate.
      - crossover_rate: Probability of applying crossover to two parents.
      - mutation_rate: Probability of mutating each dimension of a child.
      - mutation_scale: Mutation magnitude relative to each dimension's range.
      - seed: Optional seed for reproducible random results.
    """
    if not space_search:
      raise ValueError("space_search must contain at least one dimension")

    if any(len(bounds) != 2 or bounds[0] > bounds[1] for bounds in space_search):
      raise ValueError("each search-space bound must be [lower, upper]")

    if mode not in {"minimize", "maximize"}:
      raise ValueError("mode must be either 'minimize' or 'maximize'")

    if population_size < 3:
      raise ValueError("population_size must be at least 2")

    if max_iterations < 1:
      raise ValueError("max_iterations must be at least 1")

    if not 0 <= crossover_rate <= 1:
      raise ValueError("crossover_rate must be between 0 and 1")

    if not 0 <= mutation_rate <= 1:
      raise ValueError("mutation_rate must be between 0 and 1")

    if mutation_scale < 0:
      raise ValueError("mutation_scale must be non-negative")

    self.fitness_function = fitness_function
    self.space_search = [list(bounds) for bounds in space_search]
    self.dimension = len(space_search)
    self.mode = mode
    self.population_size = population_size
    self.max_iterations = max_iterations
    self.crossover_rate = crossover_rate
    self.mutation_rate = mutation_rate
    self.mutation_scale = mutation_scale
    self._random = random.Random(seed)

    self.best_position: Position | None = None
    self.best_fitness = self._initial_best_fitness()
    self.fitness_history: list[float] = []
    self.population_history: list[list[Position]] = []
    self.population_fitness_history: list[list[float]] = []
    self.iterations = 0

  @property
  def is_maximizing(self) -> bool:
    return self.mode == "maximize"

  def _initial_best_fitness(self) -> float:
    return float("-inf") if self.is_maximizing else float("inf")

  def _is_better(self, candidate: float, current: float) -> bool:
    return candidate > current if self.is_maximizing else candidate < current

  def _random_position(self) -> Position:
    return [
      self._random.uniform(lower, upper)
      for lower, upper in self.space_search
    ]

  def _clip_position(self, position: Position) -> Position:
    clipped_position = []
    for value, (lower, upper) in zip(position, self.space_search):
      if value < lower:
        clipped_value = lower
      elif value > upper:
        clipped_value = upper
      else:
        clipped_value = value
      clipped_position.append(clipped_value)
    return clipped_position

  def _select_parent_tournament(
    self, population: list[Position], fitness: list[float]
  ) -> Position:
    candidates = self._random.sample(range(len(population)), 3)
    if self.is_maximizing:
      best_id = max(candidates, key=fitness.__getitem__)
    else:
      best_id = min(candidates, key=fitness.__getitem__)
    return population[best_id]



  def _intermediate_arithmetic_crossover(
    self, first: Position, second: Position
  ) -> Position:
    """Create a child by interpolating between two parent positions."""
    child = []
    for first_value, second_value in zip(first, second):
      interpolation_factor = self._random.random()
      difference = second_value - first_value
      child_value = first_value + interpolation_factor * difference
      child.append(child_value)
    return child

  def _gaussian_mutation(self, child: Position) -> Position:
    """Apply independent Gaussian mutation to each child dimension."""
    for dimension, (lower, upper) in enumerate(self.space_search):
      random_value = self._random.random()
      if random_value <= self.mutation_rate:
        spread = (upper - lower) * self.mutation_scale
        mutation = self._random.gauss(0.0, spread)
        child[dimension] += mutation
    return child



  def _create_child(self, first: Position, second: Position) -> Position:
    if self._random.random() <= self.crossover_rate:
      child = self._intermediate_arithmetic_crossover(first, second)
    else:
      child = first.copy()

    child = self._gaussian_mutation(child)
    
    return self._clip_position(child)




  def start(self) -> "GeneticAlgorithm":
    """Run the genetic algorithm"""

    self.best_position = None
    self.best_fitness = self._initial_best_fitness()
    self.fitness_history = []
    self.population_history = []
    self.population_fitness_history = []
    self.iterations = 0

    population = [
      self._random_position() for _ in range(self.population_size)
    ]
    fitness = [self.fitness_function(position) for position in population]

    for iteration in range(self.max_iterations):
      best_id = (
        max(range(self.population_size), key=fitness.__getitem__)
        if self.is_maximizing
        else min(range(self.population_size), key=fitness.__getitem__)
      )
      if self._is_better(fitness[best_id], self.best_fitness):
        self.best_fitness = fitness[best_id]
        self.best_position = population[best_id].copy()

      self.fitness_history.append(self.best_fitness)
      self.population_history.append([position.copy() for position in population])
      self.population_fitness_history.append(fitness.copy())
      self.iterations = iteration + 1

      if iteration == self.max_iterations - 1:
        break

      next_population = [population[best_id].copy()]
      while len(next_population) < self.population_size:
        first = self._select_parent_tournament(population, fitness)
        second = self._select_parent_tournament(population, fitness)
        next_population.append(self._create_child(first, second))
      population = next_population
      fitness = [self.fitness_function(position) for position in population]

    if self.best_position is None:
      raise RuntimeError("Genetic Algorithm did not produce a best position")
    return self


