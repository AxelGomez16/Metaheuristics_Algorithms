"""Run a small Genetic Algorithm experiment."""

from metaheuristics_algorithms.bioinspired import GeneticAlgorithm
from scripts.examples.objective_functions_problems import rastrigin
from visualization.animate_population import AnimatePopulation
from visualization.plot_convergence_results import (
  plot_single_convergence_by_iteration,
  plot_multiple_convergence_by_iteration
  )


def ga_single_convergence_run() -> None:
  space_search:list[list[float]] = [[-5.12, 5.12], [-5.12, 5.12]]

  genetic_algorithm = GeneticAlgorithm(
    fitness_function=rastrigin,
    space_search=space_search,
    population_size=20,
    max_iterations=100,
    mode="minimize",
    seed=42,
  ).start()

  print(f"Best position: {genetic_algorithm.best_position}")
  print(f"Best fitness: {genetic_algorithm.best_fitness}")

  plot_single_convergence_by_iteration(
    fitness_history=genetic_algorithm.fitness_history,
    title="Genetic Algorithm Convergence",
    logarithmic_scale=True
  )


def ga_multiple_convergence_runs() -> None:
  space_search: list[list[float]] = [[-5.12, 5.12], [-5.12, 5.12]]
  all_fitness_histories: list[list[float]] = []

  for run in range(5):
    genetic_algorithm = GeneticAlgorithm(
      fitness_function=rastrigin,
      space_search=space_search,
      population_size=20,
      max_iterations=100,
      mode="minimize",
      seed=42 + run,
    ).start()

    all_fitness_histories.append(genetic_algorithm.fitness_history)

  plot_multiple_convergence_by_iteration(
    fitness_histories=all_fitness_histories,
    labels=[f"Run {i + 1}" for i in range(5)],
    title="Genetic Algorithm Convergence",
    logarithmic_scale=True
  )


def ga_animate_population() -> None:
  space_search:list[list[float]] = [[-5.12, 5.12], [-5.12, 5.12]]

  genetic_algorithm = GeneticAlgorithm(
    fitness_function=rastrigin,
    space_search=space_search,
    population_size=20,
    max_iterations=100,
    mode="minimize",
    seed=42,
  ).start()

  animator = AnimatePopulation(
    objective_function=rastrigin,
    space_search=space_search,
  )
  animator.start_population_animation(
    population_history=genetic_algorithm.population_history,
    fitness_history=genetic_algorithm.population_fitness_history,
    title="Genetic Algorithm",
  )


if __name__ == "__main__":
  # ga_single_convergence_run()
  # ga_multiple_convergence_runs()
  ga_animate_population() # Only works for 2-dimensional problems