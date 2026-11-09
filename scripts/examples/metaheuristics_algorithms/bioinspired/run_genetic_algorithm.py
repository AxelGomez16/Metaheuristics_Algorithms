"""Run a small Genetic Algorithm experiment."""

from metaheuristics_algorithms.bioinspired import GeneticAlgorithm
from scripts.examples.objective_functions_problems import rastrigin


def main() -> None:
  space_search = [[-5.12, 5.12], [-5.12, 5.12]]

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


if __name__ == "__main__":
  main()
