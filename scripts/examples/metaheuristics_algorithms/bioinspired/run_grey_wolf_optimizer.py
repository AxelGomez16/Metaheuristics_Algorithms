"""Run a small Grey Wolf Optimizer experiment."""

from metaheuristics_algorithms.bioinspired import GreyWolfOptimizer
from scripts.examples.objective_functions_problems import rastrigin


def main() -> None:
  space_search = [[-5.12, 5.12], [-5.12, 5.12]]

  gwo_optimizer = GreyWolfOptimizer(
    fitness_function=rastrigin,
    space_search=space_search,
    agents_number=20,
    max_iterations=100,
    mode="minimize",
  ).start()

  print(f"Best position: {gwo_optimizer.best_position}")
  print(f"Best fitness: {gwo_optimizer.best_fitness}")

  # gwo_optimizer.visualize_movement(delay_interval=250)
  # gwo_optimizer.visualize_movement_html()
  gwo_optimizer.visualize_charts()



if __name__ == "__main__":
    main()
