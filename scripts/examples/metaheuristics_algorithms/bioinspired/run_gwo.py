"""Run a small Grey Wolf Optimizer experiment."""

import math

from metaheuristics_algorithms.bioinspired import GreyWolfOptimizer

def rastrigin(position: list[float]) -> float:
  A = 10
  return A * len(position) + sum(
    value**2 - A * math.cos(2 * math.pi * value) for value in position
  )


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

  gwo_optimizer.visualize_movement(delay_interval=250)
  gwo_optimizer.visualize_movement_html()
  gwo_optimizer.visualize_charts()



if __name__ == "__main__":
    main()
