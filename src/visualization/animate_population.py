import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Just for my pycharm
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


class AnimatePopulation:
  def __init__( self,
                objective_function,
                space_search: list[list[float]],
                grid_resolution: int = 50,
                ):
    self.objective_function = objective_function
    self.space_search = space_search
    self.grid_resolution = grid_resolution
    """
    Parameters:
    - objective_function:   objective function. Receives a list [x, y] and returns a float.
    - space_search:     bounds for each dimension, e.g. [[x_min, x_max], [y_min, y_max]].
    - grid_resolution: grid resolution for the surface (default: 50).
    """

    if len(space_search) != 2:
      raise ValueError("space_search must have exactly two dimensions.")
    for bound in space_search:
      if len(bound) != 2:
        raise ValueError("Each dimension in space_search must have exactly two bounds.")

    # Precompute the surface grid
    self.x_axis, self.y_axis, self.z_axis = self._build_surface()

  # Build the meshgrid + Z values
  def _build_surface(self):
    x_lin = np.linspace(
      self.space_search[0][0],
      self.space_search[0][1],
      self.grid_resolution
    )
    y_lin = np.linspace(
      self.space_search[1][0],
      self.space_search[1][1],
      self.grid_resolution
    )

    z_axis = np.zeros((self.grid_resolution, self.grid_resolution))
    for i in range(self.grid_resolution):
      for j in range(self.grid_resolution):
        z_axis[j, i] = self.objective_function([x_lin[i], y_lin[j]])

    # meshgrid: rows -> y, cols -> x
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    return x_grid, y_grid, z_axis


  def start_population_animation(
    self,
    population_history: list[list[list[float]]],
    fitness_history: list[list[float]] | None = None,
    title: str = "Population Animation",
    delay_interval: int = 200
  ) -> FuncAnimation:
    """Animate a 2D population history over the objective function surface"""

    if not population_history:
      raise ValueError("population_history cannot be empty.")
    
    populations = [
      np.asarray(population, dtype=float) for population in population_history
    ]
    agent_count = populations[0].shape[0]
    if agent_count == 0 or any(population.shape != (agent_count, 2)
                                for population in populations):
      raise ValueError(
        "each population must have the same nonzero shape (N, 2)")

    if fitness_history is None:
      fitness_values = [
        np.asarray([self.objective_function(point) for point in population])
        for population in populations
      ]
    else:
      fitness_values = [
        np.asarray(values, dtype=float) for values in fitness_history
      ]
      if (len(fitness_values) != len(populations)
          or any(values.shape != (agent_count, )
                  for values in fitness_values)):
        raise ValueError("fitness_history must match the population history")

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, agent_count))
    figure = plt.figure(figsize=(12, 6))
    surface_axis = figure.add_subplot(121, projection="3d")
    surface_axis.plot_surface(
      self.x_axis,
      self.y_axis,
      self.z_axis,
      cmap="viridis",
      alpha=0.65
    )
    initial_positions = populations[0]
    initial_fitness = fitness_values[0]
    initial_best_index = int(np.argmin(initial_fitness))

    surface_points = surface_axis.scatter(
      initial_positions[:, 0],
      initial_positions[:, 1],
      initial_fitness,
      c=colors,
      s=35
    )
    surface_best = surface_axis.scatter(
      [initial_positions[initial_best_index, 0]],
      [initial_positions[initial_best_index, 1]],
      [initial_fitness[initial_best_index]],
      c="gold",
      marker="*",
      s=180,
      edgecolor="black")
    surface_axis.set_xlabel("x_1")
    surface_axis.set_ylabel("x_2")
    surface_axis.set_zlabel("f(x)")
    surface_axis.set_title("3D Surface Plot")

    contour_axis = figure.add_subplot(122)
    contour_axis.contour(
      self.x_axis,
      self.y_axis,
      self.z_axis,
      levels=20,
      cmap="viridis",
      zorder=1
    )
    contour_points = contour_axis.scatter(
      initial_positions[:, 0],
      initial_positions[:, 1],
      c=colors,
      s=35,
      zorder=2
    )
    contour_best = contour_axis.scatter(
      [initial_positions[initial_best_index, 0]],
      [initial_positions[initial_best_index, 1]],
      c="gold",
      marker="*",
      s=180,
      edgecolor="black",
      zorder=3
    )
    contour_axis.set_xlabel("x_1")
    contour_axis.set_ylabel("x_2")
    contour_axis.set_title("2D Contour Plot")
    contour_axis.set_aspect("equal", adjustable="box")

    figure.suptitle(title, fontsize=14, y=0.99)
    subtitle = figure.text(0.5, 0.95, "", ha="center", va="top", fontsize=10)

    def update(frame: int):
      positions = populations[frame]
      scores = fitness_values[frame]
      best_index = int(np.argmin(scores))
      surface_points._offsets3d = (positions[:, 0], positions[:, 1], scores)
      surface_best._offsets3d = (
        [positions[best_index, 0]], [positions[best_index, 1]], [scores[best_index]])
      contour_points.set_offsets(positions)
      contour_best.set_offsets(positions[[best_index]])

      subtitle.set_text(f"Iteration {frame}/{len(populations) - 1}"
                        f" | Best fitness: {scores[best_index]:.6g}")
      return surface_points, surface_best, contour_points, contour_best

    animation = FuncAnimation(figure,
                              update,
                              frames=len(populations),
                              interval=delay_interval,
                              repeat=False,
                              blit=False)
    figure.tight_layout(rect=(0, 0.02, 1, 0.90))
    plt.show()

    return animation