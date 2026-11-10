from pathlib import Path

import matplotlib
matplotlib.use('TkAgg') # Just for my pycharm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


plt.rcParams.update({
  "pgf.texsystem": "pdflatex",
  "font.family": "serif",
  "text.usetex": True,
  "pgf.rcfonts": False,
})


def _save_figure(
  figure: Figure,
  output_dir: str | Path | None,
  filename: str,
) -> None:
  if output_dir is not None and Path(output_dir).is_dir():
    for extension in ("pgf", "pdf", "png"):
      figure.savefig(Path(output_dir) / f"{filename}.{extension}", dpi=300)


def plot_single_convergence_by_iteration(
  fitness_history: list[float],
  title: str = "Optimization Convergence",
  logarithmic_scale: bool = False,
  output_dir: str | Path | None = None,
) -> Figure:
  """Display convergence and export PGF, PDF, and PNG if output_dir exists."""

  if len(fitness_history) == 0:
    raise ValueError("fitness_history cannot be empty.")

  if logarithmic_scale and any(fitness <= 0 for fitness in fitness_history):
    raise ValueError(
      "logarithmic_scale requires all fitness values to be positive."
    )

  iterations = range(1, len(fitness_history) + 1)
  figure, axis = plt.subplots(figsize=(7, 4.5))
  axis.plot(
    iterations,
    fitness_history,
    color="blue",
    linewidth=1.5,
  )
  axis.set_title(title)
  axis.set_xlabel("Iteration")
  axis.set_ylabel("Best fitness")
  if logarithmic_scale:
    axis.set_yscale("log")
  axis.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.6)
  figure.tight_layout()
  _save_figure(figure, output_dir, "single_convergence")
  plt.show()

  return figure



def plot_multiple_convergence_by_iteration(
  fitness_histories: list[list[float]],
  labels: list[str],
  title: str = "Optimization Convergence",
  logarithmic_scale: bool = False,
  output_dir: str | Path | None = None,
) -> Figure:
  """Display run histories and export PGF, PDF, and PNG if output_dir exists."""

  if len(fitness_histories) == 0:
    raise ValueError("fitness_histories cannot be empty.")
  if len(fitness_histories) != len(labels):
    raise ValueError("Each fitness history must have a corresponding label.")

  figure, axis = plt.subplots(figsize=(7, 4.5))
  for fitness_history, label in zip(fitness_histories, labels):
    if len(fitness_history) == 0:
      raise ValueError("Each fitness history cannot be empty.")
    if logarithmic_scale and any(fitness <= 0 for fitness in fitness_history):
      raise ValueError(
        "logarithmic_scale requires all fitness values to be positive."
      )
    iterations = range(1, len(fitness_history) + 1)
    axis.plot(
      iterations,
      fitness_history,
      linewidth=1.5,
      label=label
    )
  axis.set_title(title)
  axis.set_xlabel("Iteration")
  axis.set_ylabel("Best fitness")
  if logarithmic_scale:
    axis.set_yscale("log")
  axis.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.6)
  axis.legend()
  figure.tight_layout()
  _save_figure(figure, output_dir, "multiple_convergence")
  plt.show()

  return figure