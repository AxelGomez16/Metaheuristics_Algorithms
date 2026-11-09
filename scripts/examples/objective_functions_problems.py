"""Benchmark objective functions shared by optimization examples."""

import math


def rastrigin(position: list[float]) -> float:
  """Rastrigin function."""
  amplitude = 10

  return amplitude * len(position) + sum(
    value**2 - amplitude * math.cos(2 * math.pi * value)
    for value in position
  )


__all__ = ["rastrigin"]
