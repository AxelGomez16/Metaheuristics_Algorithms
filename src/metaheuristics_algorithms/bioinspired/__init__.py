"""Algorithms inspired by biological systems."""

from .grey_wolf_optimizer import (
    GreyWolfOptimizer,
)
from .genetic_algorithm import GeneticAlgorithm

__all__ = ["GeneticAlgorithm", "GreyWolfOptimizer"]
