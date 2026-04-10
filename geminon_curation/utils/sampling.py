"""Discrete sampling functions for stat generation."""

import numpy as np


def discrete_gaussian_sample(values, mu, sigma):
    """Sample from discrete values weighted by Gaussian distribution."""
    weights = np.exp(-0.5 * ((values - mu) / sigma) ** 2)
    weights /= weights.sum()
    return np.random.choice(values, p=weights)


def discrete_exponential_sample(values, lam):
    """Sample from discrete values weighted by exponential decay."""
    weights = np.exp(-lam * values)
    weights /= weights.sum()
    return np.random.choice(values, p=weights)


def discrete_uniform_sample(values):
    """Sample uniformly from discrete values."""
    return np.random.choice(values)
