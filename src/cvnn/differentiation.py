"""Module for holding various complex differentiation tools"""

from typing import Callable

import numpy as np


def numeric_diff(cv_func: Callable, z_nom: np.complex128, dz_mag=1e-6, dz_args=1000):
    sample_angles = np.linspace(0, 2 * np.pi, dz_args, endpoint=False)
    offsets = dz_mag * (np.cos(sample_angles) + 1j * np.sin(sample_angles))
    samples = offsets + z_nom

    nominal_output = cv_func(z_nom)
    perturbed_output = cv_func(samples)

    return (perturbed_output - nominal_output) / offsets


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def example_func(z):
        return z**0.5

    numeric_derivs = numeric_diff(example_func, 1 + 1j)
    print(numeric_derivs)
