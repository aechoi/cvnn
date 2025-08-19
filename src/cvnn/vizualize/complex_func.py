"""Module for visualizing complex valued functions"""

from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from numpy.typing import ArrayLike


def get_random_complex(shape: tuple[int]):
    """Generate a random complex number from -1-1j to 1+1j"""
    size = np.prod(shape)
    real_rand = np.random.rand(size) * 2 - 1
    imag_rand = np.random.rand(size) * 2 - 1
    rand_array = (real_rand + 1j * imag_rand).reshape(shape)
    return rand_array


def plot_cv_func(
    cv_func: Callable,
    domain: ArrayLike[np.complex128] = (-1 - 1j, 1 + 1j),
    num_samples=100,
) -> None:
    """Plot the output of a complex valued function for a rectangle of inputs

    Args:
        cv_func: a function that takes an array of complex valued inputs
        domain: the bottom left and top right corners of the complex plane that
            the function will be applied to
        num_samples: the number of points to sample along the rectangular grid
    """
    real_dom = np.linspace(*np.real(domain), num_samples)
    imag_dom = np.linspace(*np.imag(domain), num_samples)
    RE, IM = np.meshgrid(real_dom, imag_dom)

    input_grid = RE + 1j * IM
    input_thetas = np.angle(input_grid)
    input_radius = np.abs(input_grid)

    H = (input_thetas % (2 * np.pi)) / (2 * np.pi)
    S = np.ones_like(input_thetas)
    V = input_radius / np.max(input_radius)
    hsv = np.dstack((H, S, V))

    rgb = clr.hsv_to_rgb(hsv)

    output_grid = cv_func(input_grid)

    fig, axs = plt.subplots(ncols=2)
    fig.suptitle("Complex Function Transformation")
    axs[0].scatter(
        input_grid.ravel().real, input_grid.ravel().imag, c=rgb.reshape(-1, 3)
    )
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("Real")
    axs[0].set_ylabel("Imag")
    axs[0].set_title("Inputs")

    axs[1].scatter(
        output_grid.ravel().real, output_grid.ravel().imag, c=rgb.reshape(-1, 3)
    )
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("Real")
    axs[1].set_ylabel("Imag")
    axs[1].set_title("Outputs")

    min_x_lim = min(axs[0].get_xlim()[0], axs[1].get_xlim()[0])
    max_x_lim = max(axs[0].get_xlim()[1], axs[1].get_xlim()[1])
    min_y_lim = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
    max_y_lim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    axs[0].set_xlim([min_x_lim, max_x_lim])
    axs[1].set_xlim([min_x_lim, max_x_lim])
    axs[0].set_ylim([min_y_lim, max_y_lim])
    axs[1].set_ylim([min_y_lim, max_y_lim])

    axs[0].grid(True)
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    def example_func(z: np.complex128) -> np.complex128:
        return (z + 1) ** 2

    plot_cv_func(example_func)
