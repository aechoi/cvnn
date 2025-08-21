"""Module for visualizing complex valued functions"""

from dataclasses import dataclass
from typing import Callable

from matplotlib import cm
import matplotlib.colors as clr
import matplotlib.patches as pch
import matplotlib.pyplot as plt
import numpy as np


def get_random_complex(shape: tuple[int]):
    """Generate a random complex number from -1-1j to 1+1j"""
    size = np.prod(shape)
    real_rand = np.random.rand(size) * 2 - 1
    imag_rand = np.random.rand(size) * 2 - 1
    rand_array = (real_rand + 1j * imag_rand).reshape(shape)
    return rand_array


def get_complex_grid(domain: tuple[np.complex128], num_samples) -> np.ndarray:
    real_dom = np.linspace(*np.real(domain), num_samples)
    imag_dom = np.linspace(*np.imag(domain), num_samples)
    RE, IM = np.meshgrid(real_dom, imag_dom)
    return RE + 1j * IM


def complex_dot(z1, z2):
    return np.real(z1) * np.real(z2) + np.imag(z1) * np.imag(z2)


def get_complex_hsv(array: np.ndarray, scale: float):
    """Return HSV values with the angle corresponding to hue, value with
    magnitude, and saturation at 1."""
    thetas = np.angle(array)
    radius = np.abs(array)

    H = (thetas % (2 * np.pi)) / (2 * np.pi)
    S = np.ones_like(thetas)
    V = (radius / scale) ** 0.25
    hsv = np.stack((H, S, V), axis=-1)
    return hsv


def plot_colormap(
    cv_func: Callable,
    domain: tuple[np.complex128] = (-1 - 1j, 1 + 1j),
    num_samples=20,
) -> None:
    """Plot the resulting phase and magnitude of the output function using an
    HSV color scale where hue maps to phase and value maps to magnitude.

    Args:
        cv_func: a function that takes an array of complex valued inputs
        domain: the bottom left and top right corners of the complex plane that
            the function will be applied to
        num_samples: the number of points to sample along the rectangular grid
    """
    input_grid = get_complex_grid(domain, num_samples)
    output_grid = cv_func(input_grid)

    scale = max(np.max(np.abs(output_grid)), np.max(np.abs(input_grid)))
    input_hsv = get_complex_hsv(input_grid, scale)
    output_hsv = get_complex_hsv(output_grid, scale)

    fig, axs = plt.subplots(ncols=2)
    fig.suptitle("Phase-Hue Mag-Value")
    axs[0].imshow(
        clr.hsv_to_rgb(input_hsv),
        origin="lower",
        aspect="equal",
        extent=(*np.real(domain), *np.imag(domain)),
    )
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("Real Input")
    axs[0].set_ylabel("Imag Input")
    axs[0].set_title("Input Space")

    axs[1].imshow(
        clr.hsv_to_rgb(output_hsv),
        origin="lower",
        aspect="equal",
        extent=(*np.real(domain), *np.imag(domain)),
    )
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("Real Input")
    axs[1].set_ylabel("Imag Input")
    axs[1].set_title("Output Space")

    axs[0].grid(True)
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_3d_colormap(
    cv_func: Callable, domain: tuple[np.complex128], max_mag=None, num_samples=20
):
    input_grid = get_complex_grid(domain, num_samples)
    output_grid = cv_func(input_grid)

    scale = max(np.max(np.abs(output_grid)), np.max(np.abs(input_grid)))
    input_hsv = get_complex_hsv(input_grid, scale)
    output_hsv = get_complex_hsv(output_grid, scale)

    fig = plt.figure()
    fig.suptitle("Phase-Hue Mag-Value/Height")
    ax_scale = fig.add_subplot(1, 2, 1)
    ax_output = fig.add_subplot(1, 2, 2, projection="3d")

    ax_scale.imshow(
        clr.hsv_to_rgb(input_hsv),
        origin="lower",
        aspect="equal",
        extent=(*np.real(domain), *np.imag(domain)),
    )
    ax_scale.set_aspect("equal")
    ax_scale.set_xlabel("Real Input")
    ax_scale.set_ylabel("Imag Input")
    ax_scale.set_title("Input Space")

    ax_output.plot_surface(
        np.real(input_grid),
        np.imag(input_grid),
        np.abs(output_grid),
        facecolors=clr.hsv_to_rgb(output_hsv),
    )
    ax_output.set_aspect("equal")
    ax_output.set_xlabel("Real Input")
    ax_output.set_ylabel("Imag Input")
    ax_output.set_zlabel("Magnitude Output")
    ax_output.set_title("Output")

    if max_mag is not None:
        ax_output.set_zlim(0, max_mag)

    # Add snaps
    targets = [(90, -90)]  # XY  # XZ  # YZ

    def snap_view(event):
        if event.name == "button_release_event":
            elev, azim = ax_output.elev, ax_output.azim
            for te, ta in targets:
                if abs(elev - te) < 10:
                    ax_output.view_init(te, ta)
                    ax_output.set_proj_type("ortho")
                    fig.canvas.draw_idle()
                    break
        if (
            event.name == "button_press_event"
            and event.inaxes == ax_output
            and event.button == 1
        ):
            ax_output.set_proj_type("persp")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_release_event", snap_view)
    fig.canvas.mpl_connect("button_press_event", snap_view)

    plt.tight_layout()
    plt.show()


def plot_real_imag(
    cv_func: Callable, domain: tuple[np.complex128] = (-1 - 1j, 1 + 1j), num_samples=20
):
    input_grid = get_complex_grid(domain, num_samples)
    output_grid = cv_func(input_grid)

    vmin = min(np.real(output_grid).min(), np.imag(output_grid).min())
    vmax = max(np.real(output_grid).max(), np.imag(output_grid).max())
    norm = clr.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.coolwarm

    fig, axs = plt.subplots(ncols=2, subplot_kw={"projection": "3d"})
    fig.suptitle("Separate Real and Imaginary Surfaces")
    axs[0].plot_surface(
        np.real(input_grid),
        np.imag(input_grid),
        np.real(output_grid),
        facecolors=cmap(norm(np.real(output_grid))),
    )
    axs[0].set_xlabel("Real Input")
    axs[0].set_ylabel("Imag Input")
    axs[0].set_zlabel("Real Output")
    axs[0].set_title("Real Component")

    axs[1].plot_surface(
        np.real(input_grid),
        np.imag(input_grid),
        np.imag(output_grid),
        facecolors=cmap(norm(np.imag(output_grid))),
    )
    axs[1].set_xlabel("Real Input")
    axs[1].set_ylabel("Imag Input")
    axs[1].set_zlabel("Imag Output")
    axs[1].set_title("Imag Component")

    axs[0].set_zlim([vmin, vmax])
    axs[1].set_zlim([vmin, vmax])

    def sync_views(event):
        if event.inaxes == axs[0]:
            elev, azim, roll = axs[0].elev, axs[0].azim, axs[0].roll
            axs[1].view_init(elev, azim, roll)
            fig.canvas.draw_idle()
        elif event.inaxes == axs[1]:
            elev, azim, roll = axs[1].elev, axs[1].azim, axs[1].roll
            axs[0].view_init(elev, azim, roll)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", sync_views)
    plt.show()


@dataclass
class ViewSegment:
    cross_width: float = 5
    view_direction: np.complex128 = 1j
    travel: float = 0
    num_samples: int = 100

    def __post_init__(self):
        self.colors = np.vstack(
            (
                np.linspace(0, 1, self.num_samples),
                np.zeros(self.num_samples),
                np.zeros(self.num_samples),
            )
        ).T

    @property
    def norm_view_dir(self):
        return self.view_direction / np.abs(self.view_direction)

    @property
    def center(self):
        return self.norm_view_dir * self.travel

    @property
    def ax_direction(self):
        return self.norm_view_dir * -1j

    @property
    def end_points(self):
        return (
            np.array([-1, 1]) * self.ax_direction * self.cross_width / 2 + self.center
        )

    @property
    def samples(self):
        return np.linspace(*self.end_points, self.num_samples)


def plot_slice(
    cv_func: Callable,
    domain: tuple[np.complex128] = (-1 - 1j, 1 + 1j),
    range: tuple[np.complex128] = (-1 - 1j, 1 + 1j),
    num_samples: int = 20,
):
    """Plot the 3D slice of the output"""
    input_grid = get_complex_grid(domain, num_samples)
    output_grid = cv_func(input_grid)

    scale = max(np.max(np.abs(output_grid)), np.max(np.abs(input_grid)))
    output_hsv = get_complex_hsv(output_grid, scale)

    fig = plt.figure()
    ax_dir = fig.add_subplot(1, 2, 1)
    ax_line = fig.add_subplot(1, 2, 2, projection="3d")

    segment = ViewSegment(cross_width=np.abs(np.diff(domain)[0]) / 2)

    ax_dir.imshow(
        clr.hsv_to_rgb(output_hsv),
        origin="lower",
        aspect="equal",
        extent=(*np.real(domain), *np.imag(domain)),
    )
    (seg_plot,) = ax_dir.plot(np.real(segment.end_points), np.imag(segment.end_points))
    arrow = pch.FancyArrowPatch(
        (
            np.real(segment.center - segment.norm_view_dir * 0.5 * segment.cross_width),
            np.imag(segment.center - segment.norm_view_dir * 0.5 * segment.cross_width),
        ),
        (np.real(segment.center), np.imag(segment.center)),
        mutation_scale=10,
    )
    (indicator,) = ax_dir.plot(
        np.real(segment.end_points[0]), np.imag(segment.end_points[0]), "ok"
    )
    ax_dir.add_patch(arrow)
    ax_dir.grid(True)
    ax_dir.set_xlabel("Real Input")
    ax_dir.set_ylabel("Imag Input")
    ax_dir.set_xlim(np.real(domain))
    ax_dir.set_ylim(np.imag(domain))
    ax_dir.set_aspect("equal")

    outputs = cv_func(segment.samples)
    (line_plot,) = ax_line.plot(
        np.real(outputs),
        np.imag(outputs),
        complex_dot(segment.samples, segment.ax_direction),
    )
    (line_indicator,) = ax_line.plot(
        np.real(outputs[0]),
        np.imag(outputs[0]),
        -segment.cross_width / 2,
        marker="o",
        color="k",
    )
    ax_line.set_xlabel("Real Output")
    ax_line.set_ylabel("Imag Output")
    ax_line.set_zlabel("Segment")
    ax_line.set_xlim(np.real(range))
    ax_line.set_ylim(np.imag(range))

    # Interactions
    state = {"dragging": False}

    def update_segment():
        outputs = cv_func(segment.samples)

        seg_plot.set_xdata(np.real(segment.end_points))
        seg_plot.set_ydata(np.imag(segment.end_points))
        arrow.set_positions(
            (
                np.real(
                    segment.center - segment.norm_view_dir * segment.cross_width / 2
                ),
                np.imag(
                    segment.center - segment.norm_view_dir * segment.cross_width / 2
                ),
            ),
            (np.real(segment.center), np.imag(segment.center)),
        )
        indicator.set_xdata([np.real(segment.end_points[0])])
        indicator.set_ydata([np.imag(segment.end_points[0])])

        line_plot.set_data(np.real(outputs), np.imag(outputs))
        line_plot.set_3d_properties(complex_dot(segment.ax_direction, segment.samples))
        line_indicator.set_data([np.real(outputs[0])], [np.imag(outputs[0])])
        line_indicator.set_3d_properties(-segment.cross_width / 2)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes == ax_dir and event.button == 1:
            state["dragging"] = True

    def set_direction(event):
        if event.inaxes == ax_dir and state["dragging"]:
            new_view = event.xdata + 1j * event.ydata
            segment.view_direction = new_view
            update_segment()

    def on_release(event):
        if event.button == 1 and state["dragging"]:
            state["dragging"] = False

    def on_scroll(event):
        if event.inaxes == ax_dir:
            segment.travel += event.step / 10
            update_segment()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", set_direction)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    plt.show()


def plot_real_tube(
    cv_func: Callable,
    domain: tuple[np.complex128],
    range: tuple[np.complex128] = None,
    dense_count: int = 101,
    sparse_count: int = 21,
    log_min: float = -1,
):
    if np.real(np.diff(domain)) < 0 or np.imag(np.diff(domain)) < 0:
        raise ValueError("Domain should be bottom left to top right")
    if range is not None and (
        np.real(np.diff(range)) < 0 or np.imag(np.diff(range)) < 0
    ):
        raise ValueError("Range should be bottom left to top right")

    real_dense_domain = np.linspace(*np.real(domain), dense_count)
    real_sparse_domain = np.linspace(*np.real(domain), sparse_count)
    # imag_dense_domain = np.linspace(*np.imag(domain), 100)
    # imag_sparse_domain = np.linspace(*np.imag(domain), 10)
    if np.prod(np.imag(domain)) == 0:
        imag_dense_domain = np.logspace(
            log_min, np.log10(np.imag(domain[-1])), dense_count
        )
        imag_sparse_domain = np.logspace(
            log_min, np.log10(np.imag(domain[-1])), sparse_count
        )
    elif np.prod(np.imag(domain)) < 0:
        imag_dense_domain_pos = np.logspace(
            log_min, np.log10(max(np.imag(domain))), dense_count // 2
        )
        imag_dense_domain_neg = -np.logspace(
            log_min, np.log10(np.abs(min(np.imag(domain)))), dense_count // 2
        )
        imag_dense_domain = np.hstack(
            (imag_dense_domain_neg[::-1], 0, imag_dense_domain_pos)
        )

        imag_sparse_domain_pos = np.logspace(
            log_min, np.log10(max(np.imag(domain))), sparse_count // 2
        )
        imag_sparse_domain_neg = -np.logspace(
            log_min, np.log10(np.abs(min(np.imag(domain)))), sparse_count // 2
        )
        imag_sparse_domain = np.hstack(
            (imag_sparse_domain_neg[::-1], 0, imag_sparse_domain_pos)
        )
    else:
        imag_dense_domain = np.logspace(*np.log10(np.imag(domain)), dense_count)
        imag_sparse_domain = np.logspace(*np.log10(np.imag(domain)), sparse_count)

    imag_lines = real_sparse_domain[:, None] + 1j * imag_dense_domain[None, :]
    real_lines = real_dense_domain[None, :] + 1j * imag_sparse_domain[:, None]

    output_level = [cv_func(line) for line in imag_lines]
    output_plunge = [cv_func(line) for line in real_lines]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    cmap = plt.get_cmap("coolwarm")

    norm = clr.SymLogNorm(
        linthresh=10**log_min,
        vmin=imag_sparse_domain.min(),
        vmax=imag_sparse_domain.max(),
    )

    for idx, line in enumerate(output_level):
        ax.plot(
            np.real(imag_lines[idx]),
            np.real(line),
            np.imag(line),
            c="k",
            clip_on=True,
        )
    for idx, line in enumerate(output_plunge):
        color = cmap(norm(imag_sparse_domain[idx]))
        ax.plot(
            np.real(real_lines[idx]),
            np.real(line),
            np.imag(line),
            c=color,
            clip_on=True,
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Imag Input")

    if range is not None:
        ax.set_ylim(np.real(range))
        ax.set_zlim(np.imag(range))

    ax.set_xlabel("Real Input")
    ax.set_ylabel("Real Output")
    ax.set_zlabel("Imag Output")

    ax.set_aspect("equal")

    targets = [(90, None), (0, -90), (0, 0)]  # XY  # XZ  # YZ

    def snap_view(event):
        if event.name == "button_release_event":
            elev, azim = ax.elev, ax.azim
            for te, ta in targets:
                if abs(elev - te) < 10 and ta is None:
                    ax.view_init(te, -90)
                    ax.set_proj_type("ortho")
                    fig.canvas.draw_idle()
                    break
                if ta is None:
                    continue
                if abs(elev - te) < 10 and abs(azim - ta) < 10:
                    ax.view_init(te, ta)
                    ax.set_proj_type("ortho")
                    fig.canvas.draw_idle()
                    break
        if (
            event.name == "button_press_event"
            and event.inaxes == ax
            and event.button == 1
        ):
            ax.set_proj_type("persp")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_release_event", snap_view)
    fig.canvas.mpl_connect("button_press_event", snap_view)
    plt.show()


if __name__ == "__main__":

    def example_func(z: np.complex128) -> np.complex128:
        # return (z + 1) / (z - 2)
        return np.sin(z)
        # return np.exp(z)
        # return z**2

    domain = (-5 - 2j, 5 + 2j)
    range = (-5 - 5j, 5 + 5j)
    # plot_colormap(example_func, domain)
    # plot_3d_colormap(example_func, domain)
    # plot_real_imag(example_func, domain)
    # plot_slice(example_func)
    plot_real_tube(example_func, domain, range)
