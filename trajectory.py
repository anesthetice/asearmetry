from os import PathLike
from prelude import pf, np, jax, jnp, jsp, PI
import dataclasses
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib as mpl
from utils import enforce_filepath
from mpl_toolkits.mplot3d import Axes3D


@dataclasses.dataclass
class Trajectory:
    points: jax.Array  # Mx3 matrix
    frequency: int  # in Hz

    def __len__(self) -> int:
        return self.len()

    def len(self) -> int:
        return self.points.shape[0]

    def as_coordinates(self) -> pf.Coordinates:
        return pf.Coordinates.from_cartesian(
            self.points[:, 0], self.points[:, 1], self.points[:, 2]
        )

    def graph(self, filepath: str | PathLike):
        filepath = enforce_filepath(filepath, [".png", ".svg", ".pdf", ".jpg"])

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        if "Axes3D" not in ax.__str__():
            raise ValueError("Only three-dimensional axes supported.")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")  # type: ignore

        bounds = (jnp.min(self.points).item(), jnp.max(self.points).item())

        # unfortunately ax.set_aspect('equal') does not work on Axes3D
        ax_lims = [
            bounds[0] - 0.15 * jnp.abs(bounds[0]),
            bounds[1] + 0.15 * jnp.abs(bounds[1]),
        ]
        if not ax.get_autoscale_on():
            if ax_lims[0] > ax.get_xlim()[0]:
                ax_lims[0] = ax.get_xlim()[0]
            if ax_lims[1] < ax.get_xlim()[1]:
                ax_lims[1] = ax.get_xlim()[1]
        ax_lims = tuple(ax_lims)
        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_zlim(ax_lims)  # type: ignore

        ax.plot([bounds[0], bounds[1]], [0, 0], [0, 0], linestyle="--", color="darkred")
        ax.plot(
            [0, 0], [bounds[0], bounds[1]], [0, 0], linestyle="--", color="darkgreen"
        )
        ax.plot(
            [0, 0], [0, 0], [bounds[0], bounds[1]], linestyle="--", color="darkblue"
        )

        ax.scatter(
            self.points[:, 0],
            self.points[:, 1],
            self.points[:, 2],
            c=jnp.linspace(0, 1, self.len()),
            cmap="plasma",
        )

        fig.savefig(filepath)

    def graph_animated(self, filepath: str | PathLike):
        filepath = enforce_filepath(filepath, [".gif", ".apng"])

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        if "Axes3D" not in ax.__str__():
            raise ValueError("Only three-dimensional axes supported.")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")  # type: ignore

        bounds = (jnp.min(self.points).item(), jnp.max(self.points).item())

        ax_lims = [
            bounds[0] - 0.15 * jnp.abs(bounds[0]),
            bounds[1] + 0.15 * jnp.abs(bounds[1]),
        ]
        if not ax.get_autoscale_on():
            if ax_lims[0] > ax.get_xlim()[0]:
                ax_lims[0] = ax.get_xlim()[0]
            if ax_lims[1] < ax.get_xlim()[1]:
                ax_lims[1] = ax.get_xlim()[1]
        ax_lims = tuple(ax_lims)
        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_zlim(ax_lims)  # type: ignore

        ax.plot([bounds[0], bounds[1]], [0, 0], [0, 0], linestyle="--", color="darkred")
        ax.plot(
            [0, 0], [bounds[0], bounds[1]], [0, 0], linestyle="--", color="darkgreen"
        )
        ax.plot(
            [0, 0], [0, 0], [bounds[0], bounds[1]], linestyle="--", color="darkblue"
        )

        colors = mpl.colormaps.get_cmap("plasma")(jnp.linspace(0, 1, self.len()))

        animated = ax.scatter(self.points[0, 0], self.points[0, 1], self.points[0, 2])

        def update(frame, *fargs):
            animated._offsets3d = ( # type: ignore
                self.points[0:frame, 0],
                self.points[0:frame, 1],
                self.points[0:frame, 2],
            )  # type: ignore
            animated.set(color=colors[0:frame])

        anim = ani.FuncAnimation(
            fig=fig,
            func=update,  # type: ignore
            frames=self.len(),
        )
        writer = ani.PillowWriter(fps=self.frequency)
        anim.save(filename=filepath, writer=writer)


def example(frequency: int, duration: float):
    T_polar = 5  # seconds
    R_polar = 2  # meters
    T_z = 40

    @jax.jit
    def pos(t):
        return jnp.column_stack(
            [
                R_polar * jnp.cos(2 * PI * t / T_polar),
                R_polar * jnp.sin(2 * PI * t / T_polar),
                0.2 + 0.5 * jnp.sin(2 * PI * t / T_z),
            ]
        )

    t_values = jnp.linspace(0, duration, int(jnp.ceil(frequency * duration)))

    trajectory = Trajectory(pos(t_values), frequency)
    return trajectory
