"""Matplotlib based solara components for visualization MESA spaces and plots."""

import warnings
from collections import defaultdict
from collections.abc import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import solara
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.figure import Figure

import mesa
from mesa.experimental.cell_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
    VoronoiGrid,
)
from mesa.space import (
    ContinuousSpace,
    HexMultiGrid,
    HexSingleGrid,
    MultiGrid,
    NetworkGrid,
    PropertyLayer,
    SingleGrid,
)
from mesa.visualization.utils import update_counter

OrthogonalGrid = SingleGrid | MultiGrid | OrthogonalMooreGrid | OrthogonalVonNeumannGrid
HexGrid = HexSingleGrid | HexMultiGrid | mesa.experimental.cell_space.HexGrid
Network = NetworkGrid | mesa.experimental.cell_space.Network


def make_space_matplotlib(agent_portrayal=None, propertylayer_portrayal=None):
    """Create a Matplotlib-based space visualization component.

    Args:
        agent_portrayal (function): Function to portray agents
        propertylayer_portrayal (dict): Dictionary of PropertyLayer portrayal specifications

    Returns:
        function: A function that creates a SpaceMatplotlib component
    """
    if agent_portrayal is None:
        def agent_portrayal(a):
            return {"id": a.unique_id}

    def MakeSpaceMatplotlib(model):
        return SpaceMatplotlib(model, agent_portrayal, propertylayer_portrayal)

    return MakeSpaceMatplotlib


@solara.component
def SpaceMatplotlib(
        model,
        agent_portrayal,
        propertylayer_portrayal,
        dependencies: list[any] | None = None,
):
    """Create a Matplotlib-based space visualization component."""
    update_counter.get()

    space = getattr(model, "grid", None)
    if space is None:
        space = getattr(model, "space", None)

    # https://stackoverflow.com/questions/67524641/convert-multiple-isinstance-checks-to-structural-pattern-matching
    match space:
        case mesa.space._Grid():
            fig, ax = draw_orthogonal_grid(space, agent_portrayal, propertylayer_portrayal, model)
        case OrthogonalMooreGrid():
            fig, ax = draw_orthogonal_grid(space, agent_portrayal, propertylayer_portrayal, model)
        case OrthogonalVonNeumannGrid():
            fig, ax = draw_orthogonal_grid(space, agent_portrayal, propertylayer_portrayal, model)
        case HexSingleGrid():
            fig, ax = draw_hex_grid(space, agent_portrayal, propertylayer_portrayal, model)
        case HexSingleGrid():
            fig, ax = draw_hex_grid(space, agent_portrayal, propertylayer_portrayal, model)
        case mesa.experimental.cell_space.HexGrid():
            fig, ax = draw_hex_grid(space, agent_portrayal, propertylayer_portrayal, model)
        case mesa.space.ContinuousSpace():
            fig, ax = draw_continuous_space(space, agent_portrayal)
        case mesa.space.NetworkGrid():
            fig, ax = draw_network(space, agent_portrayal)
        case VoronoiGrid():
            fig, ax = draw_voroinoi_grid(space, agent_portrayal)
        case mesa.experimental.cell_space.Network():
            fig, ax = draw_network(space, agent_portrayal)
        case None:
            fig = Figure()
            ax = fig.subplots(111)
            if propertylayer_portrayal:
                draw_property_layers(ax, space, propertylayer_portrayal, model)

    solara.FigureMatplotlib(
        fig, format="png", bbox_inches="tight", dependencies=dependencies
    )


def draw_property_layers(ax, space, propertylayer_portrayal, model):
    """Draw PropertyLayers on the given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes to draw on.
        space (mesa.space._Grid): The space containing the PropertyLayers.
        propertylayer_portrayal (dict): Dictionary of PropertyLayer portrayal specifications.
        model (mesa.Model): The model instance.
    """
    for layer_name, portrayal in propertylayer_portrayal.items():
        layer = getattr(model, layer_name, None)
        if not isinstance(layer, PropertyLayer):
            continue

        data = layer.data.astype(float) if layer.data.dtype == bool else layer.data
        width, height = data.shape if space is None else (space.width, space.height)

        if space and data.shape != (width, height):
            warnings.warn(
                f"Layer {layer_name} dimensions ({data.shape}) do not match space dimensions ({width}, {height}).",
                UserWarning,
                stacklevel=2,
            )

        # Get portrayal properties, or use defaults
        alpha = portrayal.get("alpha", 1)
        vmin = portrayal.get("vmin", np.min(data))
        vmax = portrayal.get("vmax", np.max(data))
        colorbar = portrayal.get("colorbar", True)

        # Draw the layer
        if "color" in portrayal:
            rgba_color = to_rgba(portrayal["color"])
            normalized_data = (data - vmin) / (vmax - vmin)
            rgba_data = np.full((*data.shape, 4), rgba_color)
            rgba_data[..., 3] *= normalized_data * alpha
            rgba_data = np.clip(rgba_data, 0, 1)
            cmap = LinearSegmentedColormap.from_list(
                layer_name, [(0, 0, 0, 0), (*rgba_color[:3], alpha)]
            )
            im = ax.imshow(
                rgba_data.transpose(1, 0, 2),
                extent=(0, width, 0, height),
                origin="lower",
            )
            if colorbar:
                norm = Normalize(vmin=vmin, vmax=vmax)
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                ax.figure.colorbar(sm, ax=ax, orientation="vertical")

        elif "colormap" in portrayal:
            cmap = portrayal.get("colormap", "viridis")
            if isinstance(cmap, list):
                cmap = LinearSegmentedColormap.from_list(layer_name, cmap)
            im = ax.imshow(
                data.T,
                cmap=cmap,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
                extent=(0, width, 0, height),
                origin="lower",
            )
            if colorbar:
                plt.colorbar(im, ax=ax, label=layer_name)
        else:
            raise ValueError(
                f"PropertyLayer {layer_name} portrayal must include 'color' or 'colormap'."
            )


def collect_agent_data(space: OrthogonalGrid | HexGrid | Network | ContinuousSpace, agent_portrayal: Callable):
    """Collect the plotting data for all agents in the space.

    Args:
        space: The space containing the Agents.
        agent_portrayal: A callable that is called with the agent and returns a dict
        loc: a boolean indicating whether to gather agent x, y data or not

    Notes:
        agent portray dict is limited to s (size of marker), c (color of marker, and marker (marker style)
        see `Matplotlib`_.


    .. _Matplotlib: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

    """
    default_size = (180 / max(space.width, space.height)) ** 2

    arguments = {"x":[], "y":[], "s":[], "c":[], "marker":[]}
    for agent in space.agents:
        portray = agent_portrayal(agent)
        loc = agent.pos
        if loc is None:
            loc = agent.cell.coordinate
        x, y = loc
        arguments["x"].append(x)
        arguments["y"].append(y)

        arguments["s"].append(portray.get("s", default_size))
        arguments["c"].append(portray.get("c", "tab:blue"))
        arguments["marker"].append(portray.get("marker", "o"))

    return {k: np.asarray(v) for k, v in arguments.items()}


def draw_orthogonal_grid(space: OrthogonalGrid, agent_portrayal: Callable, propertylayer_portrayal: Callable, model):
    """Visualize a orthogonal grid..

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a dict
        propertylayer_portrayal: a callable that is called with the agent and returns a dict
        model: a model instance

    Returns:
        A Figure and Axes instance

    """
    arguments = collect_agent_data(space, agent_portrayal)

    fig = Figure()
    ax = fig.add_subplot(111)

    ax.set_xlim(-0.5, space.width-0.5)
    ax.set_ylim(-0.5, space.height-0.5)

    # Draw grid lines
    for x in np.arange(-0.5, space.width-0.5, 1):
        ax.axvline(x, color="gray", linestyle=":")
    for y in np.arange(-0.5, space.height-0.5, 1):
        ax.axhline(y, color="gray", linestyle=":")

    _scatter(ax, arguments)

    if propertylayer_portrayal:
        draw_property_layers(ax, space, propertylayer_portrayal, model)

    return fig, ax


def draw_hex_grid(space: HexGrid, agent_portrayal: Callable, propertylayer_portrayal: Callable, model):
    """Visualize a hex grid.

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a dict
        propertylayer_portrayal: a callable that is called with the agent and returns a dict
        model: a model instance

    Returns:
        A Figure and Axes instance

    """
    arguments = collect_agent_data(space, agent_portrayal)

    # give all odd rows an offset in the x direction
    logical = np.mod(arguments["y"], 2) == 1
    arguments["x"] = arguments["x"][logical] + 1

    fig = Figure()
    ax = fig.add_subplot(111)

    _scatter(ax, arguments)

    if propertylayer_portrayal:
        draw_property_layers(ax, space, propertylayer_portrayal, model)
    return fig, ax

def collect_agent_data_for_networks(space, agent_portrayal):
    """Collect the plotting data for all agents in the network.

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a dict

    """
    arguments = defaultdict(list)
    for agent in space.agents:
        portray = agent_portrayal(agent)
        for k, v in portray.items():
            arguments[k].append(v)

    return arguments

def draw_network(space: Network, agent_portrayal: Callable):
    """Visualize a network space.

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a dict

    Notes:
        this uses networkx.draw under the hood so agent portrayal fields should match those used there
        i.e., node_side, node_color, node_shape, edge_color


    """
    arguments = collect_agent_data_for_networks(agent_portrayal)

    fig, ax = plt.subplots()
    graph = space.G
    pos = nx.spring_layout(graph, seed=0)
    nx.draw(
        graph,
        ax=ax,
        pos=pos,
        **arguments,
    )

    return fig, ax


def draw_continuous_space(space: ContinuousSpace, agent_portrayal: Callable):
    """Visualize a continuous space.

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a dict

    Returns:
        A Figure and Axes instance

    """
    arguments = collect_agent_data(space, agent_portrayal)

    fig = Figure()
    ax = fig.add_subplot(111)

    border_style = "solid" if not space.torus else (0, (5, 10))

    # Set the border of the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")
        spine.set_linestyle(border_style)

    width = space.x_max - space.x_min
    x_padding = width / 20
    height = space.y_max - space.y_min
    y_padding = height / 20

    ax.set_xlim(space.x_min - x_padding, space.x_max + x_padding)
    ax.set_ylim(space.y_min - y_padding, space.y_max + y_padding)

    _scatter(ax, arguments)
    return fig, ax


def draw_voroinoi_grid(space: VoronoiGrid, agent_portrayal: Callable):
    """Visualize a voronoi grid.

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a dict

    Returns:
        A Figure and Axes instance

    """
    arguments = collect_agent_data(space, agent_portrayal)

    fig = Figure()
    ax = fig.add_subplot(111)

    x_list = [i[0] for i in space.centroids_coordinates]
    y_list = [i[1] for i in space.centroids_coordinates]
    x_max = max(x_list)
    x_min = min(x_list)
    y_max = max(y_list)
    y_min = min(y_list)

    width = x_max - x_min
    x_padding = width / 20
    height = y_max - y_min
    y_padding = height / 20
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    _scatter(ax, arguments)


    for cell in space.all_cells:
        polygon = cell.properties["polygon"]
        ax.fill(
            *zip(*polygon),
            alpha=min(1, cell.properties[space.cell_coloring_property]),
            c="red", zorder=0
        )  # Plot filled polygon
        ax.plot(*zip(*polygon), color="black")  # Plot polygon edges in black
    return fig, ax



def _scatter(ax, arguments):
    x = arguments.pop("x")
    y = arguments.pop("y")
    marker = arguments.pop("marker")

    for mark in np.unique(marker):
        mask = marker == mark
        ax.scatter(x[mask], y[mask], marker=mark, **{k: v[mask] for k, v in arguments.items()})


def make_plot_measure(measure: str | dict[str, str] | list[str] | tuple[str]):
    """Create a plotting function for a specified measure.

    Args:
        measure (str | dict[str, str] | list[str] | tuple[str]): Measure(s) to plot.

    Returns:
        function: A function that creates a PlotMatplotlib component.
    """

    def MakePlotMeasure(model):
        return PlotMatplotlib(model, measure)

    return MakePlotMeasure


@solara.component
def PlotMatplotlib(model, measure, dependencies: list[any] | None = None):
    """Create a Matplotlib-based plot for a measure or measures.

    Args:
        model (mesa.Model): The model instance.
        measure (str | dict[str, str] | list[str] | tuple[str]): Measure(s) to plot.
        dependencies (list[any] | None): Optional dependencies for the plot.

    Returns:
        solara.FigureMatplotlib: A component for rendering the plot.
    """
    update_counter.get()
    fig = Figure()
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if isinstance(measure, str):
        ax.plot(df.loc[:, measure])
        ax.set_ylabel(measure)
    elif isinstance(measure, dict):
        for m, color in measure.items():
            ax.plot(df.loc[:, m], label=m, color=color)
        ax.legend(loc="best")
    elif isinstance(measure, list | tuple):
        for m in measure:
            ax.plot(df.loc[:, m], label=m)
        ax.legend(loc="best")
    ax.set_xlabel("Step")
    # Set integer x axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    solara.FigureMatplotlib(
        fig, format="png", bbox_inches="tight", dependencies=dependencies
    )
