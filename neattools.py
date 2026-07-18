"""Visualization and analysis helpers for MultiNEAT2.

The historical public helpers (``Genome2NX``, ``DrawGenome``,
``DrawGenomes``, ``compute_node_positions``, and the text/export utilities)
remain available.  The module now also provides topology-aware layouts,
genome comparison, population dashboards, evolution tracking, and optional
Plotly HTML output.

NetworkX, matplotlib, and NumPy are the only dependencies required for the
static tools. Plotly and pydot are imported only by the exporters that need
them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
import warnings

import matplotlib.pyplot as plt
from matplotlib import lines as mpl_lines
import networkx as nx
import numpy as np
import pymultineat as pnt


INPUT = pnt.INPUT
BIAS = pnt.BIAS
OUTPUT = pnt.OUTPUT
HIDDEN = pnt.HIDDEN

_TYPE_NAMES = {
    INPUT: "Input",
    BIAS: "Bias",
    HIDDEN: "Hidden",
    OUTPUT: "Output",
}

_ACTIVATION_NAMES = {
    getattr(pnt, name): name.replace("_", " ").title()
    for name in (
        "SIGNED_SIGMOID",
        "UNSIGNED_SIGMOID",
        "TANH",
        "TANH_CUBIC",
        "SIGNED_STEP",
        "UNSIGNED_STEP",
        "SIGNED_GAUSS",
        "UNSIGNED_GAUSS",
        "ABS",
        "SIGNED_SINE",
        "UNSIGNED_SINE",
        "LINEAR",
        "RELU",
        "SOFTPLUS",
    )
    if hasattr(pnt, name)
}


@dataclass(frozen=True)
class VisualTheme:
    """Colors and scaling shared by static genome visualizations."""

    background: str = "#0f172a"
    foreground: str = "#e2e8f0"
    muted: str = "#94a3b8"
    grid: str = "#334155"
    input_color: str = "#22c55e"
    bias_color: str = "#facc15"
    hidden_color: str = "#38bdf8"
    output_color: str = "#fb7185"
    positive_color: str = "#2563eb"
    negative_color: str = "#ef4444"
    zero_color: str = "#64748b"


DEFAULT_THEME = VisualTheme()


def _trait_value(value: Any) -> Any:
    """Return a JSON-friendly trait representation."""

    raw = getattr(value, "value", value)
    if isinstance(raw, (str, int, float, bool)) or raw is None:
        return raw
    if isinstance(raw, Sequence):
        return [_trait_value(item) for item in raw]
    return repr(raw)


def _traits(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}
    return {str(key): _trait_value(value) for key, value in values.items()}


def Genome2NX(genome: pnt.Genome) -> nx.DiGraph:
    """Convert a genome to an attributed :class:`networkx.DiGraph`.

    Node IDs and innovation IDs are preserved. Link weights, recurrence,
    activation parameters, geometry, and universal traits are included as
    attributes, making the result useful beyond drawing.
    """

    graph = nx.DiGraph(
        genome_id=genome.GetID(),
        fitness=genome.GetFitness(),
    )
    for neuron in genome.m_NeuronGenes:
        node_id = neuron.m_ID
        neuron_type = neuron.m_Type
        activation = neuron.m_ActFunction
        graph.add_node(
            node_id,
            type=neuron_type,
            type_name=_TYPE_NAMES.get(neuron_type, str(neuron_type)),
            x=neuron.x,
            y=neuron.y,
            split_y=neuron.m_SplitY,
            a=neuron.m_A,
            b=neuron.m_B,
            time_constant=neuron.m_TimeConstant,
            bias=neuron.m_Bias,
            act_function=activation,
            activation_name=_ACTIVATION_NAMES.get(activation, str(activation)),
            traits=_traits(neuron.m_Traits),
        )
    for link in genome.m_LinkGenes:
        graph.add_edge(
            link.m_FromNeuronID,
            link.m_ToNeuronID,
            innovation_id=link.m_InnovationID,
            weight=link.m_Weight,
            is_recurrent=bool(link.m_IsRecurrent),
            traits=_traits(link.m_Traits),
        )
    return graph


def genome_summary(genome: pnt.Genome) -> dict[str, Any]:
    """Return machine-readable topology and weight statistics."""

    graph = Genome2NX(genome)
    weights = np.asarray(
        [data["weight"] for _, _, data in graph.edges(data=True)],
        dtype=float,
    )
    recurrent = sum(bool(data["is_recurrent"]) for _, _, data in graph.edges(data=True))
    type_counts = defaultdict(int)
    for _, data in graph.nodes(data=True):
        type_counts[data["type_name"]] += 1
    feed_forward = nx.DiGraph(
        (source, target)
        for source, target, data in graph.edges(data=True)
        if not data["is_recurrent"] and source != target
    )
    feed_forward.add_nodes_from(graph.nodes)
    strongly_connected = list(nx.strongly_connected_components(graph))
    cyclic_components = sum(
        len(component) > 1
        or any(graph.has_edge(node, node) for node in component)
        for component in strongly_connected
    )
    condensation = nx.condensation(feed_forward)
    feed_forward_depth = (
        nx.dag_longest_path_length(condensation)
        if condensation.number_of_nodes()
        else 0
    )
    return {
        "id": genome.GetID(),
        "fitness": genome.GetFitness(),
        "neurons": graph.number_of_nodes(),
        "links": graph.number_of_edges(),
        "recurrent_links": recurrent,
        "type_counts": dict(type_counts),
        "weight_min": float(weights.min()) if weights.size else 0.0,
        "weight_max": float(weights.max()) if weights.size else 0.0,
        "weight_mean": float(weights.mean()) if weights.size else 0.0,
        "weight_std": float(weights.std()) if weights.size else 0.0,
        "is_feed_forward": nx.is_directed_acyclic_graph(feed_forward),
        "feed_forward_depth": int(feed_forward_depth),
        "self_loops": nx.number_of_selfloops(graph),
        "strongly_connected_components": len(strongly_connected),
        "cyclic_components": cyclic_components,
        "density": float(nx.density(graph)),
    }


def compare_genomes(
    left: pnt.Genome,
    right: pnt.Genome,
) -> dict[str, Any]:
    """Compare two genomes by neuron ID and link innovation ID."""

    left_neurons = {node.m_ID: node for node in left.m_NeuronGenes}
    right_neurons = {node.m_ID: node for node in right.m_NeuronGenes}
    left_links = {link.m_InnovationID: link for link in left.m_LinkGenes}
    right_links = {link.m_InnovationID: link for link in right.m_LinkGenes}
    matching = sorted(left_links.keys() & right_links.keys())
    changed_weights = {
        innovation: (
            left_links[innovation].m_Weight,
            right_links[innovation].m_Weight,
        )
        for innovation in matching
        if not math.isclose(
            left_links[innovation].m_Weight,
            right_links[innovation].m_Weight,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    }
    return {
        "left_id": left.GetID(),
        "right_id": right.GetID(),
        "shared_neurons": sorted(left_neurons.keys() & right_neurons.keys()),
        "left_only_neurons": sorted(left_neurons.keys() - right_neurons.keys()),
        "right_only_neurons": sorted(right_neurons.keys() - left_neurons.keys()),
        "matching_innovations": matching,
        "left_only_innovations": sorted(left_links.keys() - right_links.keys()),
        "right_only_innovations": sorted(right_links.keys() - left_links.keys()),
        "changed_weights": changed_weights,
    }


def _feed_forward_graph(graph: nx.DiGraph) -> nx.DiGraph:
    result = nx.DiGraph()
    result.add_nodes_from(graph.nodes)
    result.add_edges_from(
        (source, target)
        for source, target, data in graph.edges(data=True)
        if not data.get("is_recurrent", False) and source != target
    )
    return result


def _topology_layers(graph: nx.DiGraph) -> list[list[int]]:
    """Build stable layers, condensing cycles when recurrence is unmarked."""

    feed_forward = _feed_forward_graph(graph)
    components = list(nx.strongly_connected_components(feed_forward))
    component_index = {
        node: index for index, component in enumerate(components) for node in component
    }
    condensed = nx.DiGraph()
    condensed.add_nodes_from(range(len(components)))
    for source, target in feed_forward.edges:
        source_component = component_index[source]
        target_component = component_index[target]
        if source_component != target_component:
            condensed.add_edge(source_component, target_component)

    component_layer: dict[int, int] = {}
    for component in nx.topological_sort(condensed):
        predecessors = list(condensed.predecessors(component))
        component_layer[component] = (
            max(component_layer[item] for item in predecessors) + 1
            if predecessors
            else 0
        )

    node_layer = {node: component_layer[component_index[node]] for node in graph.nodes}
    inputs = [
        node for node, data in graph.nodes(data=True) if data["type"] in (INPUT, BIAS)
    ]
    for node in inputs:
        node_layer[node] = 0

    # Relax longest paths after pinning inputs. This produces intuitive layers
    # even when node IDs or split coordinates carry no geometric information.
    for _ in range(max(1, graph.number_of_nodes())):
        changed = False
        for source, target in condensed_edges_from_nodes(feed_forward, node_layer):
            if graph.nodes[target]["type"] in (INPUT, BIAS):
                continue
            proposed = node_layer[source] + 1
            if proposed > node_layer[target]:
                node_layer[target] = proposed
                changed = True
        if not changed:
            break

    output_nodes = [
        node for node, data in graph.nodes(data=True) if data["type"] == OUTPUT
    ]
    maximum_hidden = max(
        (layer for node, layer in node_layer.items() if node not in output_nodes),
        default=0,
    )
    for node in output_nodes:
        node_layer[node] = max(node_layer[node], maximum_hidden + 1)

    grouped: MutableMapping[int, list[int]] = defaultdict(list)
    for node, layer in node_layer.items():
        grouped[layer].append(node)

    # A few barycentric sweeps reduce edge crossings without external
    # Graphviz dependencies.
    layers = [sorted(grouped[layer]) for layer in sorted(grouped)]
    for _ in range(4):
        positions = {
            node: index for layer in layers for index, node in enumerate(layer)
        }
        for layer_index in range(1, len(layers)):
            layers[layer_index].sort(
                key=lambda node: _barycenter(
                    feed_forward.predecessors(node), positions, node
                )
            )
        positions = {
            node: index for layer in layers for index, node in enumerate(layer)
        }
        for layer_index in range(len(layers) - 2, -1, -1):
            layers[layer_index].sort(
                key=lambda node: _barycenter(
                    feed_forward.successors(node), positions, node
                )
            )
    return layers


def condensed_edges_from_nodes(
    graph: nx.DiGraph,
    node_layers: Mapping[int, int],
) -> Iterable[tuple[int, int]]:
    """Yield forward edges in a stable order, ignoring cycle back-edges."""

    for source, target in sorted(graph.edges):
        if source == target:
            continue
        if node_layers[source] < node_layers[target]:
            yield source, target


def _barycenter(
    neighbors: Iterable[int],
    positions: Mapping[int, int],
    fallback: int,
) -> float:
    values = [positions[node] for node in neighbors]
    return float(np.mean(values)) if values else float(fallback)


def compute_node_positions(
    genome: pnt.Genome,
    layout: str = "auto",
) -> dict[int, tuple[float, float]]:
    """Compute stable node positions.

    ``layout`` can be ``"auto"`` (explicit coordinates when informative,
    otherwise topology), ``"topology"``, ``"split"``, ``"coordinates"``,
    ``"spring"``, or ``"kamada_kawai"``. Inputs are shown at the top and
    outputs at the bottom for layered layouts.
    """

    graph = Genome2NX(genome)
    if not graph:
        return {}
    supported = {
        "auto",
        "topology",
        "split",
        "coordinates",
        "spring",
        "kamada_kawai",
    }
    if layout not in supported:
        raise ValueError(
            f"Unknown layout {layout!r}; expected one of {sorted(supported)}"
        )

    coordinates = {
        node: (float(data["x"]), float(data["y"]))
        for node, data in graph.nodes(data=True)
    }
    coordinate_spread = np.ptp([value[0] for value in coordinates.values()]) + np.ptp(
        [value[1] for value in coordinates.values()]
    )
    if layout == "auto":
        layout = "coordinates" if coordinate_spread > 0.0 else "topology"

    if layout == "coordinates":
        if coordinate_spread == 0.0:
            warnings.warn(
                "All explicit coordinates coincide; using topology layout.",
                RuntimeWarning,
                stacklevel=2,
            )
            layout = "topology"
        else:
            return _normalize_positions(coordinates, invert_y=False)

    if layout == "spring":
        return {
            node: tuple(map(float, value))
            for node, value in nx.spring_layout(
                graph, seed=genome.GetID() & 0xFFFFFFFF
            ).items()
        }
    if layout == "kamada_kawai":
        return {
            node: tuple(map(float, value))
            for node, value in nx.kamada_kawai_layout(graph).items()
        }
    if layout == "split":
        grouped: MutableMapping[float, list[int]] = defaultdict(list)
        for node, data in graph.nodes(data=True):
            if data["type"] in (INPUT, BIAS):
                layer = 0.0
            elif data["type"] == OUTPUT:
                layer = 1.0
            else:
                layer = min(1.0, max(0.0, float(data["split_y"])))
            grouped[layer].append(node)
        layers = [sorted(grouped[layer]) for layer in sorted(grouped)]
    else:
        layers = _topology_layers(graph)

    positions: dict[int, tuple[float, float]] = {}
    denominator = max(1, len(layers) - 1)
    for layer_index, nodes in enumerate(layers):
        count = len(nodes)
        for order, node in enumerate(nodes):
            positions[node] = (
                (order + 1) / (count + 1),
                1.0 - layer_index / denominator,
            )
    return positions


def _normalize_positions(
    positions: Mapping[int, tuple[float, float]],
    *,
    invert_y: bool,
) -> dict[int, tuple[float, float]]:
    xs = np.asarray([value[0] for value in positions.values()], dtype=float)
    ys = np.asarray([value[1] for value in positions.values()], dtype=float)
    x_span = float(np.ptp(xs))
    y_span = float(np.ptp(ys))
    normalized = {}
    for node, (x, y) in positions.items():
        normalized_x = (x - float(xs.min())) / x_span if x_span else 0.5
        normalized_y = (y - float(ys.min())) / y_span if y_span else 0.5
        normalized[node] = (
            normalized_x,
            1.0 - normalized_y if invert_y else normalized_y,
        )
    return normalized


def get_layered_nodes(
    genome: pnt.Genome,
    layout: str = "auto",
) -> dict[float, list[int]]:
    """Group node IDs by displayed y-level, top to bottom."""

    positions = compute_node_positions(genome, layout=layout)
    grouped: MutableMapping[float, list[tuple[int, float]]] = defaultdict(list)
    for node, (x, y) in positions.items():
        grouped[round(y, 6)].append((node, x))
    return {
        layer: [node for node, _ in sorted(grouped[layer], key=lambda item: item[1])]
        for layer in sorted(grouped, reverse=True)
    }


def get_topologically_sorted_nodes(genome: pnt.Genome) -> list[int]:
    """Return a topological order, condensing recurrent cycles if needed."""

    graph = _feed_forward_graph(Genome2NX(genome))
    if nx.is_directed_acyclic_graph(graph):
        return list(nx.lexicographical_topological_sort(graph))
    result = []
    condensation = nx.condensation(graph)
    for component in nx.topological_sort(condensation):
        result.extend(sorted(condensation.nodes[component]["members"]))
    return result


def _edge_style(
    weight: float,
    maximum_weight: float,
    theme: VisualTheme,
) -> tuple[str, float, float]:
    magnitude = abs(weight) / maximum_weight if maximum_weight else 0.0
    width = 0.6 + 4.4 * math.sqrt(magnitude)
    alpha = 0.25 + 0.7 * math.sqrt(magnitude)
    color = (
        theme.positive_color
        if weight > 0.0
        else theme.negative_color
        if weight < 0.0
        else theme.zero_color
    )
    return color, width, alpha


def DrawGenome(
    genome: pnt.Genome,
    ax: Any = None,
    node_size: float = 650,
    with_edge_labels: bool = False,
    *,
    layout: str = "auto",
    show_activation: bool = False,
    show_traits: bool = False,
    show_legend: bool = True,
    title: str | None = None,
    theme: VisualTheme = DEFAULT_THEME,
    show: bool = True,
) -> Any:
    """Draw a genome with weight magnitude, sign, and recurrence encoding.

    Positive links are blue, negative links red, line width represents
    magnitude, and recurrent links are dashed and curved. The returned axis
    allows callers to continue customizing or save the figure.
    """

    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    figure = ax.figure
    figure.patch.set_facecolor(theme.background)
    ax.set_facecolor(theme.background)

    graph = Genome2NX(genome)
    positions = compute_node_positions(genome, layout=layout)
    maximum_weight = max(
        (abs(data["weight"]) for _, _, data in graph.edges(data=True)),
        default=1.0,
    )

    node_groups = [
        (INPUT, theme.input_color, "s", "Input"),
        (BIAS, theme.bias_color, "h", "Bias"),
        (HIDDEN, theme.hidden_color, "o", "Hidden"),
        (OUTPUT, theme.output_color, "D", "Output"),
    ]
    legend_handles: list[Any] = []
    for neuron_type, color, shape, label in node_groups:
        nodes = [
            node for node, data in graph.nodes(data=True) if data["type"] == neuron_type
        ]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=nodes,
            node_color=color,
            node_shape=shape,
            node_size=node_size,
            linewidths=1.2,
            edgecolors=theme.foreground,
            ax=ax,
        )
        legend_handles.append(
            mpl_lines.Line2D(
                [],
                [],
                marker=shape,
                linestyle="",
                markerfacecolor=color,
                markeredgecolor=theme.foreground,
                markersize=9,
                label=label,
            )
        )

    edge_groups: MutableMapping[tuple[bool, str], list[tuple[int, int]]] = defaultdict(
        list
    )
    edge_widths: MutableMapping[tuple[bool, str], list[float]] = defaultdict(list)
    edge_alphas: MutableMapping[tuple[bool, str], list[float]] = defaultdict(list)
    for source, target, data in graph.edges(data=True):
        color, width, alpha = _edge_style(data["weight"], maximum_weight, theme)
        recurrent = bool(data["is_recurrent"]) or source == target
        key = recurrent, color
        edge_groups[key].append((source, target))
        edge_widths[key].append(width)
        edge_alphas[key].append(alpha)

    for (recurrent, color), edges in edge_groups.items():
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edges,
            width=edge_widths[(recurrent, color)],
            edge_color=color,
            alpha=float(np.mean(edge_alphas[(recurrent, color)])),
            style="dashed" if recurrent else "solid",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            connectionstyle="arc3,rad=0.22" if recurrent else "arc3,rad=0.02",
            min_source_margin=8,
            min_target_margin=10,
            ax=ax,
        )

    labels = {}
    for node, data in graph.nodes(data=True):
        parts = [str(node)]
        if show_activation and data["type"] not in (INPUT, BIAS):
            parts.append(data["activation_name"])
        if show_traits and data["traits"]:
            parts.extend(
                f"{key}={value}" for key, value in sorted(data["traits"].items())
            )
        labels[node] = "\n".join(parts)
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=8,
        font_color=theme.foreground,
        ax=ax,
    )

    if with_edge_labels:
        edge_labels = {
            (source, target): (f"{data['weight']:.3g}\n#{data['innovation_id']}")
            for source, target, data in graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=edge_labels,
            font_size=7,
            font_color=theme.foreground,
            bbox={
                "facecolor": theme.background,
                "edgecolor": "none",
                "alpha": 0.75,
            },
            ax=ax,
        )

    summary = genome_summary(genome)
    ax.set_title(
        title
        or (
            f"Genome {summary['id']}  |  fitness {summary['fitness']:.6g}"
            f"  |  {summary['neurons']} neurons · {summary['links']} links"
            f" · {summary['recurrent_links']} recurrent"
        ),
        color=theme.foreground,
        fontsize=12,
        pad=14,
    )
    ax.margins(0.16)
    ax.axis("off")
    if show_legend and legend_handles:
        legend_handles.extend(
            [
                mpl_lines.Line2D([], [], color=theme.positive_color, label="Positive"),
                mpl_lines.Line2D([], [], color=theme.negative_color, label="Negative"),
                mpl_lines.Line2D(
                    [],
                    [],
                    color=theme.muted,
                    linestyle="dashed",
                    label="Recurrent",
                ),
            ]
        )
        legend = ax.legend(
            handles=legend_handles,
            loc="best",
            frameon=False,
            fontsize=8,
            ncol=min(4, len(legend_handles)),
        )
        for text in legend.get_texts():
            text.set_color(theme.foreground)
    if own_figure and show:
        plt.show()
    return ax


def DrawGenomes(
    genomes: Sequence[pnt.Genome],
    node_size: float = 420,
    with_edge_labels: bool = False,
    *,
    layout: str = "auto",
    max_columns: int = 5,
    theme: VisualTheme = DEFAULT_THEME,
    show: bool = True,
) -> Any:
    """Draw genomes in a compact comparison grid and return the figure."""

    genomes = list(genomes)
    if not genomes:
        raise ValueError("DrawGenomes requires at least one genome")
    columns = min(max_columns, max(1, math.ceil(math.sqrt(len(genomes) * 5 / 3))))
    rows = math.ceil(len(genomes) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 4.5, rows * 3.5),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, genome in zip(axes.flat, genomes):
        DrawGenome(
            genome,
            ax=axis,
            node_size=node_size,
            with_edge_labels=with_edge_labels,
            layout=layout,
            show_legend=False,
            theme=theme,
            show=False,
        )
    for axis in list(axes.flat)[len(genomes) :]:
        axis.set_facecolor(theme.background)
        axis.axis("off")
    figure.patch.set_facecolor(theme.background)
    if show:
        plt.show()
    return figure


def DrawGenomeComparison(
    left: pnt.Genome,
    right: pnt.Genome,
    *,
    layout: str = "topology",
    with_edge_labels: bool = False,
    theme: VisualTheme = DEFAULT_THEME,
    show: bool = True,
) -> Any:
    """Draw two genomes and a structural diff between them.

    The center panel aligns shared neuron IDs and colors genes as unchanged,
    weight/endpoint changed, left-only, or right-only. This makes crossover
    and structural-mutation effects visible without losing the complete
    parent views.
    """

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(20, 7),
        constrained_layout=True,
    )
    DrawGenome(
        left,
        ax=axes[0],
        layout=layout,
        show_legend=False,
        title=f"Left · genome {left.GetID()}",
        theme=theme,
        show=False,
    )
    DrawGenome(
        right,
        ax=axes[2],
        layout=layout,
        show_legend=False,
        title=f"Right · genome {right.GetID()}",
        theme=theme,
        show=False,
    )

    left_graph = Genome2NX(left)
    right_graph = Genome2NX(right)
    union = nx.DiGraph()
    union.add_nodes_from(left_graph.nodes(data=True))
    for node, data in right_graph.nodes(data=True):
        if node not in union:
            union.add_node(node, **data)

    left_positions = compute_node_positions(left, layout=layout)
    right_positions = compute_node_positions(right, layout=layout)
    positions = {}
    for node in union.nodes:
        if node in left_positions and node in right_positions:
            positions[node] = tuple(
                (left_value + right_value) / 2.0
                for left_value, right_value in zip(
                    left_positions[node], right_positions[node]
                )
            )
        elif node in left_positions:
            positions[node] = left_positions[node]
        else:
            positions[node] = right_positions[node]
    if positions:
        positions = _normalize_positions(positions, invert_y=False)

    left_nodes = set(left_graph.nodes)
    right_nodes = set(right_graph.nodes)
    node_groups = {
        "Shared neuron": (left_nodes & right_nodes, theme.muted),
        "Left-only neuron": (left_nodes - right_nodes, "#f97316"),
        "Right-only neuron": (right_nodes - left_nodes, "#a3e635"),
    }
    diff_axis = axes[1]
    diff_axis.set_facecolor(theme.background)
    for _, (nodes, color) in node_groups.items():
        if nodes:
            nx.draw_networkx_nodes(
                union,
                positions,
                nodelist=sorted(nodes),
                node_color=color,
                node_size=620,
                linewidths=1.2,
                edgecolors=theme.foreground,
                ax=diff_axis,
            )
    nx.draw_networkx_labels(
        union,
        positions,
        font_size=8,
        font_color=theme.foreground,
        ax=diff_axis,
    )

    left_links = {
        data["innovation_id"]: (source, target, data)
        for source, target, data in left_graph.edges(data=True)
    }
    right_links = {
        data["innovation_id"]: (source, target, data)
        for source, target, data in right_graph.edges(data=True)
    }
    edge_groups: MutableMapping[str, list[tuple[int, int]]] = defaultdict(list)
    edge_labels = {}
    for innovation in sorted(left_links.keys() | right_links.keys()):
        if innovation not in right_links:
            source, target, _ = left_links[innovation]
            status = "Left only"
        elif innovation not in left_links:
            source, target, _ = right_links[innovation]
            status = "Right only"
        else:
            left_source, left_target, left_data = left_links[innovation]
            right_source, right_target, right_data = right_links[innovation]
            source, target = left_source, left_target
            changed = (
                (left_source, left_target) != (right_source, right_target)
                or not math.isclose(
                    left_data["weight"],
                    right_data["weight"],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or left_data["is_recurrent"] != right_data["is_recurrent"]
            )
            status = "Changed" if changed else "Unchanged"
            if changed:
                edge_labels[(source, target)] = (
                    f"#{innovation}\n"
                    f"{left_data['weight']:.3g} → {right_data['weight']:.3g}"
                )
        union.add_edge(source, target)
        edge_groups[status].append((source, target))

    status_styles = {
        "Unchanged": (theme.grid, "solid", 1.2),
        "Changed": ("#c084fc", "solid", 3.0),
        "Left only": ("#f97316", "dashed", 2.2),
        "Right only": ("#a3e635", "dashed", 2.2),
    }
    legend_handles = []
    for status, edges in edge_groups.items():
        color, style, width = status_styles[status]
        nx.draw_networkx_edges(
            union,
            positions,
            edgelist=edges,
            edge_color=color,
            style=style,
            width=width,
            alpha=0.9,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            connectionstyle="arc3,rad=0.05",
            ax=diff_axis,
        )
        legend_handles.append(
            mpl_lines.Line2D(
                [],
                [],
                color=color,
                linestyle=style,
                linewidth=width,
                label=status,
            )
        )
    if with_edge_labels and edge_labels:
        nx.draw_networkx_edge_labels(
            union,
            positions,
            edge_labels=edge_labels,
            font_size=7,
            font_color=theme.foreground,
            bbox={
                "facecolor": theme.background,
                "edgecolor": "none",
                "alpha": 0.8,
            },
            ax=diff_axis,
        )

    comparison = compare_genomes(left, right)
    diff_axis.set_title(
        "Structural diff"
        f" · {len(comparison['matching_innovations'])} shared"
        f" · {len(comparison['changed_weights'])} reweighted",
        color=theme.foreground,
        fontsize=12,
        pad=14,
    )
    diff_axis.margins(0.16)
    diff_axis.axis("off")
    if legend_handles:
        legend = diff_axis.legend(
            handles=legend_handles,
            loc="best",
            frameon=False,
            fontsize=8,
        )
        for text in legend.get_texts():
            text.set_color(theme.foreground)
    figure.patch.set_facecolor(theme.background)
    if show:
        plt.show()
    return figure


def _population_rows(population: pnt.Population) -> list[dict[str, Any]]:
    rows = []
    for species in population.m_Species:
        for genome in species.m_Individuals:
            summary = genome_summary(genome)
            summary["species_id"] = species.ID()
            summary["species_color"] = (
                species.m_R / 255.0,
                species.m_G / 255.0,
                species.m_B / 255.0,
            )
            summary["evaluated"] = bool(genome.IsEvaluated())
            rows.append(summary)
    return rows


def species_summary(species: pnt.Species) -> dict[str, Any]:
    """Return fitness, complexity, and lifecycle metrics for one species."""

    genomes = list(species.m_Individuals)
    summaries = [genome_summary(genome) for genome in genomes]
    finite_fitness = np.asarray(
        [
            item["fitness"]
            for item in summaries
            if math.isfinite(float(item["fitness"]))
        ],
        dtype=float,
    )
    links = np.asarray([item["links"] for item in summaries], dtype=float)
    neurons = np.asarray([item["neurons"] for item in summaries], dtype=float)
    recurrent = np.asarray([item["recurrent_links"] for item in summaries], dtype=float)
    return {
        "id": int(species.ID()),
        "size": len(genomes),
        "evaluated": int(species.NumEvaluated()),
        "best_fitness": (
            float(np.max(finite_fitness)) if finite_fitness.size else None
        ),
        "mean_fitness": (
            float(np.mean(finite_fitness)) if finite_fitness.size else None
        ),
        "median_fitness": (
            float(np.median(finite_fitness)) if finite_fitness.size else None
        ),
        "fitness_std": (float(np.std(finite_fitness)) if finite_fitness.size else None),
        "historical_best_fitness": float(species.GetBestFitness()),
        "age_generations": int(species.AgeGens()),
        "stagnation_generations": int(species.GensNoImprovement()),
        "mean_links": float(np.mean(links)) if links.size else 0.0,
        "mean_neurons": (float(np.mean(neurons)) if neurons.size else 0.0),
        "mean_recurrent_links": (float(np.mean(recurrent)) if recurrent.size else 0.0),
        "color_rgb": (
            int(species.m_R),
            int(species.m_G),
            int(species.m_B),
        ),
    }


def population_summary(population: pnt.Population) -> dict[str, Any]:
    """Return population health, diversity, and complexity diagnostics."""

    rows = _population_rows(population)
    if not rows:
        raise ValueError("Cannot summarize an empty population")
    finite_fitness = np.asarray(
        [row["fitness"] for row in rows if math.isfinite(float(row["fitness"]))],
        dtype=float,
    )
    species = [species_summary(item) for item in population.m_Species]
    sizes = np.asarray([item["size"] for item in species], dtype=float)
    proportions = sizes / float(np.sum(sizes))
    entropy = float(
        -np.sum(proportions[proportions > 0.0] * np.log(proportions[proportions > 0.0]))
    )
    normalized_entropy = entropy / math.log(len(species)) if len(species) > 1 else 0.0
    links = np.asarray([row["links"] for row in rows], dtype=float)
    neurons = np.asarray([row["neurons"] for row in rows], dtype=float)
    recurrent = np.asarray([row["recurrent_links"] for row in rows], dtype=float)
    return {
        "generation": int(population.GetGeneration()),
        "population_size": len(rows),
        "evaluated": sum(bool(row["evaluated"]) for row in rows),
        "best_fitness": (
            float(np.max(finite_fitness)) if finite_fitness.size else None
        ),
        "worst_fitness": (
            float(np.min(finite_fitness)) if finite_fitness.size else None
        ),
        "mean_fitness": (
            float(np.mean(finite_fitness)) if finite_fitness.size else None
        ),
        "median_fitness": (
            float(np.median(finite_fitness)) if finite_fitness.size else None
        ),
        "fitness_std": (float(np.std(finite_fitness)) if finite_fitness.size else None),
        "species": len(species),
        "species_entropy": entropy,
        "normalized_species_entropy": normalized_entropy,
        "effective_species": float(math.exp(entropy)),
        "largest_species_fraction": float(np.max(proportions)),
        "mean_links": float(np.mean(links)),
        "links_std": float(np.std(links)),
        "mean_neurons": float(np.mean(neurons)),
        "neurons_std": float(np.std(neurons)),
        "mean_recurrent_links": float(np.mean(recurrent)),
        "compatibility_threshold": float(population.m_Parameters.CompatTreshold),
        "stagnation": int(population.GetStagnation()),
        "mean_population_complexity": float(population.GetCurrentMPC()),
        "species_detail": species,
    }


def DrawPopulation(
    population: pnt.Population,
    *,
    theme: VisualTheme = DEFAULT_THEME,
    show: bool = True,
) -> Any:
    """Draw a four-panel population/speciation dashboard."""

    rows = _population_rows(population)
    if not rows:
        raise ValueError("Cannot draw an empty population")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        constrained_layout=True,
    )
    figure.patch.set_facecolor(theme.background)
    for axis in axes.flat:
        axis.set_facecolor(theme.background)
        axis.tick_params(colors=theme.muted)
        for spine in axis.spines.values():
            spine.set_color(theme.grid)
        axis.grid(color=theme.grid, alpha=0.25)

    fitness = np.asarray([row["fitness"] for row in rows], dtype=float)
    links = np.asarray([row["links"] for row in rows], dtype=float)
    neurons = np.asarray([row["neurons"] for row in rows], dtype=float)
    colors = [row["species_color"] for row in rows]

    axes[0, 0].hist(
        fitness[np.isfinite(fitness)],
        bins="auto",
        color=theme.hidden_color,
        alpha=0.8,
    )
    axes[0, 0].set_title("Fitness distribution")
    axes[0, 0].set_xlabel("fitness")
    axes[0, 0].set_ylabel("genomes")

    axes[0, 1].scatter(
        links,
        fitness,
        s=30 + 4 * neurons,
        c=colors,
        alpha=0.8,
        edgecolors=theme.foreground,
        linewidths=0.4,
    )
    axes[0, 1].set_title("Fitness vs. topology complexity")
    axes[0, 1].set_xlabel("links")
    axes[0, 1].set_ylabel("fitness")

    species_rows: MutableMapping[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        species_rows[row["species_id"]].append(row)
    species_ids = sorted(species_rows)
    sizes = [len(species_rows[item]) for item in species_ids]
    species_colors = [species_rows[item][0]["species_color"] for item in species_ids]
    axes[1, 0].bar(
        [str(item) for item in species_ids],
        sizes,
        color=species_colors,
    )
    axes[1, 0].set_title("Species sizes")
    axes[1, 0].set_xlabel("species ID")
    axes[1, 0].set_ylabel("genomes")

    species_fitness = [
        np.asarray(
            [
                row["fitness"]
                for row in species_rows[item]
                if math.isfinite(float(row["fitness"]))
            ],
            dtype=float,
        )
        for item in species_ids
    ]
    best = [
        float(np.max(values)) if values.size else np.nan for values in species_fitness
    ]
    mean = [
        float(np.mean(values)) if values.size else np.nan for values in species_fitness
    ]
    x = np.arange(len(species_ids))
    axes[1, 1].plot(
        x,
        best,
        marker="o",
        color=theme.output_color,
        label="best",
    )
    axes[1, 1].plot(
        x,
        mean,
        marker=".",
        color=theme.hidden_color,
        label="mean",
    )
    axes[1, 1].set_xticks(x, [str(item) for item in species_ids])
    axes[1, 1].set_title("Species fitness")
    axes[1, 1].set_xlabel("species ID")
    axes[1, 1].set_ylabel("fitness")
    axes[1, 1].legend(frameon=False, labelcolor=theme.foreground)

    for axis in axes.flat:
        axis.title.set_color(theme.foreground)
        axis.xaxis.label.set_color(theme.foreground)
        axis.yaxis.label.set_color(theme.foreground)
    if show:
        plt.show()
    return figure


class EvolutionTracker:
    """Collect lightweight population metrics and plot their trajectories."""

    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []

    def record(
        self,
        population: pnt.Population,
        generation: int | None = None,
    ) -> dict[str, float]:
        summary = population_summary(population)
        if summary["best_fitness"] is None:
            raise ValueError("Cannot record a population without finite fitness")
        record = {
            "generation": float(
                len(self.history) if generation is None else generation
            ),
            "best_fitness": float(summary["best_fitness"]),
            "worst_fitness": float(summary["worst_fitness"]),
            "mean_fitness": float(summary["mean_fitness"]),
            "median_fitness": float(summary["median_fitness"]),
            "fitness_std": float(summary["fitness_std"]),
            "species": float(summary["species"]),
            "species_entropy": float(summary["species_entropy"]),
            "effective_species": float(summary["effective_species"]),
            "largest_species_fraction": float(summary["largest_species_fraction"]),
            "mean_links": float(summary["mean_links"]),
            "links_std": float(summary["links_std"]),
            "mean_neurons": float(summary["mean_neurons"]),
            "neurons_std": float(summary["neurons_std"]),
            "mean_recurrent_links": float(summary["mean_recurrent_links"]),
            "compatibility_threshold": float(summary["compatibility_threshold"]),
            "stagnation": float(summary["stagnation"]),
        }
        self.history.append(record)
        return record

    def draw(
        self,
        *,
        theme: VisualTheme = DEFAULT_THEME,
        show: bool = True,
    ) -> Any:
        return DrawEvolution(self.history, theme=theme, show=show)

    def interactive(self, *, title: str | None = None) -> Any:
        """Return an interactive Plotly view of the recorded trajectory."""

        return InteractiveEvolution(self.history, title=title)

    def save(self, path: str | Path) -> Path:
        """Write history as JSON or CSV, selected by the file suffix."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() == ".json":
            destination.write_text(
                json.dumps(self.history, indent=2),
                encoding="utf-8",
            )
        elif destination.suffix.lower() == ".csv":
            import csv

            keys = list(self.history[0]) if self.history else []
            with destination.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                if keys:
                    writer.writeheader()
                    writer.writerows(self.history)
        else:
            raise ValueError("Evolution history export requires .json or .csv")
        return destination


def DrawEvolution(
    history: Sequence[Mapping[str, float]],
    *,
    theme: VisualTheme = DEFAULT_THEME,
    show: bool = True,
) -> Any:
    """Plot fitness, diversity, and topology trends."""

    if not history:
        raise ValueError("DrawEvolution requires at least one record")
    generation = [row["generation"] for row in history]
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
        constrained_layout=True,
    )
    figure.patch.set_facecolor(theme.background)
    series = [
        (
            axes[0],
            (
                ("best_fitness", theme.output_color, "best"),
                ("mean_fitness", theme.hidden_color, "mean"),
                ("median_fitness", theme.input_color, "median"),
            ),
            "Fitness",
        ),
        (
            axes[1],
            (
                ("species", theme.bias_color, "species"),
                (
                    "effective_species",
                    theme.hidden_color,
                    "effective species",
                ),
            ),
            "Diversity",
        ),
        (
            axes[2],
            (
                ("mean_links", theme.negative_color, "mean links"),
                ("mean_neurons", theme.positive_color, "mean neurons"),
                (
                    "mean_recurrent_links",
                    theme.output_color,
                    "mean recurrent links",
                ),
            ),
            "Complexity",
        ),
    ]
    for axis, metrics, title in series:
        axis.set_facecolor(theme.background)
        for key, color, label in metrics:
            if key in history[0]:
                axis.plot(
                    generation,
                    [row[key] for row in history],
                    color=color,
                    label=label,
                    linewidth=2,
                )
        axis.set_ylabel(title, color=theme.foreground)
        axis.tick_params(colors=theme.muted)
        axis.grid(color=theme.grid, alpha=0.3)
        axis.legend(frameon=False, labelcolor=theme.foreground)
    axes[-1].set_xlabel("Generation", color=theme.foreground)
    if show:
        plt.show()
    return figure


def InteractiveEvolution(
    history: Sequence[Mapping[str, float]],
    *,
    title: str | None = None,
) -> Any:
    """Return a linked interactive view of evolutionary trajectories."""

    if not history:
        raise ValueError("InteractiveEvolution requires at least one record")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise ImportError(
            "InteractiveEvolution requires plotly (pip install plotly)."
        ) from error

    generation = [row["generation"] for row in history]
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Fitness", "Diversity", "Complexity"),
    )
    groups = (
        (
            1,
            (
                ("best_fitness", "best"),
                ("mean_fitness", "mean"),
                ("median_fitness", "median"),
            ),
        ),
        (
            2,
            (
                ("species", "species"),
                ("effective_species", "effective species"),
            ),
        ),
        (
            3,
            (
                ("mean_links", "mean links"),
                ("mean_neurons", "mean neurons"),
                ("mean_recurrent_links", "mean recurrent links"),
            ),
        ),
    )
    for row, metrics in groups:
        for key, label in metrics:
            if key not in history[0]:
                continue
            figure.add_trace(
                go.Scatter(
                    x=generation,
                    y=[item[key] for item in history],
                    mode="lines+markers",
                    name=label,
                    hovertemplate=(
                        "generation %{x}<br>" + label + " %{y:.6g}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
    figure.update_xaxes(title_text="Generation", row=3, col=1)
    figure.update_layout(
        title=title or "Evolution dynamics",
        template="plotly_dark",
        hovermode="x unified",
        height=850,
    )
    return figure


def InteractivePopulation(
    population: pnt.Population,
    *,
    title: str | None = None,
) -> Any:
    """Return an interactive population, species, and topology dashboard."""

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise ImportError(
            "InteractivePopulation requires plotly (pip install plotly)."
        ) from error

    rows = _population_rows(population)
    if not rows:
        raise ValueError("Cannot draw an empty population")
    summary = population_summary(population)
    finite_rows = [row for row in rows if math.isfinite(float(row["fitness"]))]
    species_detail = summary["species_detail"]
    figure = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.13,
        subplot_titles=(
            "Fitness vs. topology complexity",
            "Fitness distribution",
            "Species sizes",
            "Species fitness",
        ),
    )
    marker_sizes = [
        max(8.0, min(38.0, 6.0 + float(row["neurons"]))) for row in finite_rows
    ]
    marker_colors = [
        "rgb({},{},{})".format(
            round(255 * row["species_color"][0]),
            round(255 * row["species_color"][1]),
            round(255 * row["species_color"][2]),
        )
        for row in finite_rows
    ]
    figure.add_trace(
        go.Scatter(
            x=[row["links"] for row in finite_rows],
            y=[row["fitness"] for row in finite_rows],
            mode="markers",
            marker={
                "size": marker_sizes,
                "color": marker_colors,
                "line": {"width": 0.5},
                "opacity": 0.82,
            },
            customdata=[
                [
                    row["id"],
                    row["species_id"],
                    row["neurons"],
                    row["recurrent_links"],
                ]
                for row in finite_rows
            ],
            hovertemplate=(
                "genome %{customdata[0]}<br>"
                "species %{customdata[1]}<br>"
                "fitness %{y:.6g}<br>"
                "links %{x}<br>"
                "neurons %{customdata[2]}<br>"
                "recurrent %{customdata[3]}<extra></extra>"
            ),
            name="genomes",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Histogram(
            x=[row["fitness"] for row in finite_rows],
            name="fitness",
            showlegend=False,
            hovertemplate="fitness %{x:.6g}<br>genomes %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=[str(item["id"]) for item in species_detail],
            y=[item["size"] for item in species_detail],
            marker_color=[
                "rgb({},{},{})".format(*item["color_rgb"]) for item in species_detail
            ],
            customdata=[
                [
                    item["age_generations"],
                    item["stagnation_generations"],
                ]
                for item in species_detail
            ],
            hovertemplate=(
                "species %{x}<br>size %{y}<br>"
                "age %{customdata[0]}<br>"
                "stagnation %{customdata[1]}<extra></extra>"
            ),
            name="species size",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for metric, label in (
        ("best_fitness", "best"),
        ("mean_fitness", "mean"),
    ):
        figure.add_trace(
            go.Scatter(
                x=[str(item["id"]) for item in species_detail],
                y=[item[metric] for item in species_detail],
                mode="lines+markers",
                name=label,
                hovertemplate=(
                    "species %{x}<br>" + label + " fitness %{y:.6g}<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )
    figure.update_xaxes(title_text="Links", row=1, col=1)
    figure.update_yaxes(title_text="Fitness", row=1, col=1)
    figure.update_xaxes(title_text="Fitness", row=1, col=2)
    figure.update_yaxes(title_text="Genomes", row=1, col=2)
    figure.update_xaxes(title_text="Species ID", row=2, col=1)
    figure.update_yaxes(title_text="Genomes", row=2, col=1)
    figure.update_xaxes(title_text="Species ID", row=2, col=2)
    figure.update_yaxes(title_text="Fitness", row=2, col=2)
    figure.update_layout(
        title=title
        or (
            f"Population generation {summary['generation']} · "
            f"{summary['species']} species · "
            f"effective diversity {summary['effective_species']:.2f}"
        ),
        template="plotly_dark",
        height=850,
        barmode="overlay",
    )
    return figure


def InteractiveGenome(
    genome: pnt.Genome,
    *,
    layout: str = "auto",
    title: str | None = None,
) -> Any:
    """Return an interactive Plotly figure with rich node/edge hover data."""

    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "InteractiveGenome requires plotly (pip install plotly)."
        ) from error

    graph = Genome2NX(genome)
    positions = compute_node_positions(genome, layout=layout)
    edge_traces = []
    maximum_weight = max(
        1.0e-12,
        max(
            (abs(data["weight"]) for _, _, data in graph.edges(data=True)),
            default=1.0,
        ),
    )
    for source, target, data in graph.edges(data=True):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        color = (
            DEFAULT_THEME.positive_color
            if data["weight"] >= 0.0
            else DEFAULT_THEME.negative_color
        )
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={
                    "color": color,
                    "width": 1 + 5 * abs(data["weight"]) / maximum_weight,
                    "dash": "dash" if data["is_recurrent"] else "solid",
                },
                hoverinfo="text",
                text=(
                    f"innovation {data['innovation_id']}<br>"
                    f"{source} → {target}<br>"
                    f"weight {data['weight']:.6g}<br>"
                    f"recurrent {data['is_recurrent']}"
                ),
                showlegend=False,
            )
        )

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    type_colors = {
        INPUT: DEFAULT_THEME.input_color,
        BIAS: DEFAULT_THEME.bias_color,
        HIDDEN: DEFAULT_THEME.hidden_color,
        OUTPUT: DEFAULT_THEME.output_color,
    }
    for node, data in graph.nodes(data=True):
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(type_colors.get(data["type"], DEFAULT_THEME.muted))
        traits = "<br>".join(f"{key}: {value}" for key, value in data["traits"].items())
        node_text.append(
            f"neuron {node}<br>{data['type_name']}<br>"
            f"{data['activation_name']}<br>"
            f"a={data['a']:.4g}, b={data['b']:.4g}, "
            f"bias={data['bias']:.4g}" + (f"<br>{traits}" if traits else "")
        )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(node) for node in graph.nodes],
        textposition="middle center",
        hovertext=node_text,
        hoverinfo="text",
        marker={
            "size": 30,
            "color": node_colors,
            "line": {"color": DEFAULT_THEME.foreground, "width": 1},
        },
        showlegend=False,
    )
    figure = go.Figure(data=[*edge_traces, node_trace])
    figure.update_layout(
        title=title or f"Genome {genome.GetID()} · fitness {genome.GetFitness():.6g}",
        template="plotly_dark",
        xaxis={"visible": False},
        yaxis={"visible": False},
        hovermode="closest",
        showlegend=False,
    )
    return figure


def narrate_traits(genome: pnt.Genome) -> None:
    """Print genome, neuron, and link traits in a readable form."""

    print("Genome traits")
    traits = _traits(genome.m_GenomeGene.m_Traits)
    print(json.dumps(traits, indent=2, ensure_ascii=False))
    print("\nNeuron traits")
    for neuron in genome.m_NeuronGenes:
        print(
            f"  {neuron.m_ID} ({_TYPE_NAMES.get(neuron.m_Type, neuron.m_Type)}): "
            f"{_traits(neuron.m_Traits)}"
        )
    print("\nLink traits")
    for link in genome.m_LinkGenes:
        print(
            f"  #{link.m_InnovationID} "
            f"{link.m_FromNeuronID} -> {link.m_ToNeuronID}: "
            f"{_traits(link.m_Traits)}"
        )


def export_genome_graph(
    genome: pnt.Genome,
    filename: str | Path,
) -> Path:
    """Export DOT, GraphML, GEXF, JSON, SVG, PNG, PDF, or interactive HTML."""

    destination = Path(filename)
    suffix = destination.suffix.lower()
    graph = Genome2NX(genome)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".dot":
        try:
            from networkx.drawing.nx_pydot import write_dot
        except ImportError as error:
            raise ImportError(
                "DOT export requires pydot (pip install pydot)."
            ) from error
        serializable = _serializable_graph(graph)
        write_dot(serializable, destination)
    elif suffix == ".graphml":
        nx.write_graphml(_serializable_graph(graph), destination)
    elif suffix == ".gexf":
        nx.write_gexf(_serializable_graph(graph), destination)
    elif suffix == ".json":
        payload = nx.node_link_data(graph)
        destination.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
    elif suffix == ".html":
        InteractiveGenome(genome).write_html(destination, include_plotlyjs="cdn")
    elif suffix in {".svg", ".png", ".pdf"}:
        axis = DrawGenome(genome, show=False)
        axis.figure.savefig(
            destination,
            dpi=180,
            bbox_inches="tight",
            facecolor=axis.figure.get_facecolor(),
        )
        plt.close(axis.figure)
    else:
        raise ValueError(
            "Unsupported export extension. Use .dot, .graphml, .gexf, "
            ".json, .svg, .png, .pdf, or .html."
        )
    return destination


def _serializable_graph(graph: nx.DiGraph) -> nx.DiGraph:
    result = nx.DiGraph()
    result.graph.update(
        {
            key: value
            for key, value in graph.graph.items()
            if isinstance(value, (str, int, float, bool))
        }
    )
    for node, data in graph.nodes(data=True):
        result.add_node(
            node,
            **{key: _serializable_attribute(value) for key, value in data.items()},
        )
    for source, target, data in graph.edges(data=True):
        result.add_edge(
            source,
            target,
            **{key: _serializable_attribute(value) for key, value in data.items()},
        )
    return result


def _serializable_attribute(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def print_genome_summary(genome: pnt.Genome) -> None:
    """Print the complete :func:`genome_summary` result."""

    print(
        json.dumps(
            genome_summary(genome),
            indent=2,
            ensure_ascii=False,
        )
    )


__all__ = [
    "DEFAULT_THEME",
    "DrawEvolution",
    "DrawGenome",
    "DrawGenomeComparison",
    "DrawGenomes",
    "DrawPopulation",
    "EvolutionTracker",
    "Genome2NX",
    "InteractiveEvolution",
    "InteractiveGenome",
    "InteractivePopulation",
    "VisualTheme",
    "compare_genomes",
    "compute_node_positions",
    "export_genome_graph",
    "genome_summary",
    "get_layered_nodes",
    "get_topologically_sorted_nodes",
    "narrate_traits",
    "population_summary",
    "print_genome_summary",
    "species_summary",
]
