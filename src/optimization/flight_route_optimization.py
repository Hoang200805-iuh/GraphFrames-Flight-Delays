"""
Flight route optimization on top of the sample Databricks departure delays data.

This module upgrades the original repository from graph exploration into a
reusable route-optimization layer:
- weighted shortest path with Dijkstra
- multi-objective cost with alpha / beta / gamma
- Yen's K-shortest paths for route recommendation
- hub analysis with PageRank, degree centrality, and betweenness centrality

Example
-------
python py/flight_route_optimization.py --source SEA --target BUF --objective composite --top-k 3
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # __file__ is not defined in notebook/REPL contexts
    PROJECT_ROOT = Path("/Workspace/Users/nbaduc137@gmail.com/graph_mining_optimize_flight")

REPO_ROOT = PROJECT_ROOT.parent


def _resolve_default_data_path() -> Path:
    candidates = [
        Path("/Workspace/Users/nbaduc137@gmail.com/graph_mining_optimize_flight/data/departuredelays.csv"),
        REPO_ROOT / "Graph Mining - On Time Flight" / "flight-data" / "departuredelays.csv",
        PROJECT_ROOT / "data" / "flights" / "departuredelays.csv.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DATA_PATH = _resolve_default_data_path()


@dataclass(frozen=True)
class RouteEdge:
    src: str
    dst: str
    distance: float
    mean_delay: float
    positive_delay_ratio: float
    mean_positive_delay: float
    trip_count: int
    on_time_ratio: float


@dataclass(frozen=True)
class NormalizationStats:
    min_distance: float
    max_distance: float
    min_delay: float
    max_delay: float
    max_trip_count: int


@dataclass(frozen=True)
class PathResult:
    path: List[str]
    total_cost: float
    total_distance: float
    total_mean_delay: float
    stops: int


Graph = Dict[str, List[RouteEdge]]
WeightFn = Callable[[RouteEdge], float]


def load_departure_delays(csv_path: Path) -> pd.DataFrame:
    """Load the sample departure delays data shipped in this repository."""
    df = pd.read_csv(csv_path, compression="infer")
    required = {"date", "delay", "distance", "origin", "destination"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["delay"] = pd.to_numeric(df["delay"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df.dropna(subset=["delay", "distance", "origin", "destination"])
    df["origin"] = df["origin"].astype(str).str.upper()
    df["destination"] = df["destination"].astype(str).str.upper()
    return df


def aggregate_routes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw flights into route-level statistics.

    Each directed route (origin -> destination) becomes one weighted edge.
    """
    enriched = df.copy()
    enriched["is_on_time"] = (enriched["delay"] <= 0).astype(int)
    enriched["positive_delay"] = enriched["delay"].clip(lower=0)
    enriched["is_positive_delay"] = (enriched["delay"] > 0).astype(int)

    grouped = (
        enriched.groupby(["origin", "destination"], as_index=False)
        .agg(
            distance=("distance", "mean"),
            mean_delay=("delay", "mean"),
            positive_delay_ratio=("is_positive_delay", "mean"),
            mean_positive_delay=("positive_delay", "mean"),
            trip_count=("delay", "size"),
            on_time_ratio=("is_on_time", "mean"),
        )
        .sort_values(["origin", "destination"])
        .reset_index(drop=True)
    )
    return grouped


def compute_normalization_stats(routes: pd.DataFrame) -> NormalizationStats:
    """Compute global min/max statistics used by the composite cost."""
    non_negative_delay = routes["mean_delay"].clip(lower=0.0)
    return NormalizationStats(
        min_distance=float(routes["distance"].min()),
        max_distance=float(routes["distance"].max()),
        min_delay=float(non_negative_delay.min()),
        max_delay=float(non_negative_delay.max()),
        max_trip_count=int(routes["trip_count"].max()),
    )


def build_graph(routes: pd.DataFrame) -> Graph:
    """Convert the aggregated route table into an adjacency-list graph."""
    graph: Graph = {}
    for row in routes.itertuples(index=False):
        edge = RouteEdge(
            src=row.origin,
            dst=row.destination,
            distance=float(row.distance),
            mean_delay=float(row.mean_delay),
            positive_delay_ratio=float(row.positive_delay_ratio),
            mean_positive_delay=float(row.mean_positive_delay),
            trip_count=int(row.trip_count),
            on_time_ratio=float(row.on_time_ratio),
        )
        graph.setdefault(edge.src, []).append(edge)
        graph.setdefault(edge.dst, [])
    return graph


def min_max_normalize(value: float, min_value: float, max_value: float) -> float:
    """Min-max normalize into [0, 1]."""
    if max_value <= min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def normalized_frequency(edge: RouteEdge, stats: NormalizationStats) -> float:
    """
    Normalize route frequency to (0, 1].

    We use trip_count / max_trip_count so more frequent routes have a lower cost
    after applying the inverse-frequency penalty.
    """
    if stats.max_trip_count <= 0:
        return 1.0
    return max(edge.trip_count / float(stats.max_trip_count), 1e-9)


def hop_weight(_: RouteEdge) -> float:
    return 1.0


def distance_weight(edge: RouteEdge) -> float:
    return edge.distance


def delay_weight(edge: RouteEdge) -> float:
    return max(edge.mean_delay, 0.0)


def make_composite_weight(
    stats: NormalizationStats,
    alpha: float = 0.45,
    beta: float = 0.35,
    gamma: float = 0.20,
) -> WeightFn:
    """
    Create a multi-objective cost function.

    cost = alpha * normalized_distance
         + beta  * normalized_delay
         + gamma * (1 / normalized_frequency)

    Normalization:
    - normalized_distance = min-max(distance)
    - normalized_delay = min-max(max(mean_delay, 0))
    - normalized_frequency = trip_count / max_trip_count
    """
    total = alpha + beta + gamma
    if total <= 0:
        raise ValueError("alpha + beta + gamma must be > 0")

    alpha /= total
    beta /= total
    gamma /= total

    def weight(edge: RouteEdge) -> float:
        dist = min_max_normalize(edge.distance, stats.min_distance, stats.max_distance)
        delay = min_max_normalize(max(edge.mean_delay, 0.0), stats.min_delay, stats.max_delay)
        freq = normalized_frequency(edge, stats)
        return alpha * dist + beta * delay + gamma * (1.0 / freq)

    return weight


def get_weight_fn(
    objective: str,
    stats: NormalizationStats,
    alpha: float = 0.45,
    beta: float = 0.35,
    gamma: float = 0.20,
) -> WeightFn:
    """Resolve a named objective into a weight function."""
    if objective == "hops":
        return hop_weight
    if objective == "distance":
        return distance_weight
    if objective == "delay":
        return delay_weight
    if objective == "composite":
        return make_composite_weight(stats, alpha=alpha, beta=beta, gamma=gamma)
    raise ValueError(f"Unknown objective: {objective}")


def get_edge_lookup(graph: Graph) -> Dict[Tuple[str, str], RouteEdge]:
    """Build a fast lookup from (src, dst) to edge metadata."""
    return {(edge.src, edge.dst): edge for edges in graph.values() for edge in edges}


def _dijkstra_search(
    graph: Graph,
    source: str,
    target: str,
    weight_fn: WeightFn,
    banned_nodes: Optional[Set[str]] = None,
    banned_edges: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[float, List[str], List[RouteEdge]]:
    """Internal Dijkstra that supports banning nodes and edges."""
    if source not in graph:
        raise ValueError(f"Unknown source airport: {source}")
    if target not in graph:
        raise ValueError(f"Unknown target airport: {target}")

    banned_nodes = banned_nodes or set()
    banned_edges = banned_edges or set()
    if source in banned_nodes or target in banned_nodes:
        raise ValueError("Source/target cannot be in banned_nodes")

    pq: List[Tuple[float, str]] = [(0.0, source)]
    best_cost: Dict[str, float] = {source: 0.0}
    previous: Dict[str, RouteEdge] = {}
    visited: Set[str] = set()

    while pq:
        cost, node = heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == target:
            break

        for edge in graph.get(node, []):
            if edge.dst in banned_nodes:
                continue
            if (edge.src, edge.dst) in banned_edges:
                continue

            candidate = cost + weight_fn(edge)
            if candidate < best_cost.get(edge.dst, float("inf")):
                best_cost[edge.dst] = candidate
                previous[edge.dst] = edge
                heappush(pq, (candidate, edge.dst))

    if source == target:
        return 0.0, [source], []
    if target not in previous:
        raise ValueError(f"No path found from {source} to {target}")

    path_edges: List[RouteEdge] = []
    path_nodes: List[str] = [target]
    current = target
    while current != source:
        edge = previous[current]
        path_edges.append(edge)
        current = edge.src
        path_nodes.append(current)

    path_edges.reverse()
    path_nodes.reverse()
    return best_cost[target], path_nodes, path_edges


def dijkstra_shortest_path(
    graph: Graph,
    source: str,
    target: str,
    weight_fn: WeightFn,
) -> Tuple[float, List[RouteEdge]]:
    """Public single shortest-path API, kept compatible with the existing repo."""
    cost, _, edges = _dijkstra_search(graph, source, target, weight_fn)
    return cost, edges


def compute_path_cost(path: Sequence[str], edge_lookup: Dict[Tuple[str, str], RouteEdge], weight_fn: WeightFn) -> float:
    """Compute the total path cost from a node list."""
    total = 0.0
    for src, dst in zip(path[:-1], path[1:]):
        total += weight_fn(edge_lookup[(src, dst)])
    return total


def summarize_node_path(
    path: Sequence[str],
    edge_lookup: Dict[Tuple[str, str], RouteEdge],
    weight_fn: WeightFn,
) -> PathResult:
    """Convert a node path into a richer result object."""
    if len(path) <= 1:
        return PathResult(path=list(path), total_cost=0.0, total_distance=0.0, total_mean_delay=0.0, stops=0)

    distance = 0.0
    delay = 0.0
    for src, dst in zip(path[:-1], path[1:]):
        edge = edge_lookup[(src, dst)]
        distance += edge.distance
        delay += edge.mean_delay

    return PathResult(
        path=list(path),
        total_cost=compute_path_cost(path, edge_lookup, weight_fn),
        total_distance=distance,
        total_mean_delay=delay,
        stops=max(len(path) - 2, 0),
    )


def summarize_path(path_edges: Iterable[RouteEdge], score: float) -> Dict[str, object]:
    """Backward-compatible summary helper for the original single-path workflow."""
    edges = list(path_edges)
    if not edges:
        return {
            "path": [],
            "score": score,
            "total_distance": 0.0,
            "total_mean_delay": 0.0,
            "stops": 0,
        }

    airports = [edges[0].src] + [edge.dst for edge in edges]
    return {
        "path": airports,
        "score": score,
        "total_distance": sum(edge.distance for edge in edges),
        "total_mean_delay": sum(edge.mean_delay for edge in edges),
        "stops": max(len(airports) - 2, 0),
    }


def yen_k_shortest_paths(
    graph: Graph,
    source: str,
    target: str,
    k: int,
    weight_fn: WeightFn,
) -> List[PathResult]:
    """
    Compute the top-k shortest simple paths using Yen's algorithm.

    This implementation is intentionally built on top of our local Dijkstra,
    without relying on third-party shortest-path libraries.
    """
    if k <= 0:
        return []

    edge_lookup = get_edge_lookup(graph)
    first_cost, first_path_nodes, _ = _dijkstra_search(graph, source, target, weight_fn)
    first_result = summarize_node_path(first_path_nodes, edge_lookup, weight_fn)
    first_result = PathResult(
        path=first_result.path,
        total_cost=first_cost,
        total_distance=first_result.total_distance,
        total_mean_delay=first_result.total_mean_delay,
        stops=first_result.stops,
    )

    accepted: List[PathResult] = [first_result]
    candidates: List[Tuple[float, Tuple[str, ...], PathResult]] = []
    seen_candidates: Set[Tuple[str, ...]] = {tuple(first_result.path)}

    for kth in range(1, k):
        previous_best = accepted[kth - 1]
        previous_path = previous_best.path

        for i in range(len(previous_path) - 1):
            spur_node = previous_path[i]
            root_path = previous_path[: i + 1]

            banned_edges: Set[Tuple[str, str]] = set()
            for accepted_path in accepted:
                if len(accepted_path.path) > i and accepted_path.path[: i + 1] == root_path:
                    banned_edges.add((accepted_path.path[i], accepted_path.path[i + 1]))

            banned_nodes = set(root_path[:-1])

            try:
                _, spur_nodes, _ = _dijkstra_search(
                    graph,
                    spur_node,
                    target,
                    weight_fn,
                    banned_nodes=banned_nodes,
                    banned_edges=banned_edges,
                )
            except ValueError:
                continue

            total_path = root_path[:-1] + spur_nodes
            total_path_key = tuple(total_path)
            if total_path_key in seen_candidates:
                continue

            result = summarize_node_path(total_path, edge_lookup, weight_fn)
            heappush(candidates, (result.total_cost, total_path_key, result))
            seen_candidates.add(total_path_key)

        if not candidates:
            break

        _, _, next_best = heappop(candidates)
        accepted.append(next_best)

    return accepted


def pagerank_centrality(
    graph: Graph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Weighted PageRank using trip frequency as transition strength."""
    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return {}

    ranks = {node: 1.0 / n for node in nodes}
    out_weight_sum = {
        node: sum(max(edge.trip_count, 1) for edge in edges)
        for node, edges in graph.items()
    }

    for _ in range(max_iter):
        updated = {node: (1.0 - damping) / n for node in nodes}
        dangling_mass = sum(ranks[node] for node in nodes if out_weight_sum.get(node, 0) == 0)
        dangling_share = damping * dangling_mass / n

        for node in nodes:
            updated[node] += dangling_share

        for src, edges in graph.items():
            if out_weight_sum.get(src, 0) == 0:
                continue
            total_out = out_weight_sum[src]
            for edge in edges:
                weight = max(edge.trip_count, 1) / total_out
                updated[edge.dst] += damping * ranks[src] * weight

        error = sum(abs(updated[node] - ranks[node]) for node in nodes)
        ranks = updated
        if error < tol:
            break

    return ranks


def degree_centrality(graph: Graph) -> Dict[str, float]:
    """Directed degree centrality based on in-degree + out-degree."""
    nodes = list(graph.keys())
    n = len(nodes)
    if n <= 1:
        return {node: 0.0 for node in nodes}

    in_degree = {node: 0 for node in nodes}
    out_degree = {node: 0 for node in nodes}
    for src, edges in graph.items():
        out_degree[src] = len(edges)
        for edge in edges:
            in_degree[edge.dst] = in_degree.get(edge.dst, 0) + 1

    normalizer = 2.0 * (n - 1)
    return {node: (in_degree[node] + out_degree[node]) / normalizer for node in nodes}


def betweenness_centrality(graph: Graph) -> Dict[str, float]:
    """
    Brandes algorithm for directed unweighted graphs.

    For hub discovery, hop-based intermediary importance is often a good first
    signal even before introducing weighted variants.
    """
    nodes = list(graph.keys())
    centrality = {node: 0.0 for node in nodes}
    if len(nodes) <= 2:
        return centrality

    neighbors = {node: [edge.dst for edge in edges] for node, edges in graph.items()}

    for source in nodes:
        stack: List[str] = []
        predecessors = {node: [] for node in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        sigma[source] = 1.0
        distance = dict.fromkeys(nodes, -1)
        distance[source] = 0
        queue: List[str] = [source]

        while queue:
            vertex = queue.pop(0)
            stack.append(vertex)
            for neighbor in neighbors.get(vertex, []):
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[vertex] + 1
                if distance[neighbor] == distance[vertex] + 1:
                    sigma[neighbor] += sigma[vertex]
                    predecessors[neighbor].append(vertex)

        dependency = dict.fromkeys(nodes, 0.0)
        while stack:
            vertex = stack.pop()
            for predecessor in predecessors[vertex]:
                if sigma[vertex] != 0:
                    dependency[predecessor] += (sigma[predecessor] / sigma[vertex]) * (1.0 + dependency[vertex])
            if vertex != source:
                centrality[vertex] += dependency[vertex]

    scale = 1.0 / ((len(nodes) - 1) * (len(nodes) - 2))
    return {node: value * scale for node, value in centrality.items()}


def top_n_scores(scores: Dict[str, float], top_n: int = 10) -> List[Tuple[str, float]]:
    """Sort centrality scores descending and keep the top-N."""
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


def compare_route_strategies(
    graph: Graph,
    stats: NormalizationStats,
    source: str,
    target: str,
    alpha: float = 0.45,
    beta: float = 0.35,
    gamma: float = 0.20,
) -> pd.DataFrame:
    """Build a compact comparison table for the three core routing strategies."""
    edge_lookup = get_edge_lookup(graph)
    rows = []
    for objective in ["hops", "distance", "composite"]:
        weight_fn = get_weight_fn(objective, stats, alpha=alpha, beta=beta, gamma=gamma)
        _, path_nodes, _ = _dijkstra_search(graph, source, target, weight_fn)
        result = summarize_node_path(path_nodes, edge_lookup, weight_fn)
        rows.append(
            {
                "objective": objective,
                "path": " -> ".join(result.path),
                "total_cost": round(result.total_cost, 4),
                "total_distance": round(result.total_distance, 2),
                "total_mean_delay": round(result.total_mean_delay, 2),
                "stops": result.stops,
            }
        )
    return pd.DataFrame(rows)


def format_path_result(result: PathResult, index: Optional[int] = None) -> List[str]:
    """Pretty-print a path result as terminal-friendly lines."""
    prefix = f"[{index}] " if index is not None else ""
    return [
        f"{prefix}Path: {' -> '.join(result.path)}",
        f"{prefix}Total cost: {result.total_cost:.4f}",
        f"{prefix}Total distance: {result.total_distance:.2f}",
        f"{prefix}Total mean delay sum: {result.total_mean_delay:.2f}",
        f"{prefix}Stops: {result.stops}",
    ]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flight route optimization on sample departure delay data")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to departuredelays.csv or departuredelays.csv.gz")
    parser.add_argument("--source", type=str, default="SEA", help="Origin airport IATA")
    parser.add_argument("--target", type=str, default="BUF", help="Destination airport IATA")
    parser.add_argument(
        "--objective",
        type=str,
        default="composite",
        choices=["hops", "distance", "delay", "composite"],
        help="Optimization objective",
    )
    parser.add_argument("--top-k", type=int, default=1, help="Number of recommended routes")
    parser.add_argument("--alpha", type=float, default=0.45, help="Composite distance weight")
    parser.add_argument("--beta", type=float, default=0.35, help="Composite delay weight")
    parser.add_argument("--gamma", type=float, default=0.20, help="Composite frequency penalty weight")
    parser.add_argument("--show-centrality", action="store_true", help="Print top hub airports")
    parser.add_argument("--compare-objectives", action="store_true", help="Print comparison table for hops/distance/composite")
    args, _unknown = parser.parse_known_args(argv)
    return args


def main() -> None:
    args = parse_args()
    source = args.source.upper()
    target = args.target.upper()

    flights = load_departure_delays(args.data)
    routes = aggregate_routes(flights)
    stats = compute_normalization_stats(routes)
    graph = build_graph(routes)
    weight_fn = get_weight_fn(args.objective, stats, alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    if args.top_k <= 1:
        score, path_edges = dijkstra_shortest_path(graph, source, target, weight_fn)
        summary = summarize_path(path_edges, score)
        result = PathResult(
            path=list(summary["path"]),
            total_cost=float(summary["score"]),
            total_distance=float(summary["total_distance"]),
            total_mean_delay=float(summary["total_mean_delay"]),
            stops=int(summary["stops"]),
        )
        print(f"Objective: {args.objective}")
        for line in format_path_result(result):
            print(line)
    else:
        print(f"Objective: {args.objective}")
        print(f"Top-k route recommendation (k={args.top_k})")
        results = yen_k_shortest_paths(graph, source, target, args.top_k, weight_fn)
        for idx, result in enumerate(results, start=1):
            for line in format_path_result(result, index=idx):
                print(line)

    if args.compare_objectives:
        print("\nRoute strategy comparison:")
        comparison = compare_route_strategies(
            graph,
            stats,
            source,
            target,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )
        print(comparison.to_string(index=False))

    if args.show_centrality:
        print("\nTop airports by PageRank:")
        for airport, score in top_n_scores(pagerank_centrality(graph), top_n=10):
            print(f"  {airport}: {score:.6f}")

        print("\nTop airports by Degree Centrality:")
        for airport, score in top_n_scores(degree_centrality(graph), top_n=10):
            print(f"  {airport}: {score:.6f}")

        print("\nTop airports by Betweenness Centrality:")
        for airport, score in top_n_scores(betweenness_centrality(graph), top_n=10):
            print(f"  {airport}: {score:.6f}")


if __name__ == "__main__":
    main()
