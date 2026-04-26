# Databricks notebook source
# MAGIC %md
# MAGIC # Flight Route Optimization Demo
# MAGIC
# MAGIC This notebook extends the original flight notebooks from graph exploration
# MAGIC into route optimization and route recommendation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data

# COMMAND ----------

import sys
from pathlib import Path

workspace_module_dir = Path("/Workspace/Users/nbaduc137@gmail.com/graph_mining_optimize_flight/src/optimization")
local_repo_root = next(
    (
        candidate
        for candidate in [Path.cwd(), *Path.cwd().parents]
        if (candidate / "project" / "src" / "optimization" / "flight_route_optimization.py").exists()
    ),
    None,
)

if workspace_module_dir.exists():
    sys.path.append(str(workspace_module_dir))
elif local_repo_root is not None:
    sys.path.append(str(local_repo_root / "project" / "src" / "optimization"))
else:
    raise ImportError("Cannot find flight_route_optimization.py in Workspace Files or local project/src/optimization")

from flight_route_optimization import (
    aggregate_routes,
    betweenness_centrality,
    build_graph,
    compare_route_strategies,
    compute_normalization_stats,
    degree_centrality,
    get_weight_fn,
    load_departure_delays,
    pagerank_centrality,
    top_n_scores,
    yen_k_shortest_paths,
)

DATA_PATH = Path("/Workspace/Users/nbaduc137@gmail.com/graph_mining_optimize_flight/data/departuredelays.csv")

flights = load_departure_delays(DATA_PATH)
routes = aggregate_routes(flights)
stats = compute_normalization_stats(routes)
graph = build_graph(routes)

print("Data path:", DATA_PATH)
print("Flights rows:", len(flights))
print("Routes:", len(routes))
routes.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build Graph
# MAGIC
# MAGIC - Node: airport
# MAGIC - Edge: route `origin -> destination`
# MAGIC - Route-level features: distance, mean delay, trip count, on-time ratio

# COMMAND ----------

print("Number of airports:", len(graph))
print("Sample routes:")
routes[["origin", "destination", "distance", "mean_delay", "trip_count"]].head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Visualize Graph
# MAGIC
# MAGIC If you want a quick Databricks visualization, display the route table and
# MAGIC use built-in charting. For a local matplotlib/networkx view, use the code
# MAGIC snippet in the visualization section below.

# COMMAND ----------

display(routes[["origin", "destination", "distance", "mean_delay", "trip_count"]].head(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run Shortest Path and K-Shortest Paths

# COMMAND ----------

source = "SEA"
target = "BUF"
alpha = 0.45
beta = 0.35
gamma = 0.20

weight_fn = get_weight_fn("composite", stats, alpha=alpha, beta=beta, gamma=gamma)
top_k_paths = yen_k_shortest_paths(graph, source, target, k=5, weight_fn=weight_fn)

for idx, result in enumerate(top_k_paths, start=1):
    print(f"[{idx}] {' -> '.join(result.path)}")
    print("    total_cost =", round(result.total_cost, 4))
    print("    total_distance =", round(result.total_distance, 2))
    print("    total_mean_delay =", round(result.total_mean_delay, 2))
    print("    stops =", result.stops)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare Results

# COMMAND ----------

comparison = compare_route_strategies(
    graph,
    stats,
    source,
    target,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
)
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Hub Analysis
# MAGIC
# MAGIC We compute:
# MAGIC - PageRank
# MAGIC - Degree centrality
# MAGIC - Betweenness centrality

# COMMAND ----------

pagerank_df = top_n_scores(pagerank_centrality(graph), top_n=10)
degree_df = top_n_scores(degree_centrality(graph), top_n=10)
betweenness_df = top_n_scores(betweenness_centrality(graph), top_n=10)

display(spark.createDataFrame(pagerank_df, ["airport", "pagerank"]))
display(spark.createDataFrame(degree_df, ["airport", "degree_centrality"]))
display(spark.createDataFrame(betweenness_df, ["airport", "betweenness_centrality"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Kết luận
# MAGIC
# MAGIC - `hops` gives the least-stop route
# MAGIC - `distance` minimizes total route length
# MAGIC - `composite` balances distance, delay, and route frequency
# MAGIC - hub airports strongly influence feasible and recommended routes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization Snippet
# MAGIC
# MAGIC If your environment has `networkx` and `matplotlib`, you can visualize the
# MAGIC recommended path like this:
# MAGIC
# MAGIC ```python
# MAGIC import networkx as nx
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC g = nx.DiGraph()
# MAGIC for src, edges in graph.items():
# MAGIC     for edge in edges:
# MAGIC         g.add_edge(edge.src, edge.dst, weight=edge.trip_count)
# MAGIC
# MAGIC best = top_k_paths[0].path
# MAGIC best_edges = list(zip(best[:-1], best[1:]))
# MAGIC pos = nx.spring_layout(g, seed=42)
# MAGIC
# MAGIC plt.figure(figsize=(14, 10))
# MAGIC nx.draw_networkx_nodes(g, pos, node_size=60, alpha=0.7)
# MAGIC nx.draw_networkx_edges(g, pos, alpha=0.15, arrows=False)
# MAGIC nx.draw_networkx_edges(g, pos, edgelist=best_edges, edge_color="red", width=2.5, arrows=True)
# MAGIC nx.draw_networkx_labels(g, pos, font_size=7)
# MAGIC plt.title("Best recommended route highlighted in red")
# MAGIC plt.axis("off")
# MAGIC plt.show()
# MAGIC ```

# COMMAND ----------

import networkx as nx
import matplotlib.pyplot as plt

g = nx.DiGraph()
for src, edges in graph.items():
    for edge in edges:
        g.add_edge(edge.src, edge.dst, weight=edge.trip_count)

best = top_k_paths[0].path
best_edges = list(zip(best[:-1], best[1:]))
pos = nx.spring_layout(g, seed=42)

plt.figure(figsize=(14, 10))
nx.draw_networkx_nodes(g, pos, node_size=60, alpha=0.7)
nx.draw_networkx_edges(g, pos, alpha=0.15, arrows=False)
nx.draw_networkx_edges(g, pos, edgelist=best_edges, edge_color="red", width=2.5, arrows=True)
nx.draw_networkx_labels(g, pos, font_size=7)
plt.title("Best recommended route highlighted in red")
plt.axis("off")
plt.show()

# COMMAND ----------

# MAGIC %pip install networkx