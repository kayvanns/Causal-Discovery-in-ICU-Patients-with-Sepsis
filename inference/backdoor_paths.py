import pandas as pd
import networkx as nx
from dowhy import CausalModel


table = pd.read_csv("results/v3/ensemble_table.csv")


high_agreement = table[table["agreement_score"] >= 6]


G = nx.DiGraph()
for _, row in high_agreement.iterrows():
    G.add_edge(row["cause"], row["effect"])

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")

# edges = list(G.edges())
# dot_graph = "digraph { " + " ".join([f'"{c}" -> "{e}";' for c, e in edges]) + " }"

# df = pd.read_csv("data/processed/analysis_cleaned.csv")
# df = df[[c for c in G.nodes() if c in df.columns]]

# model = CausalModel(
#     data=df,
#     treatment="mechvent_24h_onset",
#     outcome="aki_post24h_stage",
#     graph=dot_graph
# )

# identified_estimand = model.identify_effect()
# print(identified_estimand)