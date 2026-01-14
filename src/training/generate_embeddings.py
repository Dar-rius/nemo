# import pandas as pd
# import networkx as nx
# from karateclub import Node2Vec
# from py2neo import Graph
# from pathlib import Path

# # Connexion
# creds_path = Path("src/config/credentials.txt")
# creds = dict(line.strip().split("=") for line in open(creds_path) if "=" in line)
# graph = Graph(creds["bolt_url"], auth=(creds["username"], creds["password"]))

# # Extraction complète du graphe
# query = """
# MATCH (n)
# RETURN id(n) AS node_id, 
#        head(labels(n)) AS main_label,
#        coalesce(n.name, n.id, 'UNKNOWN') AS display_name
# """
# nodes_df = graph.run(query).to_data_frame()

# query_edges = """
# MATCH (a)-[r]->(b)
# RETURN id(a) AS source, id(b) AS target, type(r) AS rel_type
# """
# edges_df = graph.run(query_edges).to_data_frame()

# # Construction NetworkX
# G = nx.Graph()
# for _, row in nodes_df.iterrows():
#     G.add_node(row['node_id'], label=row['main_label'], name=row['display_name'])

# for _, row in edges_df.iterrows():
#     G.add_edge(row['source'], row['target'])  # non orienté pour Node2Vec

# print(f"Graphe: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

# # Embeddings
# model = Node2Vec(dimensions=64, walk_length=30, window_size=5, workers=4)
# model.fit(G)
# emb = model.get_embedding()

# # Mapping
# node_ids = list(G.nodes())
# emb_df = pd.DataFrame(emb, index=node_ids)
# nodes_indexed = nodes_df.set_index('node_id')
# result_df = emb_df.join(nodes_indexed[['name', 'label']])

# # Sauvegarde
# Path("results/embeddings").mkdir(parents=True, exist_ok=True)
# result_df.reset_index().rename(columns={'index': 'node_id'}).to_csv(
#     "results/embeddings/node2vec.csv", index=False
# )
# print("Embeddings sauvegardés.")


# src/training/generate_embeddings.py

import pandas as pd
import networkx as nx
from karateclub import Node2Vec
from py2neo import Graph
from pathlib import Path

# --- Chargement des credentials ---
creds_path = Path(__file__).parent.parent / "config" / "credentials.txt"
creds = {}
with open(creds_path) as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            creds[k] = v

graph = Graph(creds["bolt_url"], auth=(creds["username"], creds["password"]))
print("Connecté à Neo4j")

# --- Extraction des nœuds avec gestion des valeurs manquantes ---
query_nodes = """
MATCH (n)
RETURN 
    id(n) AS node_id,
    head(labels(n)) AS main_label,
    coalesce(n.name, n.id, toString(id(n)), 'UNKNOWN') AS display_name
"""
nodes_df = graph.run(query_nodes).to_data_frame()
print(f"Nœuds récupérés : {len(nodes_df)}")

# --- Extraction des arêtes ---
query_edges = """
MATCH (a)-[r]->(b)
RETURN id(a) AS source, id(b) AS target
"""
edges_df = graph.run(query_edges).to_data_frame()
print(f"Arêtes récupérées : {len(edges_df)}")

# --- Construction du graphe NetworkX ---
G = nx.Graph()
for _, row in nodes_df.iterrows():
    G.add_node(row['node_id'], label=row['main_label'], name=row['display_name'])

for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'])

print(f"Graphe construit : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

# --- Génération des embeddings ---
model = Node2Vec(dimensions=64, walk_length=30, window_size=5, workers=4)
model.fit(G)
emb = model.get_embedding()

# --- Création du DataFrame final ---
node_ids = list(G.nodes())
emb_df = pd.DataFrame(emb, index=node_ids)

# Assure-toi que nodes_df a bien 'node_id' comme colonne
nodes_df = nodes_df.set_index('node_id')

# Jointure sécurisée
result_df = emb_df.join(nodes_df[['display_name', 'main_label']], how='left')
result_df = result_df.rename(columns={'display_name': 'name', 'main_label': 'label'})

# Remplir les valeurs manquantes au cas où
result_df['name'] = result_df['name'].fillna('UNKNOWN')
result_df['label'] = result_df['label'].fillna('Entity')

# Sauvegarde
Path("results/embeddings").mkdir(parents=True, exist_ok=True)
result_df.reset_index().rename(columns={'index': 'node_id'}).to_csv(
    "results/embeddings/node2vec.csv", index=False
)

print("Embeddings sauvegardés avec noms et labels.")