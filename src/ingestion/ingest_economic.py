# src/ingestion/ingest_economic.py
import pandas as pd
from py2neo import Graph
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ontology.defense_ontology import is_valid_relation, get_node_supertype

# Chargement des credentials
creds_path = Path("src/config/credentials.txt")
creds = {}
with open(creds_path) as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            creds[k] = v

graph = Graph(creds["bolt_url"], auth=(creds["username"], creds["password"]))
print("Connecté à Neo4j")

# Chargement des données
nodes_df = pd.read_csv("data/raw/economic_nodes.csv")
relations_df = pd.read_csv("data/raw/economic_relations.csv")

# Nettoyage
nodes_df.columns = nodes_df.columns.str.strip()
relations_df.columns = relations_df.columns.str.strip()

# Création des nœuds avec labels spécifiques
for _, row in nodes_df.iterrows():
    node_id = row['id']
    node_type = row['type']  # ex: "FinancialInstitution"
    supertype = get_node_supertype(node_type) or "Entity"
    props = {k: v for k, v in row.items() if pd.notna(v)}
    query = f"""
    MERGE (n:{node_type}:{supertype} {{id: $id}})
    SET n += $props
    """
    graph.run(query, id=node_id, props=props)

print(f"{len(nodes_df)} nœuds économiques créés.")

# Création des relations sémantiques
for _, row in relations_df.iterrows():
    source = row['source']
    target = row['target']
    rel_type = row['type'].replace(" ", "_").upper()  # ex: "FUNDS"
    
    # Validation ontologique (optionnel mais recommandé)
    # Ici on suppose que le CSV respecte l'ontologie
    
    query = f"""
    MATCH (a {{id: $source}})
    MATCH (b {{id: $target}})
    MERGE (a)-[:{rel_type}]->(b)
    """
    try:
        graph.run(query, source=source, target=target)
    except Exception as e:
        print(f"Erreur sur relation {source} -[{rel_type}]-> {target}: {e}")

print("Relations économiques créées.")
