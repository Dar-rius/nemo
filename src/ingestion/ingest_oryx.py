import pandas as pd
from py2neo import Graph
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

creds_path = Path("src/config/credentials.txt")
creds = {}
with open(creds_path) as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            creds[k] = v

graph = Graph(creds["bolt_url"], auth=(creds["username"], creds["password"]))
print("Connecté à Neo4j")

df = pd.read_csv("data/raw/oryx.csv")
df.columns = df.columns.str.strip()

for _, row in df.iterrows():
    # Nœuds
    actor = {"name": row["actor"], "type": "StateActor"}
    opponent = {"name": row["opponent"], "type": "NonStateActor"}
    location = {"name": row["location"], "type": "Location"}
    equipment = {"name": row["equipment_name"], "type": row["equipment_type"]}
    event = {"id": row["event_id"], "date": row["date"], "type": "ConflictEvent"}

    # Création avec labels
    graph.run("MERGE (a:StateActor:Actor {name: $name})", name=actor["name"])
    graph.run("MERGE (o:NonStateActor:Actor {name: $name})", name=opponent["name"])
    graph.run("MERGE (l:Location {name: $name})", name=location["name"])
    graph.run("MERGE (e:Equipment {name: $name, type: $type})", name=equipment["name"], type=equipment["type"])
    graph.run("MERGE (ev:ConflictEvent:Event {id: $id, date: $date})", id=event["id"], date=event["date"])

    # Relations sémantiques
    graph.run("""
        MATCH (a:Actor {name: $actor}), (ev:Event {id: $event_id})
        MERGE (a)-[:Initiates]->(ev)
    """, actor=row["actor"], event_id=row["event_id"])

    graph.run("""
        MATCH (o:Actor {name: $opponent}), (ev:Event {id: $event_id})
        MERGE (o)-[:Opposes]->(ev)
    """, opponent=row["opponent"], event_id=row["event_id"])

    graph.run("""
        MATCH (ev:Event {id: $event_id}), (l:Location {name: $location})
        MERGE (ev)-[:LocatedAt]->(l)
    """, event_id=row["event_id"], location=row["location"])

    graph.run("""
        MATCH (ev:Event {id: $event_id}), (eq:Equipment {name: $equip})
        MERGE (ev)-[:Uses]->(eq)
    """, event_id=row["event_id"], equip=row["equipment_name"])

print("Données Oryx ingérées avec types sémantiques.")