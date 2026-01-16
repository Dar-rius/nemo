import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

# --- CONFIGURATION ---
NODES_PATH = "data/raw/economic_nodes.csv"
RELATIONS_PATH = "data/raw/economic_relations.csv"
OUTPUT_PATH = "results/embeddings/gnn_embeddings.csv"

# 1. Chargement
nodes_df = pd.read_csv(NODES_PATH)
rels_df = pd.read_csv(RELATIONS_PATH)

# --- PRÉPARATION DES DONNÉES ---

# A. Encodage des nœuds (Mapping ID -> Index)
all_ids = nodes_df['id'].unique()
node_encoder = LabelEncoder()
node_encoder.fit(all_ids)
nodes_df['id_idx'] = node_encoder.transform(nodes_df['id'])
NUM_NODES = len(all_ids)

# B. Encodage des relations (CORRECTIF fit_transform)
rel_encoder = LabelEncoder()
rels_df['type_idx'] = rel_encoder.fit_transform(rels_df['type'])
NUM_RELATIONS = len(rel_encoder.classes_)

# C. Encodage des labels pour l'entraînement (Target)
target_encoder = LabelEncoder()
nodes_df['label_idx'] = target_encoder.fit_transform(nodes_df['type'])
NUM_CLASSES = len(target_encoder.classes_)

# D. Préparation des Features (X)
# On transforme le montant en log pour éviter que BlackRock n'écrase tout statistiquement
nodes_df['montant_clean'] = pd.to_numeric(nodes_df['montant'], errors='coerce').fillna(0)
nodes_df['montant_log'] = np.log1p(nodes_df['montant_clean'])

# Encodage du pays
country_encoder = LabelEncoder()
nodes_df['pays_idx'] = country_encoder.fit_transform(nodes_df['pays'].fillna('Unknown'))

# On crée le tenseur de caractéristiques [Montant_Log, Pays_Idx]
X = torch.tensor(nodes_df[['montant_log', 'pays_idx']].values, dtype=torch.float)

# E. Construction du Graphe PyTorch
# On ne garde que les relations dont les IDs existent dans notre liste de nœuds
mask_rels = rels_df['source'].isin(all_ids) & rels_df['target'].isin(all_ids)
valid_rels = rels_df[mask_rels].copy()

edge_index = torch.tensor([
    node_encoder.transform(valid_rels['source']),
    node_encoder.transform(valid_rels['target'])
], dtype=torch.long)

edge_type = torch.tensor(valid_rels['type_idx'].values, dtype=torch.long)
y = torch.tensor(nodes_df['label_idx'].values, dtype=torch.long)

# --- MODÈLE RGCN ---

class NemoRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, num_classes):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        embeddings = self.conv2(x, edge_index, edge_type)
        out = self.classifier(embeddings)
        return out, embeddings

# Initialisation
model = NemoRGCN(X.shape[1], 64, NUM_RELATIONS, NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- ENTRAÎNEMENT ---
print(f"Entraînement sur {NUM_NODES} nœuds et {len(valid_rels)} relations...")
model.train()
for epoch in range(101):
    optimizer.zero_grad()
    out, _ = model(X, edge_index, edge_type)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# --- EXTRACTION ---
model.eval()
with torch.no_grad():
    _, final_embeddings = model(X, edge_index, edge_type)

# Sauvegarde compatible avec ton script de clustering
emb_df = pd.DataFrame(final_embeddings.numpy())
emb_df.columns = [str(i) for i in range(emb_df.shape[1])]
emb_df['name'] = nodes_df['id']
emb_df['label'] = nodes_df['type']

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
emb_df.to_csv(OUTPUT_PATH, index=False)
print(f"Fini ! Embeddings sauvegardés dans {OUTPUT_PATH}")
