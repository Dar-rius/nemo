import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Chargement
df = pd.read_csv("results/embeddings/gnn_embeddings.csv")

# Nettoyage
df = df.dropna(subset=['name', 'label'])
print(f"Données après nettoyage : {len(df)} nœuds")

# Features : embeddings + type sémantique
X_emb = df.iloc[:, 3:-2].values  # colonnes 3 à -3 = embeddings 64D

# Encodage du type sémantique
le = LabelEncoder()
semantic_labels_array = np.array(le.fit_transform(df['label'])).reshape(-1, 1)
X_combined = np.hstack([X_emb, semantic_labels_array])

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Clustering
kmeans = KMeans(n_clusters=min(4, len(X_scaled)), random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Réduction dimensionnelle pour visualisation
perp = min(5, len(X_scaled) - 1)
tsne = TSNE(n_components=2, perplexity=min(5, len(X_scaled) - 1), random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Sauvegarde
df['cluster'] = clusters
df['tsne_1'] = X_tsne[:, 0]
df['tsne_2'] = X_tsne[:, 1]
df.to_csv("results/embeddings/gnn_clustered.csv", index=False)

# Visualisation
plt.figure(figsize=(12, 8))
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=df, palette='tab10', s=100)
for i, row in df.iterrows():
    plt.text(row['tsne_1']+0.5, row['tsne_2']+0.5, row['name'], fontsize=8)
plt.title("Clustering sémantique (Node2Vec + Type de nœud)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("results/embeddings/clustering_semantic_gnn.png", dpi=150)
plt.show()

print("Clustering sémantique sauvegardé.")
