# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from pathlib import Path

# df = pd.read_csv("results/embeddings/node2vec_clustered.csv")
# df = df.dropna(subset=['cluster'])

# # Features et target
# embedding_cols = [str(i) for i in range(64)]
# X = df[embedding_cols].values
# y = df['cluster'].astype(int).to_numpy() # clusters comme labels

# # Validation croisée stratisée
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# rf = RandomForestClassifier(n_estimators=100, random_state=42)

# scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')

# print("K-Fold Cross-Validation (5 plis) ===")
# print(f"Accuracy moyenne : {scores.mean():.4f} ± {scores.std():.4f}")
# print(f"Scores par pli : {scores}")

# # Sauvegarde
# result = pd.DataFrame({
#     "fold": range(1, 6),
#     "accuracy": scores
# })
# result.to_csv("results/embeddings/cv_results.csv", index=False)


# src/evaluation/cross_validation.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Charger les données
df = pd.read_csv("results/embeddings/node2vec_clustered.csv")

# Vérifier qu'on a bien les colonnes 0 à 63
embedding_cols = [str(i) for i in range(64)]
missing = [col for col in embedding_cols if col not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes : {missing}")

# Extraire features et target
X = df[embedding_cols].values.astype(np.float32)
y = df['cluster'].astype(int).to_numpy()

print(f"Données prêtes : X={X.shape}, y={y.shape}, classes={np.unique(y)}")

# Ajuster le nombre de splits si trop de classes rares
n_splits = min(5, len(np.unique(y)), sum(np.bincount(y) >= 2))
print(f"Utilisation de {n_splits} folds pour la CV")

# Validation croisée
rf = RandomForestClassifier(n_estimators=100, random_state=42)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")

print(f"CV Accuracy : {scores.mean():.4f} ± {scores.std():.4f}")
pd.DataFrame({"fold": range(1, n_splits+1), "acc": scores}).to_csv("results/embeddings/cv.csv", index=False)