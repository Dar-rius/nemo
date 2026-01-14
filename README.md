NEMO — Neuro-Symbolic Architecture for Strategic Decision Support

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-brightgreen?logo=neo4j)](https://neo4j.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Mémoire de Master 2 en Intelligence Artificielle**  
> _IA appliquée à la défense : Conception d’une Architecture Neuro-Symbolique pour l’aide à la décision stratégique par Graph Representation Learning_

---

Objectif

Concevoir un système d’**aide à la décision stratégique** capable de :

- Structurer des données multi-source (OSINT, économiques, géospatiales) dans un **graphe de connaissances sémantique**,
- Apprendre des **représentations neuronales interprétables** via Graph Representation Learning,
- Détecter des **points de non-retour** dans un scénario de crise simulé,
- Fournir des **recommandations traçables** fondées sur une ontologie formelle inspirée de la doctrine militaire (Clausewitz) et de la causalité (Pearl).

Ce projet explore la synergie entre **perception neuronale** et **raisonnement symbolique** pour répondre aux exigences critiques de la défense : **robustesse, explicabilité et anticipation**.

---

Composants 

Ontologie : classes (FinancialInstitution, NonStateActor, MilitaryBase), relations (FUNDS, LOCATED_AT), axiomes.
Graphe de connaissances : stocké dans Neo4j avec labels et types de relation natifs.
Embeddings : générés via Node2Vec, enrichis avec le type sémantique des nœuds.
Évaluation : K-Fold Cross-Validation, métriques hybrides, visualisation t-SNE.
Interface : dashboard Streamlit + Neo4j Browser pour la validation.


Structure du Projet

nemo/
├── data/raw/               # Données sources (CSV synthétiques)
├── src/
│   ├── ontology/           # Définition formelle de l'ontologie
│   ├── ingestion/          # Scripts d'ingestion Neo4j
│   ├── training/           # Génération d'embeddings & clustering
│   └── evaluation/         # Métriques hybrides & validation croisée
├── platform/               # Dashboard interactif (Streamlit)
├── results/embeddings/     # Résultats générés (non versionnés)
├── docs/                   # Documentation & mémoire
├── requirements.txt        # Dépendances Python
└── README.md

Installation & Exécution

Prérequis
- Python 3.10+
- Neo4j Desktop (ou Neo4j Aura)
- Accès local à http://localhost:7474

1. Cloner le dépôt
git clone https://github.com/Vicenzaa/nemo.git
cd nemo

2. Installer les dépendances
pip install -r requirements.txt

3. Configurer Neo4j
Crée ceci => src/config/credentials.txt :

bolt_url=bolt://localhost:7687
username=neo4j
password=passer1234

4. Lancer le pipeline
Ingestion
python src/ingestion/ingest_economic.py
python src/ingestion/ingest_oryx.py

Apprentissage
python src/training/generate_embeddings.py
python src/training/semantic_clustering.py

Évaluation
python src/evaluation/cross_validation.py

