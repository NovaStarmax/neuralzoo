# NeuralZOO

Classification d'images d'animaux avec CNN — EPITECH MSc Big Data

## Stack

- **Python 3.13** — langage principal
- **uv** — package manager
- **FastAPI** — API REST
- **Streamlit** — interface web
- **PyTorch / TorchVision** — modèles CNN
- **Just** — task runner

## Démarrage rapide

### Prérequis

- [uv](https://docs.astral.sh/uv/) installé
- [just](https://github.com/casey/just) installé
- Docker + Docker Compose (pour le mode conteneurisé)

### Installation des dépendances

```bash
# Tout installer (api + front + notebooks)
just sync-all

# Ou par composant
just sync-api
just sync-front
just sync-notebooks
```

### Variables d'environnement

```bash
cp .env.example .env
```

### Lancement avec Docker Compose

```bash
docker compose up --build
```

- API : http://localhost:8000
- Front : http://localhost:8501
- Docs API : http://localhost:8000/docs

## Structure

```
neuralzoo/
├── api/          # FastAPI — prédiction, métriques, health
│   ├── routers/  # Endpoints par domaine
│   ├── ml/       # Chargement du modèle et inférence
│   └── data/     # Chargement des données
├── front/        # Streamlit — interface utilisateur
│   └── pages/    # Pages supplémentaires
├── notebooks/    # Exploration et modélisation
├── models/       # Poids des modèles (ignorés par git)
└── veille/       # Ressources de veille technologique
```
