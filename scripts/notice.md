# Notice — scripts/tune.py

## Lancer un run

```bash
just tune  [options]
```

## Options

| Option | Défaut | Description |
|--------|--------|-------------|
| `--epochs` | 50 | Nombre max d'epochs |
| `--lr` | 1e-3 | Learning rate initial |
| `--patience` | 10 | Patience early stopping |
| `--weight-decay` | 0.0 | Régularisation L2 |
| `--augment` | False | Active la data augmentation |

## Exemples

```bash
# Baseline
just tune baseline --epochs 50

# Avec augmentation
just tune aug_v1 --augment --epochs 100

# Configuration complète
just tune combined --augment --epochs 150 --patience 20 --weight-decay 1e-4
```

## Ce que ça produit
models/experiments/<nom>/
├── config.json     ← hyperparamètres du run
├── weights.pth     ← meilleurs poids
└── metrics.json    ← accuracy, F1, matrice de confusion, courbes

## Promouvoir en production

```bash
cp models/experiments/<nom>/weights.pth models/cnn_cifar10.pth