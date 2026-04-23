# Veille — Multilayer Perceptron (MLP)

---

## 1. Architecture du Perceptron Multicouches

Un MLP (Multilayer Perceptron) est un réseau de neurones artificiels organisé en couches successives :

| Couche | Rôle |
|--------|------|
| **Couche d'entrée** | Reçoit les données brutes (une entrée par feature). Pas de calcul, simple passage des données. |
| **Couches cachées** | Effectuent les transformations via des neurones connectés. Plus il y en a, plus le réseau est "profond". |
| **Couche de sortie** | Produit la prédiction finale. Le nombre de neurones dépend de la tâche. |
| **Couches denses (fully connected)** | Chaque neurone est connecté à tous les neurones de la couche précédente. C'est le type de couche standard d'un MLP. |

**Hyperparamètres principaux :**
- Nombre de couches cachées
- Nombre de neurones par couche
- Fonction d'activation
- Learning rate
- Batch size
- Nombre d'epochs
- Optimiseur (Adam, SGD…)
- Taux de dropout (régularisation)

---

## 2. Choix de l'architecture selon la problématique

| Tâche | Couche de sortie | Fonction d'activation sortie | Loss function |
|-------|-----------------|-------------------------------|---------------|
| **Classification binaire** | 1 neurone | Sigmoid | Binary crossentropy |
| **Classification multiclasse** | N neurones (= nb classes) | Softmax | Categorical crossentropy |
| **Régression** | 1 neurone | Linéaire (aucune) | MSE / MAE |

---

## 3. Définitions clés

**Fonction d'activation**
Fonction mathématique appliquée à la sortie d'un neurone pour introduire de la non-linéarité. Sans elle, le réseau serait équivalent à une simple régression linéaire.

**Propagation (forward pass)**
Passage des données de la couche d'entrée vers la couche de sortie, couche par couche, en calculant à chaque étape : `sortie = activation(W·x + b)`.

**Rétropropagation (backpropagation)**
Algorithme qui calcule le gradient de la loss par rapport à chaque poids du réseau, en remontant de la sortie vers l'entrée via la règle de la chaîne. Permet de mettre à jour les poids.

**Loss function (fonction de perte)**
Mesure l'écart entre la prédiction du modèle et la vraie valeur. Le réseau cherche à la minimiser. Ex : MSE pour la régression, crossentropy pour la classification.

**Descente de gradient**
Algorithme d'optimisation qui ajuste les poids dans la direction opposée au gradient de la loss : `W = W - lr × ∂L/∂W`. L'objectif est de converger vers un minimum de la loss.

**Vanishing gradients**
Problème où les gradients deviennent très proches de zéro en remontant dans les couches profondes, ce qui bloque l'apprentissage des premières couches. Fréquent avec les fonctions sigmoid/tanh sur des réseaux profonds.

---

## 4. Fonctions d'activation — Exemples

| Fonction | Formule | Usage typique |
|----------|---------|---------------|
| **ReLU** | `max(0, x)` | Couches cachées (standard) |
| **Sigmoid** | `1 / (1 + e^-x)` | Sortie binaire |
| **Softmax** | `e^xi / Σe^xj` | Sortie multiclasse |
| **Tanh** | `(e^x - e^-x) / (e^x + e^-x)` | Couches cachées (centré sur 0) |
| **Leaky ReLU** | `max(0.01x, x)` | Alternative à ReLU (évite les neurones morts) |

---

## 5. Epochs, Iterations, Batch size

- **Batch size** : Nombre d'exemples traités avant une mise à jour des poids. Un batch de 32 signifie que les poids sont mis à jour après 32 images.
- **Iteration** : Une mise à jour des poids = un passage d'un batch. Pour 1000 images et un batch size de 100 → 10 iterations par epoch.
- **Epoch** : Un passage complet sur l'ensemble des données d'entraînement. 10 epochs = les données ont été vues 10 fois.

> **Relation :** `Iterations par epoch = Nb d'exemples / Batch size`

---

## 6. Learning rate

Le learning rate (lr) contrôle la taille du pas lors de la mise à jour des poids.

| Cas | Conséquence |
|-----|-------------|
| **Trop bas** | Convergence lente, risque de rester coincé dans un minimum local |
| **Trop élevé** | Divergence, la loss oscille ou explose sans converger |
| **Optimal** | Convergence rapide et stable vers un bon minimum |

---

## 7. Batch Normalization

La **Batch Normalization** normalise les activations d'une couche (moyenne ≈ 0, variance ≈ 1) pour chaque mini-batch pendant l'entraînement.

**Pourquoi l'utiliser ?**
- Stabilise et accélère l'entraînement
- Réduit la sensibilité au learning rate
- Agit comme une légère régularisation (réduit le besoin de dropout)
- Atténue le problème du vanishing gradient

---

## 8. Algorithme Adam

**Adam** (Adaptive Moment Estimation) est un optimiseur qui combine deux approches :
- **Momentum** : utilise la moyenne des gradients passés pour donner de l'élan
- **RMSProp** : adapte le learning rate pour chaque poids selon la magnitude des gradients récents

Il est le plus utilisé en pratique car il est rapide à converger et nécessite peu de réglages.

---

## 9. Résumé — Qu'est-ce qu'un MLP ?

Un **Perceptron Multicouches** est un réseau de neurones artificiels constitué de plusieurs couches de neurones entièrement connectés. Il apprend à partir de données labellisées en ajustant ses poids par rétropropagation et descente de gradient, afin de minimiser l'écart entre ses prédictions et les vraies valeurs. C'est le modèle de base du Deep Learning, adapté à la classification et à la régression sur des données tabulaires ou vectorielles.
