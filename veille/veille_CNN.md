# Veille — Convolutional Neural Network (CNN)

---

## 1. Architecture typique d'un CNN

Un CNN est un réseau de neurones spécialisé dans le traitement de données spatiales (images, audio). Il extrait automatiquement des caractéristiques visuelles grâce à ses couches convolutives.

**Architecture standard :**
```
Input → [Conv → ReLU → Pooling] × N → Flatten → Dense → Softmax
```

| Couche | Rôle |
|--------|------|
| **Convolution** | Extraction de features (contours, textures, formes) |
| **Activation (ReLU)** | Introduit la non-linéarité |
| **Pooling** | Réduit la taille spatiale, conserve les features importantes |
| **Flatten** | Transforme la feature map 2D en vecteur 1D |
| **Dense (fully connected)** | Classification finale |
| **Softmax** | Produit les probabilités par classe |

**Hyperparamètres principaux :**
- Nombre de filtres par couche convolutive
- Taille des filtres (kernel size) : souvent 3×3 ou 5×5
- Stride (pas du filtre)
- Padding (same / valid)
- Type et taille de pooling
- Learning rate, batch size, epochs
- Taux de dropout

---

## 2. Couche convolutive & filtre de convolution

Une **couche convolutive** fait glisser un filtre (noyau) sur l'image d'entrée et calcule, à chaque position, le produit scalaire entre le filtre et la région de l'image correspondante.

Un **filtre de convolution** est une petite matrice de poids (ex. 3×3) apprise pendant l'entraînement. Chaque filtre détecte un pattern spécifique : contour vertical, horizontal, texture, etc.

> Pour une image 32×32 avec un filtre 3×3 et 32 filtres → on obtient 32 feature maps de taille ≈ 30×30.

---

## 3. Fonction d'activation dans un CNN

La fonction d'activation standard dans un CNN est **ReLU** (`max(0, x)`).

**Pourquoi ReLU ?**
- Simple et rapide à calculer
- Évite le vanishing gradient (gradient constant = 1 pour x > 0)
- Introduit la non-linéarité sans saturer les grandes valeurs positives
- Passe les activations inutiles (négatives) à zéro, ce qui rend le réseau sparse et efficace

---

## 4. Feature Map

Une **Feature Map** (carte de caractéristiques) est le résultat de l'application d'un filtre sur une image. Elle représente la présence et la localisation d'un pattern spécifique dans l'image.

- Pour chaque filtre → une feature map
- Après plusieurs couches → les feature maps représentent des concepts de plus en plus abstraits (contours → formes → objets)

---

## 5. Couche de Pooling

La **couche de Pooling** réduit les dimensions spatiales des feature maps pour diminuer la complexité computationnelle et rendre le modèle robuste aux petites translations.

| Type | Fonctionnement |
|------|----------------|
| **Max Pooling** | Prend la valeur maximale dans chaque région. Conserve les features les plus saillantes. |
| **Average Pooling** | Prend la moyenne dans chaque région. Lisse les features. |

> Exemple : un Max Pooling 2×2 avec stride 2 divise la taille spatiale par 2 (32×32 → 16×16).

---

## 6. Couche entièrement connectée (Fully Connected)

La dernière partie d'un CNN est une ou plusieurs couches **Dense (fully connected)**.

**Ce qu'elle reçoit :** le vecteur issu du Flatten — c'est-à-dire toutes les feature maps de la dernière couche de pooling aplaties en un seul vecteur 1D.

**Son rôle :** combiner toutes les features extraites par les couches convolutives pour produire la classification finale. En sortie, une activation **Softmax** transforme les scores en probabilités pour chaque classe.

---

## 7. CNN vs MLP pour la classification d'images

| Critère | MLP | CNN |
|---------|-----|-----|
| **Connexions** | Chaque pixel connecté à chaque neurone → explosion du nombre de paramètres | Partage des poids via les filtres → très peu de paramètres |
| **Invariance spatiale** | Aucune — sensible à la position des objets | Oui — détecte un pattern où qu'il soit dans l'image |
| **Extraction de features** | Manuelle ou absente | Automatique et hiérarchique |
| **Performance sur images** | Faible pour des images complexes | Excellente |
| **Scalabilité** | Très coûteux pour de grandes images | Efficace même sur des images haute résolution |

> **Conclusion :** Le CNN est préféré pour les images car il exploite la structure spatiale locale des pixels, partage ses poids entre positions, et apprend des représentations hiérarchiques (des contours simples aux objets complexes) avec bien moins de paramètres qu'un MLP.
