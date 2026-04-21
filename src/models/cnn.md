## Veille scientifique — Deep Learning appliqué à la classification d'images

### Multilayer Perceptron (MLP)

#### Architecture

Un MLP est un réseau de neurones entièrement connecté organisé en couches successives :

- **Couche d'entrée** : reçoit les données brutes. Pour une image 32×32 RGB, c'est un vecteur de 3072 valeurs (un neurone par pixel).
- **Couches cachées** : chaque neurone est connecté à tous les neurones de la couche précédente. C'est là que le réseau apprend des représentations intermédiaires.
- **Couche de sortie** : produit autant de valeurs que de classes (6 pour notre problème).

#### Hyperparamètres

- `hidden_sizes` : nombre et taille des couches cachées
- `learning_rate` : vitesse d'apprentissage
- `batch_size` : nombre d'exemples traités par itération
- `epochs` : nombre de passes complètes sur le dataset
- `dropout` : taux de régularisation

#### Définitions fondamentales

**Fonction d'activation**
Fonction non-linéaire appliquée à la sortie de chaque neurone. Sans elle, empiler des couches linéaires revient à n'en avoir qu'une seule. Exemples :
- **ReLU** (Rectified Linear Unit) : `f(x) = max(0, x)`. Met à zéro les valeurs négatives, laisse les positives intactes. Standard pour les couches cachées — simple, efficace, ne souffre pas du vanishing gradient.
- **Softmax** : transforme un vecteur de scores bruts (logits) en probabilités qui somment à 1. Utilisée en sortie pour la classification multi-classes.
- **Sigmoid** : écrase les valeurs entre 0 et 1. Historiquement utilisée dans les couches cachées, remplacée par ReLU en pratique.

**Propagation (forward pass)**
Passage des données de l'entrée vers la sortie. Chaque couche applique sa transformation : `sortie = activation(W × entrée + b)` où W sont les poids et b le biais.

**Rétropropagation (backward pass)**
Algorithme qui calcule le gradient de la loss par rapport à chaque paramètre du réseau, en remontant de la sortie vers l'entrée via la règle de dérivation en chaîne. C'est ce gradient qui indique dans quelle direction ajuster chaque poids.

**Loss function**
Mesure l'écart entre la prédiction du modèle et la vraie étiquette. Pour la classification multi-classes, on utilise la **Cross-Entropy Loss** :
`L = -Σ y_i × log(ŷ_i)`
Plus la loss est faible, meilleure est la prédiction.

**Descente de gradient**
Algorithme d'optimisation qui ajuste les paramètres dans la direction opposée au gradient de la loss :
`w = w - lr × ∂L/∂w`
À chaque itération, le modèle fait un petit pas vers un minimum de la loss.

**Vanishing gradient**
Problème qui survient dans les réseaux profonds : en remontant les gradients couche par couche via la rétropropagation, ils deviennent exponentiellement petits. Les premières couches n'apprennent presque plus. ReLU et les architectures modernes (ResNet) ont été conçues pour résoudre ce problème.

**Epochs, Iterations, Batch size**
- **Batch size** : nombre d'exemples traités simultanément avant une mise à jour des poids. Un batch de 64 signifie qu'on calcule la loss sur 64 images, puis on rétropropage.
- **Iteration** : une mise à jour des poids = un batch traité.
- **Epoch** : une passe complète sur le dataset entier. Avec 27 000 images et un batch de 64 : 27000/64 ≈ 422 iterations par epoch.

**Learning rate**
Hyperparamètre qui contrôle l'amplitude de chaque mise à jour des poids.
- Trop élevé : le modèle diverge, la loss explose ou oscille sans converger.
- Trop faible : convergence très lente, risque de rester bloqué dans un minimum local.
- En pratique : on commence autour de 1e-3 avec Adam et on ajuste selon les courbes de loss.

**Batch Normalization**
Couche qui normalise les activations d'un mini-batch pour stabiliser l'entraînement. Pour chaque feature, elle soustrait la moyenne et divise par l'écart-type du batch (standardisation), puis réapprend une échelle optimale via deux paramètres γ et β entraînables :
`y = (x - μ) / √(σ² + ε) × γ + β`
Effets : entraînement plus stable, convergence plus rapide, meilleure généralisation. Placée systématiquement après la couche linéaire/convolutive et avant la fonction d'activation.

**Algorithme Adam**
Optimiseur qui adapte le learning rate pour chaque paramètre individuellement, en combinant deux mécanismes :
- **Momentum** : accumule les gradients passés pour lisser la direction de descente (évite les oscillations).
- **RMSProp** : divise par la racine carrée des gradients passés au carré (ralentit sur les directions où le gradient est fort, accélère sur les directions où il est faible).
Résultat : Adam converge plus vite que la descente de gradient classique et est moins sensible au choix du learning rate. C'est l'optimiseur par défaut pour la majorité des projets de Deep Learning.

---

### Convolutional Neural Network (CNN)

#### Architecture typique

Un CNN est organisé en deux parties distinctes :

```
Image brute
    ↓
[Blocs convolutifs] × N    ← extraction de features
    ↓
[Couche entièrement connectée]  ← classification
    ↓
Prédiction (logits → softmax → probabilités)
```

#### Hyperparamètres

- Nombre de blocs convolutifs
- Nombre de filtres par bloc
- `kernel_size` : taille des filtres (3×3 en pratique)
- `padding` : bordure de zéros pour préserver la taille spatiale
- `pool_size` : taille de la fenêtre de pooling
- `dropout` : taux de régularisation

#### Couche convolutive

Un filtre (kernel) est une petite matrice de valeurs apprises par rétropropagation, typiquement 3×3. Il se déplace sur l'image et à chaque position effectue un **produit scalaire** entre ses valeurs et le patch d'image correspondant, puis ajoute un biais :

`valeur = Σ (filtre × patch) + biais`

Ce calcul répété à toutes les positions produit une **Feature Map** — une carte qui indique où et à quel degré le pattern du filtre est présent dans l'image. Un bloc convolutif crée N filtres en parallèle, produisant N feature maps simultanément.

Les valeurs des filtres sont initialisées aléatoirement puis optimisées par rétropropagation. Chaque filtre converge vers un pattern différent car l'initialisation aléatoire les place dans des zones distinctes de l'espace des paramètres, et le réseau n'a aucun intérêt à maintenir deux filtres identiques (redondance = perte inutile de capacité).

#### Fonction d'activation — ReLU

ReLU est la fonction d'activation standard des CNN. Elle est appliquée après chaque convolution :
`f(x) = max(0, x)`

Elle est préférée pour trois raisons :
1. Elle ne sature pas pour les valeurs positives — le gradient ne disparaît pas.
2. Elle est calculatoirement triviale.
3. Elle introduit la non-linéarité nécessaire pour apprendre des représentations complexes.

#### Feature Map

Résultat de l'application d'un filtre sur l'entrée. Une feature map encode "où ce pattern est présent dans l'image". Les couches profondes combinent les feature maps des couches précédentes pour détecter des structures de plus en plus abstraites :

```
Couche 1 : contours, coins, textures simples
Couche 2 : courbes, angles, motifs intermédiaires
Couche 3 : formes complexes (oreilles, pattes, silhouettes)
```

#### Couche de Pooling

Réduit la dimension spatiale des feature maps en appliquant une opération locale sur des fenêtres non chevauchantes.

- **Max Pooling** : conserve la valeur maximale de chaque fenêtre. Sélectionne les endroits où le filtre a le mieux "matché" l'image. Rend le réseau invariant aux petites translations — si le pattern se déplace d'un pixel, la même valeur max est conservée.
- **Average Pooling** : conserve la moyenne de chaque fenêtre. Lisse davantage l'information, moins courant en pratique pour les CNN de classification.

`MaxPool2d(2)` divise chaque dimension par 2 : 32×32 → 16×16.

#### Couche entièrement connectée

Après les blocs convolutifs, les feature maps sont aplaties (`Flatten`) en un vecteur 1D. Ce vecteur ne contient plus des pixels mais des scores de présence de features apprises. Il entre dans un MLP classique qui combine ces scores pour produire la décision finale.

Dans notre CNN : 128 feature maps 4×4 → vecteur de 2048 → couche 256 → 6 logits (un par classe).

#### Pourquoi le CNN est supérieur au MLP pour les images

Le MLP traite chaque pixel indépendamment — il ne peut pas apprendre que deux pixels voisins forment un contour. Il détruit l'information spatiale dès l'entrée via le Flatten.

Le CNN exploite la structure spatiale : les filtres regardent des patches locaux, le pooling apporte l'invariance à la translation, et la hiérarchie de blocs construit des représentations de plus en plus abstraites. Sur CIFAR-10 animaux, cela se traduit concrètement par +24 points d'accuracy (75% CNN vs 51% MLP).

---

### Application — Résultats obtenus

| Modèle     | Accuracy test | F1 macro | Observations |
|------------|--------------|----------|--------------|
| MLP        | ~51%         | ~0.51    | Overfitting sévère dès l'epoch 3, plafond rapide |
| CNN custom | 75%          | 0.75     | Entraînement stable, overfitting contrôlé |
| ResNet-18  | 77%          | 0.77     | Overfitting massif (train→97%, val→79%) |

Le CNN custom est retenu pour le déploiement. ResNet-18 souffre d'un ratio paramètres/données défavorable (11M paramètres pour 27 000 images) sans data augmentation. Le CNN custom offre le meilleur équilibre performance/généralisation sur ce dataset.