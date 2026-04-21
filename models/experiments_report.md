# NeuralZOO — Rapport d'expérimentation

## Contexte

Classification d'images d'animaux sur le dataset CIFAR-10 (6 classes : bird, cat,
deer, dog, frog, horse). 36 000 images au total, parfaitement équilibrées à 5 000
images par classe en entraînement et 1 000 en test. Résolution : 32×32 pixels RGB.

Deux architectures entraînées : un MLP comme baseline théorique, et un CNN custom
3 blocs comme modèle principal. Le MLP est abandonné rapidement (51% d'accuracy)
— il traite chaque pixel indépendamment et ne peut pas apprendre de structures
spatiales. Le CNN devient le seul candidat pour la production.

---

## Problème identifié dès le départ : cat et dog

À 32×32 pixels, chats et chiens partagent des silhouettes, des textures et des
proportions très similaires. Dès le premier run, la matrice de confusion révèle
une confusion systématique entre ces deux classes — 168 chiens prédits chats,
117 chats prédits chiens. Ce problème traverse toute l'expérimentation et guide
les choix techniques.

---

## Méthodologie

Chaque modification est testée isolément, dans un dossier d'expérience dédié
(`models/experiments/`), avec ses hyperparamètres tracés dans un `config.json`
et ses métriques dans un `metrics.json`. Aucun run n'écrase un autre.
La règle : un seul changement à la fois, on mesure l'impact, on décide.

---

## Progression des expériences

| Run | Accuracy | F1 macro | Levier ajouté |
|-----|----------|----------|---------------|
| baseline_75 | 75.4% | 0.752 | — |
| augmentation_v1 | 77.1% | 0.752 | Data augmentation (50 epochs) |
| augmentation_v2 | 80.3% | 0.768 | Data augmentation (100 epochs) |
| augmentation_v2_wd | 80.2% | 0.800 | + Weight decay (L2 = 1e-4) |
| combined_lr | 82.6% | 0.820 | + LR Scheduling |
| **combined_ls** | **83.0%** | **0.830** | + Label Smoothing (0.1) |

---

## Ce qui a fonctionné — et pourquoi

**Data augmentation (+5 points)** — le levier le plus puissant. En appliquant des
transformations aléatoires à chaque epoch (flip horizontal, crop, color jitter,
normalisation mean/std), le modèle voit des images "différentes" à chaque passage
et généralise mieux. Preuve par l'absurde : baseline_100 (sans augmentation,
100 epochs) donne 74.6% — pire que le baseline_75. Plus d'epochs sans augmentation
= plus d'overfitting, pas plus de généralisation.

**LR Scheduling (+2.3 points)** — ReduceLROnPlateau divise le learning rate par 2
dès que la val_loss stagne pendant 5 epochs. Le modèle fait de grands pas au début
pour converger vite, puis affine avec des micro-pas pour s'installer précisément
dans un minimum. Sur combined_lr, le LR est passé de 1e-3 à 1e-6 en 7 réductions
successives — chaque réduction a débloqué un plateau et permis une nouvelle
progression.

**Label Smoothing (+0.4 points, +3 points sur cat)** — au lieu d'apprendre "cette
image est chat à 100%", le modèle apprend "chat à 91.7%, 1.7% pour chaque autre
classe". Ça cible directement le problème cat/dog : le modèle n'est plus pénalisé
pour une légère hésitation entre deux classes visuellement proches. Résultat concret :
cat recall passe de 0.630 à 0.660, cat F1 de 0.70 à 0.73.

**Weight decay (impact sur F1, neutre sur accuracy)** — L2 = 1e-4 n'améliore pas
l'accuracy mais améliore la calibration du modèle. F1 macro passe de 0.768 à 0.800
— le modèle est plus précis quand il prédit, moins confiant quand il ne sait pas.

---

## Résultat final par classe

| Classe | Precision | Recall | F1 |
|--------|-----------|--------|----|
| bird | 0.85 | 0.81 | 0.83 |
| **cat** | **0.81** | **0.66** | **0.73** |
| deer | 0.83 | 0.88 | 0.85 |
| **dog** | **0.77** | **0.81** | **0.79** |
| frog | 0.84 | 0.93 | 0.88 |
| horse | 0.89 | 0.89 | 0.89 |

Cat reste la classe la plus difficile — recall à 0.66, soit 340 chats sur 1000
mal classifiés, majoritairement confondus avec dog. C'est une limite du dataset
à 32×32 pixels, pas de l'architecture : à cette résolution, les features
discriminantes entre chat et chien sont insuffisantes pour un CNN sans mécanisme
d'attention. Frog et horse sont les meilleures classes — leurs couleurs et
silhouettes sont distinctives même à faible résolution.

---

## Modèle retenu pour la production

`combined_ls` — 83.0% accuracy, F1 macro 0.830, exporté sous
`models/cnn_cifar10.pth`. Chargé par l'API FastAPI au démarrage, utilisé pour
l'inférence temps réel via `POST /predict`. Les probabilités softmax sont exposées
dans la réponse API pour visualisation dans le front Streamlit.

## Pistes d'amélioration non explorées

Pour dépasser 85%, les leviers suivants sont envisageables : architecture plus
profonde (4ème bloc conv, 256 filtres), CosineAnnealingLR, augmentation plus
agressive ciblant cat/dog (RandomRotation, RandomGrayscale), ou transfer learning
sur un ResNet-18 avec data augmentation et weight decay — ce qui résoudrait
l'overfitting massif observé sur le ResNet sans augmentation.