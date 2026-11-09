# Fiche de lecture — Musgrave, Belongie & Lim (2020)

*A Metric Learning Reality Check*, ECCV 2020, arXiv:2003.08505 (v3, 16 sept. 2020).
PDF : `musgrave2020_metric_learning_reality_check.pdf` (19 pages, sans le
supplément, qui est sur github.com/KevinMusgrave/powerful-benchmarker).

Tout ce qui suit a été relu dans le PDF. Les numéros de section renvoient à l'article.

---

## Thèse

Les articles d'apprentissage métrique de 2016 à 2019 annoncent des gains énormes
sur contrastive (2006) et triplet (2006). Une fois les comparaisons rendues
équitables, **les pertes récentes font à peine mieux, parfois pareil** (§4.2,
fig. 2b : la courbe est presque plate).

## Les trois défauts dénoncés (§2)

1. **Comparaisons inéquitables** (§2.1)
   - architecture différente d'un article à l'autre (GoogleNet, BN-Inception,
     ResNet50). Avant tout apprentissage, ResNet50 a déjà 7 points de plus que
     GoogleNet sur CUB200 (tableau 1 : 41,1 / 51,1 / 48,7 en Recall@1, après ACP
     vers 512 dimensions) ;
   - dimension des embeddings qui change ;
   - augmentation d'images plus riche dans le code que dans l'article ;
   - optimiseur et taux d'apprentissage différents ;
   - détails cachés : un article de 2019 gèle les BatchNorm, le code dit que ça
     vaut 2 points, l'article n'en parle pas ;
   - pas d'intervalles de confiance.
2. **Métriques faibles** (§2.2)
   - Recall@1 donne ~100 % à des espaces très différents (fig. 1), et 100 % aux
     quatre cas du tableau 3 ;
   - NMI et F1 demandent un clustering (k-means), qui ajoute du bruit ;
   - NMI est haute avec beaucoup de classes, même sur un réseau **aléatoire**
     (tableau 2 : 81,2 sur SOP contre 23,6 sur CUB200, GoogleNet).
3. **Retour du test** (§2.3) : pas de validation, on surveille le test pendant
   l'entraînement et on rapporte le meilleur score.

## Son protocole (§3)

**Réglages communs (§3.1)**
- BN-Inception pré-entraîné ImageNet, embedding 128 dimensions.
- **Tout le réseau est entraîné (fine-tuning)**, BatchNorm gelés.
- Lot de 32 : 8 classes × 4 images pour les pertes d'embedding,
  32 classes × 1 image pour les pertes de classification.
- Augmentation : côté court à 256, crop aléatoire de taille 40 à 256, ratio
  3/4 à 4/3, redimensionné 227×227, miroir à 50 %. En évaluation : 256 puis crop
  centré 227.
- RMSprop, taux 1e-6. Pour les pertes à poids propres (ArcFace…), le taux de ces
  poids est un hyperparamètre.
- Embeddings normalisés L2 pour la perte et l'évaluation.

**Métriques (§3.2)**
- **MAP@R** (principale), R-Precision, P@1 (pour comparaison).
- MAP@R est moins bruitée : autocorrélation d'ordre 1 de 0,81 contre 0,73 pour
  Recall@1 pendant l'entraînement.

**Réglage des hyperparamètres (§3.3)**
- 50 itérations d'optimisation bayésienne par perte.
- Chaque itération = validation croisée à 4 plis sur la **première moitié des
  classes**. Plis déterministes : classes 0–12,5 %, 12,5–25 %, etc. Entraînement
  sur 3 plis, validation sur le 4e, les 4 combinaisons. Train et validation
  toujours disjoints en classes.
- Arrêt quand la précision de validation plafonne.
- **Seconde moitié des classes = test.**
- Avec la meilleure configuration, on garde le meilleur checkpoint de chacun des
  4 plis et on évalue sur le test de deux façons :
  - *concaténé* : les 4 embeddings de 128 → un vecteur de 512, renormalisé ;
  - *séparé* : chaque modèle à 128 évalué seul, moyenne des 4.
- **10 entraînements** avec la meilleure configuration, moyenne et IC à 95 %.

**Référence « Pretrained »** : le réseau pré-entraîné sans aucun apprentissage
métrique, embeddings réduits à 128 ou 512 par ACP (tableaux 4 à 6).

## Expériences (§4)

- 13 pertes + 1 combinaison perte + mineur (tableau 7), dont contrastive,
  triplet, N. Softmax, CosFace et ArcFace.
- CUB200, Cars196, SOP. **Aucun visage** : ArcFace et CosFace sont entraînées
  sur oiseaux, voitures et produits, sans aucun modèle entraîné sur visages.
- Supplément : lot de 256 au lieu de 32 sur CUB200, résultats à peu près
  identiques sauf FastAP qui remonte.

**Chiffres utiles (MAP@R, concaténé 512)**

| | CUB200 | Cars196 | SOP |
|---|---|---|---|
| Pretrained | 14,21 | 5,91 | 23,44 |
| Contrastive | 26,53 | 24,89 | 44,39 |
| Triplet | 23,69 | 23,02 | 43,37 |
| N. Softmax | 25,25 | 26,00 | 47,13 |
| ArcFace | 26,45 | 27,22 | 47,41 |
| Meilleur | CosFace 26,70 | MS 28,07 | ArcFace 47,41 |

**D'où venaient les faux gains (§4.2)** : l'article lifted structure (2016)
n'échantillonnait que N/2 paires et N/3 triplets par lot, et fixait la marge du
triplet à 1 alors que l'optimum est vers 0,1. Les articles suivants ont recopié
ces chiffres bas.

## Conclusion (§5)

Pistes ouvertes par l'auteur : le lien entre hyperparamètres optimaux et
couple données/architecture, et **pourquoi des pertes différentes donnent des
résultats proches**. C'est exactement l'angle de notre question sur la géométrie.

## Ce que l'article ne donne pas

- Les plages de recherche des hyperparamètres : dans le supplément / le code.
- Aucune mesure de vérification (ROC, TAR à FAR fixé) : il ne fait que de la
  recherche par plus proches voisins.
- Aucune analyse de géométrie de l'espace (alignement, uniformité, spectre).
