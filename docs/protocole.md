# Protocole expérimental — SRD sujet 09

Document de travail du groupe. Il fixe **comment** l'expérience est menée.
Toute modification se discute en groupe et se date ici, en bas, dans le journal.

Les points marqués **[à valider]** sont des précisions techniques ajoutées à la
rédaction, pas encore discutées en groupe.

Référence principale : Musgrave, Belongie & Lim (2020), *A Metric Learning
Reality Check* — fiche de lecture dans `references/musgrave2020_fiche.md`.

---

## 1. Objectif et question de recherche

**Question.** Qu'est-ce que la fonction de coût change à la façon dont les
visages sont rangés dans l'espace des vecteurs, et est-ce que ça aide à
reconnaître des personnes **jamais vues** à l'entraînement ?

**Principe.** Une comparaison contrôlée : un réseau pré-entraîné **gelé**, une
même tête linéaire, mêmes données, même évaluation. **Seule la fonction de coût
change.**

**Périmètre.** La chaîne complète d'un système est détection → alignement →
description → comparaison. On ne traite que la **description** (image → vecteur),
sur les recadrages alignés fournis par CelebA. Détection et alignement sont hors
périmètre.

**Ce que ce n'est pas.** Pas une recherche originale ni une réplication complète
de Musgrave : on reprend l'essentiel de sa méthode, à l'échelle d'un projet de M1.

### Ordre d'exécution

| Étape | Contenu | Section |
|---|---|---|
| 0 | Dépôt, environnement, partition, test de non-recouvrement | 3 |
| 1 | Extraction des vecteurs (une fois, réutilisée par tous) | 4 |
| 2 | Mesures préalables : brute et LDA → choix du backbone et de la résolution | 5 |
| 3 | Réglage des 4 pertes sur la validation | 8 |
| 4 | Entraînements finaux : meilleure configuration × 5 graines | 9 |
| 5 | Évaluation sur le test, **une seule fois** | 10 |
| en parallèle | Méthodes classiques | 11 |

Aucun entraînement de perte avant d'avoir les résultats de l'étape 2.

---

## 2. Ce qu'on reprend de Musgrave, et ce qu'on laisse

### Repris

| Élément | Pourquoi |
|---|---|
| Personnes du test jamais vues à l'entraînement ni en validation | Sans ça, le résultat ne vaut rien |
| Test consulté une seule fois, à la fin | Éviter de se régler sur le test |
| Validation avec des personnes distinctes de l'entraînement | Choisir les réglages honnêtement |
| Mêmes conditions pour toutes les pertes | Comparaison équitable |
| Lots adaptés au type de perte (8 × 4 ou 32 × 1) | Sinon contrastive et triplet n'apprennent rien |
| Même nombre d'essais de réglage par perte | Ne favoriser aucune perte |
| Plusieurs graines et intervalles de confiance | Distinguer un vrai écart du hasard |
| Métriques MAP@R et P@1 | Ses métriques, quasi gratuites |
| Ligne de référence sans entraînement | Savoir ce que les pertes apportent |

### Laissé de côté

| Chez Musgrave | Chez nous | Raison |
|---|---|---|
| Fine-tuning du réseau entier + augmentation | Réseau gelé, vecteurs pré-calculés | Simplicité, coût |
| 50 essais bayésiens par perte | 10 essais tirés au hasard | Suffisant à notre échelle |
| Validation croisée à 4 plis | Un seul découpage entraînement / validation | 4 fois moins de calcul |
| 10 relances | 5 graines | Compromis coût / précision |
| 13 pertes | 4 pertes | On garde celles qui comptent |

### Ajouté (absent chez Musgrave)

Vérification (ROC, TAR à FAR fixé), histogrammes de similarité, alignement et
uniformité, planche des plus proches voisins, méthodes classiques. Musgrave
n'utilise aucun visage et ne mesure que la recherche par plus proches voisins.

---

## 3. Données

### CelebA

- ~200 000 images, 10 177 identités, ~20 images par personne en moyenne.
- Images alignées fournies : 178 × 218 pixels, yeux toujours au même endroit,
  mais cheveux, cou et fond visibles.
- Découpage officiel fourni (`list_eval_partition.txt`) :
  ~160 000 / ~20 000 / ~20 000 images.

### Découpage

- On utilise le **découpage officiel** de CelebA.
- Un **test automatique** vérifie qu'aucune identité n'apparaît dans deux parties.
  Ce test ne doit jamais être désactivé.
- **Repli** : si le test trouve un recouvrement, on passe à un découpage par
  identité à la Musgrave (50 % des identités pour entraînement + validation,
  50 % pour le test).

### Filtre

- On garde uniquement les identités ayant **au moins 4 images**, dans **toutes**
  les parties. Raison : les lots 8 × 4 l'exigent, et les 4 pertes doivent voir
  exactement les mêmes données.
- Le nombre d'identités et d'images perdues est consigné et rapporté.

### Résultat (étape 0f, `splits/summary.json`)

Le découpage officiel est bien **disjoint en identités** (0 identité commune
entre parties) : pas besoin du repli.

| Partie | Avant filtre | Après filtre (≥ 4 photos) |
|---|---|---|
| train | 8 192 identités, 162 770 images | **7 689 identités, 161 602 images** |
| val | 985 identités, 19 867 images | **924 identités, 19 720 images** |
| test | 1 000 identités, 19 962 images | **951 identités, 19 850 images** |

### Recadrage

- Recadrage **carré centré sur le visage**, même règle pour toutes les images.
- Objectif : retirer une bonne partie des cheveux et du fond, pour limiter le
  risque que le réseau reconnaisse un style plutôt qu'un visage.
- La taille du carré est fixée **à l'œil**, après inspection d'une dizaine
  d'images, puis consignée ici.
- **Choix (étape 1a) : carré de 128 px**, boîte (24, 52, 152, 180) dans
  l'image 178 × 218.
  - Placement : centre horizontal 88,5 px et yeux à 111,2 px, positions
    moyennes mesurées sur les repères de CelebA (écart-type 1 à 2 px) ; les yeux
    sont placés à 46 % de la hauteur du carré, comme dans le gabarit 112 × 112
    d'ArcFace.
  - Candidats comparés sur 10 identités de validation : 178, 150, 128, 100 px.
    178 et 150 gardent trop de coiffure, de fond et de vêtements ; 100 coupe le
    menton sur plusieurs images. 128 garde le visage du front au menton.
  - Il reste un peu de cheveux sur les côtés : la planche des plus proches
    voisins (section 5) contrôlera si le réseau s'en sert.

### Règles

- Jeux de données publics uniquement. Aucune photo produite par nous ou nos
  proches (données biométriques, RGPD).
- Les fichiers de partition (listes d'identités par partie) sont versionnés dans
  le dépôt.

---

## 4. Backbones et extraction

### Les deux réseaux mesurés

| Réseau | Source | Entraînement |
|---|---|---|
| ResNet-50 supervisé | torchvision, poids **IMAGENET1K_V1** | Classification ImageNet |
| ResNet-50 Barlow Twins | checkpoint officiel facebookresearch (téléchargé directement) | Auto-supervisé sur ImageNet |

Même architecture, mêmes données d'entraînement : **seul l'objectif de
pré-entraînement diffère**. Les poids V2 de torchvision sont écartés car ils
utilisent une autre recette d'entraînement.

**Chargement (étape 1b, `configs/extract.yaml`)**
- Les deux fichiers de poids sont téléchargés directement depuis leur source
  officielle et vérifiés par empreinte SHA-256, puis chargés dans le ResNet-50
  de torchvision. Pas de `torch.hub.load`, qui exécuterait du code distant ;
  c'est d'ailleurs exactement ce que fait le dépôt officiel de Barlow Twins.
- Clés absentes tolérées, et seulement elles : la couche de classification
  `fc` (absente du checkpoint Barlow Twins, retirée de toute façon) et les
  compteurs `num_batches_tracked` des BatchNorm (absents du checkpoint
  supervisé V1, inutilisés en mode évaluation).
- Vérifié sur 8 images de validation : sortie (N, 2048) finie ; même entrée →
  même sortie ; supervisé identique au modèle torchvision officiel.

**Extraction (étapes 1d–1e)**
- Faute de GPU disponible sur Colab gratuit, l'extraction a tourné **sur le Mac
  (GPU Apple M2, MPS)**, révision `145fe16`, calcul en float32, stockage en
  float16. Sur 16 images, écart avec un calcul CPU float32 ≤ 0,0005 (cosinus
  1,000000) : l'appareil n'a pas d'effet. Tous les vecteurs viennent du même
  passage.
- Durée : 94 min pour 201 172 images (train 70 min, val 12 min, test 12 min).
- 12 fichiers (2 backbones × 2 résolutions × 3 parties), 3,1 Go, vérifiés :
  formes, ordre identique aux fichiers `splits/`, aucune valeur non finie.
- Le test est extrait comme le reste mais **n'est lu par aucun calcul** avant
  l'évaluation finale.

### Extraction

- Vecteur = sortie avant la couche de classification : **2 048 dimensions**.
- **Deux résolutions** : le carré recadré est redimensionné en 112 × 112 et en
  224 × 224. Le recadrage fixe le contenu, la résolution fixe la taille d'entrée.
- Prétraitement **identique** pour les deux réseaux : normalisation ImageNet
  standard (moyenne / écart-type). Aucun traitement propre à un seul réseau.
- Les vecteurs sont calculés **une seule fois** et sauvegardés : un fichier par
  (réseau × résolution × partie). Tous les entraînements partent de ces fichiers.
- Normalisation L2 des vecteurs avant tout calcul de similarité.

---

## 5. Mesures préalables et choix du backbone

À faire **sans aucun entraînement de perte**, et **sur la validation**.

### (a) Mesure brute

Similarité cosinus entre les vecteurs de 2 048 tels quels. On calcule MAP@R,
P@1 et les histogrammes de similarité. C'est le plancher : ce que le réseau donne
sans travail de notre part.

Comme la ligne « Pretrained » de Musgrave, on calcule aussi une version réduite
par **ACP à 128 dimensions** (ACP ajustée sur l'entraînement), comparable à la
LDA et à la tête. **C'est cette version qui sert dans la règle de décision.**

### (b) Référence LDA

Analyse discriminante linéaire **ajustée sur les identités d'entraînement**,
projection en **128 dimensions**, appliquée aux identités de validation. Mêmes
métriques.

- C'est la meilleure transformation linéaire pour séparer les classes, obtenue
  par une formule, sans boucle d'entraînement.
- La tête est elle aussi linéaire en 128 : la LDA donne une idée de ce que les
  pertes peuvent atteindre. Ce n'est pas un plafond strict, c'est un repère.

### Lecture

| Brute | LDA | Conclusion |
|---|---|---|
| basse | haute | Information présente mais mal rangée : les pertes ont de quoi travailler |
| haute | haute | Déjà bien rangé : les pertes apporteront peu |
| basse | basse | Rien à récupérer : le dispositif gelé ne tient pas, à discuter en groupe |

### Règle de décision (fixée le 2026-11-09, avant tout calcul de l'étape 2)

Mesure : **MAP@R sur la validation**. « Brute » = version ACP 128.

1. **Résolution** : pour chaque backbone, celle (112 ou 224) dont la LDA a le
   meilleur MAP@R.
2. **Backbone** : celui dont la LDA (à sa résolution retenue) a le meilleur
   MAP@R, **sauf si les deux sont proches**, auquel cas le **supervisé**.
   - « Proches » : intervalle à 95 % de l'écart de MAP@R LDA entre les deux
     backbones contenant 0, obtenu par **bootstrap sur les identités de
     validation** (1 000 tirages avec remise, graine 0).
3. **Marge laissée aux pertes** : écart MAP@R LDA − brute sur le backbone retenu.

   | Écart | Lecture | Conséquence |
   |---|---|---|
   | ≥ 5 points | marge confortable | on continue |
   | 2 à 5 points | marge faible | on continue ; risque de pertes indistinguables, à écrire |
   | < 2 points | pas de marge | arrêt et discussion avant tout entraînement |

4. **Rien à exploiter** : si le MAP@R de la LDA est **< 5 %** pour les deux
   backbones, le dispositif gelé ne tient pas ; arrêt et discussion.

Les seuils sont des **choix raisonnés, pas des valeurs de la littérature** :
avec un écart-type supposé d'environ 1,5 point entre graines, une marge totale
inférieure à 2 points noierait forcément les écarts entre pertes ; chez Musgrave,
les pertes gagnent environ 10 points de MAP@R sur le réseau brut (CUB200), 5
points en est la moitié.

La comparaison des deux backbones bruts reste un résultat du rapport, quel que
soit le choix.

### Résultat (étapes 2c–2e, validation)

MAP@R en % (« brut » = ACP 128) :

| Backbone | Résolution | Brut | LDA | P@1 brut → LDA | AUC brut → LDA | TAR à FAR 10⁻² brut → LDA |
|---|---|---|---|---|---|---|
| supervisé | 112 | 2,31 | 6,07 | 18,2 → 32,0 | 80,5 → 86,6 | 11,2 → 28,0 |
| supervisé | 224 | 2,83 | 9,42 | 21,1 → 41,5 | 81,2 → 89,2 | 11,2 → 33,4 |
| Barlow Twins | 112 | 3,30 | 8,46 | 24,9 → 39,8 | 81,4 → 89,2 | 13,9 → 31,8 |
| Barlow Twins | 224 | 4,01 | **13,23** | 28,5 → 51,5 | 82,6 → 91,6 | 15,8 → 38,4 |

Application de la règle (`runs/20261109-182200_2e_decision/decision.json`) :
1. Résolution : **224 px** pour les deux backbones.
2. Backbone : LDA Barlow Twins − supervisé = **+3,81 points**, IC 95 %
   [+3,44 ; +4,18] par bootstrap sur les identités : écart solide →
   **Barlow Twins**.
3. Marge : 13,23 − 4,01 = **9,22 points** ≥ 5 → marge confortable.
4. LDA ≥ 5 % : vérifié.

**Retenu pour l'expérience principale : ResNet-50 Barlow Twins, 224 px.**

Lecture : cas « brut bas, LDA nettement plus haute » ; l'information d'identité
est présente mais mal rangée. À architecture et données identiques, l'objectif
auto-supervisé préserve mieux l'identité que l'objectif supervisé, aussi bien
en brut qu'après LDA. Reste à contrôler que ce n'est pas un effet de style
(planche des plus proches voisins, étape 2f).

### Planche des plus proches voisins

Pour une dizaine d'images de validation, afficher les 5 plus proches voisins.
**La regarder** : si les voisins sont la bonne personne, bien ; si ce sont des
personnes différentes avec la même coiffure ou le même style de photo, le score
mesure le style et non l'identité.

---

## 6. Les quatre fonctions de coût

| Nom | Type | Paramètres réglés | Composition du lot |
|---|---|---|---|
| `nsoftmax` | classification (softmax normalisée) | échelle *s*, taux d'apprentissage | 32 personnes × 1 image |
| `arcface` | classification à marge angulaire | *s*, marge *m*, taux d'apprentissage | 32 × 1 |
| `contrastive` | paires | marge négative, taux d'apprentissage | 8 personnes × 4 images |
| `triplet` | triplets + sélection semi-difficile | marge, taux d'apprentissage | 8 × 4 |

- `nsoftmax` est la softmax **normalisée** (poids et vecteurs normalisés L2),
  c'est-à-dire ArcFace avec *m* = 0. Les 4 pertes vivent ainsi sur la même sphère,
  et l'écart `arcface` − `nsoftmax` isole exactement l'effet de la marge.
- `contrastive` : marge des paires positives fixée à 0.
- `triplet` : sélection des triplets **semi-difficiles** toujours active, non
  réglée. Sans elle, presque tous les triplets sont déjà satisfaits et rien
  n'apprend.
- Implémentation : bibliothèque **pytorch-metric-learning** (Musgrave), pour
  éviter quatre implémentations maison de qualité inégale.
  **[à valider]** Attention : dans cette bibliothèque, la marge d'ArcFace est en
  **degrés** et la softmax normalisée prend une température (*s* = 1 / température).

---

## 7. Tête et entraînement

- **Tête** : une seule couche linéaire 2 048 → **128**, suivie d'une
  normalisation L2. Pas de couche cachée.
- Les matrices de poids propres à `nsoftmax` et `arcface` ne servent qu'à
  l'entraînement et sont jetées ensuite. L'évaluation n'utilise que les vecteurs
  de 128.
- **Optimiseur** : RMSprop, le même pour les 4 pertes.
- **Taux d'apprentissage** : réglé pour chaque perte (section 8). Un seul taux
  pour la tête et la matrice de poids de la perte.
- **Lot** : 32 vecteurs, composition selon la section 6.
- **Arrêt** : MAP@R mesuré sur la validation à la fin de chaque époque ; on garde
  la meilleure version ; arrêt après **5 époques sans progrès** ; **maximum 50
  époques**.

---

## 8. Réglage des paramètres

- **Méthode** : tirage au hasard dans des plages **fixées avant tout lancement**.
- **Budget** : **10 configurations par perte**, toutes avec la graine de réglage
  **100**. Ajustable si la mesure du temps montre que c'est trop long, mais le
  même nombre pour chaque perte.
- **Critère** : meilleur **MAP@R en validation**.

### Plages

**[à valider]** À revérifier contre les valeurs par défaut de
pytorch-metric-learning avant de les figer.

| Perte | Paramètre | Plage |
|---|---|---|
| toutes | taux d'apprentissage | 1e-5 à 1e-2, échelle logarithmique |
| `nsoftmax` | *s* | 8 à 64 |
| `arcface` | *s* | 8 à 64 |
| `arcface` | *m* | 0,1 à 0,6 radian |
| `contrastive` | marge négative | 0,3 à 1,5 |
| `triplet` | marge | 0,01 à 0,5 |

### Pourquoi MAP@R et pas la vérification pour choisir

- La vérification demande un seuil ou un FAR, donc un choix arbitraire de plus.
- La plupart des paires de personnes différentes sont faciles : l'aire sous la
  ROC départage mal les configurations.
- Aux faibles FAR, la mesure dépend de quelques paires et devient bruitée.
- MAP@R n'a pas de seuil, regarde les voisins les plus proches (les cas
  difficiles) et est moins bruitée (Musgrave : autocorrélation 0,81 contre 0,73
  pour P@1).

À la fin, on vérifie si MAP@R et la vérification désignent la même configuration.

---

## 9. Graines et statistiques

### Graines

- **5 graines : 0, 1, 2, 3, 4**, les mêmes pour les 4 pertes.
- Graine de réglage à part : 100.
- Une graine fixe `random`, `numpy` et `torch` : initialisation de la tête et
  ordre des lots.
- Coût : 4 pertes × 5 graines = 20 entraînements finaux.

### Intervalles de confiance

- IC à 95 % de Student sur les 5 graines, pour **chaque chiffre** rapporté.
- Limite : il ne mesure que la variation due aux graines, pas celle due au choix
  des personnes du test.

### Comparaison de deux pertes

- **Comparaison appariée graine par graine** : pour chaque graine, écart entre
  les deux pertes ; puis moyenne et IC à 95 % sur ces 5 écarts.
- Le hasard commun aux deux pertes s'annule, donc l'intervalle est plus serré
  sans entraînement de plus.
- Deux pertes sont déclarées différentes si l'IC de leur écart exclut 0.
- Correction pour comparaisons multiples (Holm) : **en option**, voir section 12.

### Consignation

- Chaque run écrit son propre dossier, **jamais écrasé** : configuration,
  graine, métriques par époque, révision du dépôt, journal.
- Tout est consigné, y compris les runs ratés.

---

## 10. Évaluation

Toutes les mesures sortent de la même matrice de similarités cosinus entre
vecteurs.

### Sur la validation (pendant le travail)

MAP@R pour le réglage et l'arrêt ; mesures préalables de la section 5 ; seuil de
vérification.

### Sur le test (une seule fois, à la fin)

**Vérification** — présentée en premier :
- courbe ROC et aire sous la courbe ;
- **TAR à FAR = 10⁻¹, 10⁻², 10⁻³** ;
- précision au seuil choisi **sur la validation**.

Paires : toutes les paires positives, et un échantillon fixe de
**1 000 000 de paires négatives** tirées avec la graine 0 (validé le
2026-11-09), identiques pour toutes les méthodes. En validation : 229 307 paires
positives sur 194 millions de négatives possibles. Les paires
négatives partagent des images, donc elles ne sont pas indépendantes : on en
affirme moins que leur nombre ne le laisse croire.

**Recherche par plus proches voisins** :
- **MAP@R**, R-Precision, **P@1**, chaque image servant tour à tour de requête,
  les autres de références (elle-même exclue).

**Géométrie** :
- **Histogrammes de similarité** des paires « même personne » et « personnes
  différentes », superposés. La zone de chevauchement, ce sont les erreurs.
  Figure centrale du poster.
- **Alignement et uniformité** (Wang & Isola, 2020) :
  $\mathcal{L}_{\text{align}} = \mathbb{E}_{\text{pos}} \lVert f(x)-f(y) \rVert_2^2$,
  $\mathcal{L}_{\text{unif}} = \log \mathbb{E}_{x,y}\, e^{-2\lVert f(x)-f(y) \rVert_2^2}$.
  Chaque perte devient un point dans un plan. Descriptif seulement pour les pertes
  à marge (section 13).

**Planche des plus proches voisins** sur les modèles finaux, comme en section 5.

---

## 11. Méthodes classiques

Mêmes partitions, même recadrage, mêmes métriques.

| Méthode | Référence | Principe | Comparaison |
|---|---|---|---|
| Eigenfaces | Turk & Pentland, 1991 | ACP sur les pixels | cosinus |
| Fisherfaces | Belhumeur et al., 1997 | ACP vers *n* − *C* dimensions, puis LDA | cosinus |
| LBP | Ahonen et al., 2006 | histogrammes de motifs binaires locaux par cases | **distance du khi-deux** |

- Fisherfaces, c'est la LDA appliquée aux pixels ; notre référence LDA, c'est la
  LDA appliquée aux vecteurs ResNet. Le parallèle est à montrer sur le poster.
- LBP se compare par khi-deux et non par cosinus : entorse à « même mesure
  partout », à écrire.
- **[à valider]** Images en niveaux de gris, taille réduite (par exemple 64 × 64)
  pour Eigenfaces et Fisherfaces.

---

## 12. Options pour plus tard

À envisager sérieusement une fois l'expérience principale terminée.

| Option | Principe | Remarque |
|---|---|---|
| Robustesse | Dégradations appliquées **au test seulement**, plusieurs intensités : masque bas du visage, bande sur les yeux, flou, basse résolution, luminosité | Avec un réseau gelé, on mesure surtout la robustesse du réseau, peu celle de la perte |
| Augmentation de données | Plusieurs versions augmentées de chaque image extraites à l'avance | Rapprocherait de Musgrave ; coûte des extractions en plus |
| Correction de Holm | Rendre les 6 comparaisons plus exigeantes pour garder 5 % de risque total | À revoir quand le principe est bien compris |
| Essai préalable | Une perte × 5 graines pour mesurer la vraie variation entre graines | Dirait si 5 graines suffisent |

Écartés pour l'instant : spectre des valeurs singulières, évaluation sur LFW.

---

## 13. Limites à écrire dans le rapport

**Sur ce qu'on mesure**
- On mesure l'effet de la perte sur une **réorganisation linéaire d'un réseau
  gelé**, pas sur un réseau qu'elle aurait façonné. On ne peut donc ni confirmer
  ni réfuter les classements publiés.
- Nos chiffres ne sont comparables à **aucun chiffre publié**. Ils mesurent un
  écart entre 4 pertes à données constantes, pas une performance absolue.
- Détection et alignement hors périmètre : on part des recadrages fournis.

**Sur l'équité de la comparaison**
- Les lots diffèrent selon le type de perte (8 × 4 ou 32 × 1). C'est nécessaire,
  mais c'est une différence.
- `arcface` a 3 paramètres réglés contre 2 pour les autres : avec le même nombre
  d'essais, elle explore moins bien son espace.
- `triplet` est désavantagée par les petits lots (FaceNet utilisait des lots
  d'environ 1 800 images). C'est une limite du dispositif, pas un verdict sur la
  perte.
- Pas d'augmentation de données, contrairement à Musgrave.

**Sur les statistiques**
- Les IC ne couvrent que la variation due aux graines.
- Les paires négatives ne sont pas indépendantes.
- Aux faibles FAR, peu de paires décident du résultat.

**Sur les données**
- Célébrités : le style (coiffure, maquillage, époque) peut aider à reconnaître
  sans regarder le visage. Le recadrage le réduit sans l'éliminer ; la planche des
  voisins sert à le contrôler.
- Pas d'annotations démographiques fiables : l'équité entre groupes n'est pas
  mesurable.

**Sur les mesures**
- Alignement et uniformité sont fondés théoriquement pour les pertes
  contrastives de type InfoNCE ; pour les pertes à marge, c'est descriptif.
- LBP comparé par khi-deux, pas par cosinus.

**Sur les écarts à Musgrave**
- Un seul découpage de validation au lieu de 4 plis, 10 essais au hasard au lieu
  de 50 bayésiens, 5 graines au lieu de 10, réseau gelé au lieu de fine-tuning.

---

## Journal des modifications

| Date | Modification |
|---|---|
| 2026-11-09 | Première version, à partir des décisions prises en séance |
| 2026-11-09 | Section 3 : résultat de la partition (découpage officiel disjoint, effet du filtre) |
| 2026-11-09 | Section 3 : recadrage fixé à 128 px après inspection de la planche |
| 2026-11-09 | Section 4 : chargement vérifié des deux backbones (SHA-256, clés tolérées) |
| 2026-11-09 | Section 4 : extraction complète sur GPU Apple (Colab sans GPU) |
| 2026-11-09 | Section 5 : règle de décision de l'étape 2 fixée avant tout calcul ; ACP 128 validée |
| 2026-11-09 | Section 10 : 1 000 000 de paires négatives, graine 0 |
| 2026-11-09 | Section 5 : résultats de l'étape 2 et choix (Barlow Twins, 224 px) |
