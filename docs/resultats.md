# Résultats obtenus — étapes 0 à 2

Document de référence pour la **synthèse d'avancement (2 pages, en anglais)**.
Tout ce qui suit est mesuré sur la **validation** ; le test n'a été lu par aucun
calcul. Le détail du protocole est dans `docs/protocole.md`.

État au 2026-11-09. Prochaine étape : 3, entraînement de la tête avec les 4 pertes.

---

## 1. Ce qui est en place

| Élément | Résultat | Où |
|---|---|---|
| Dépôt | public, github.com/Arsenicophe/srd-facegeom | — |
| Données | CelebA, fichiers d'origine (copie Hugging Face figée), empreintes MD5 officielles vérifiées | `configs/data.yaml` |
| Partition | découpage officiel CelebA, **disjoint en identités** (vérifié) ; filtre ≥ 4 photos par identité | `splits/` |
| Test de non-recouvrement | 13 tests, échouent bien sur un fichier corrompu (vérifié) | `tests/test_splits.py` |
| Recadrage | carré de 128 px, yeux à 46 % de la hauteur (gabarit ArcFace), choisi à l'œil sur 4 tailles | protocole §3 |
| Backbones | ResNet-50 supervisé (torchvision V1) et Barlow Twins, poids vérifiés par SHA-256, gelés | `configs/extract.yaml` |
| Vecteurs | 2 backbones × 2 résolutions (112, 224) × 3 parties = 12 fichiers, 3,1 Go | `features/` (Mac) |
| Tests automatiques | 44, tous au vert | `tests/` |

### Taille des données après filtre

| Partie | Identités | Images | Avant filtre |
|---|---|---|---|
| Entraînement | 7 689 | 161 602 | 8 192 / 162 770 |
| Validation | 924 | 19 720 | 985 / 19 867 |
| Test | 951 | 19 850 | 1 000 / 19 962 |

Le filtre retire environ 6 % des identités et moins de 1 % des images.

---

## 2. Mesures préalables (étape 2)

« Brut » = vecteurs du backbone réduits à 128 dimensions par ACP (ajustée sur
l'entraînement), comme la ligne « Pretrained » de Musgrave. « LDA » = analyse
discriminante linéaire à 128 dimensions ajustée sur les identités
d'entraînement, appliquée aux identités de validation (jamais vues).

Vérification : 229 307 paires positives (toutes) et 1 000 000 de paires
négatives (graine 0), identiques pour toutes les lignes.

| Backbone | Rés. | Variante | MAP@R | P@1 | R-Prec | AUC | EER | TAR@10⁻¹ | TAR@10⁻² | TAR@10⁻³ |
|---|---|---|---|---|---|---|---|---|---|---|
| supervisé | 112 | brut | 2,31 | 18,20 | 4,69 | 80,47 | 27,48 | 44,15 | 11,24 | 2,78 |
| supervisé | 112 | LDA | 6,07 | 31,99 | 10,66 | 86,60 | 21,37 | 64,28 | 27,97 | 10,31 |
| supervisé | 224 | brut | 2,83 | 21,10 | 5,40 | 81,19 | 26,96 | 44,63 | 11,16 | 2,88 |
| supervisé | 224 | LDA | 9,42 | 41,54 | 14,84 | 89,19 | 18,95 | 69,75 | 33,44 | 13,88 |
| Barlow Twins | 112 | brut | 3,30 | 24,87 | 6,11 | 81,36 | 26,61 | 47,66 | 13,89 | 4,04 |
| Barlow Twins | 112 | LDA | 8,46 | 39,75 | 13,64 | 89,17 | 19,03 | 68,85 | 31,80 | 12,53 |
| Barlow Twins | 224 | brut | 4,01 | 28,47 | 7,14 | 82,57 | 25,65 | 50,06 | 15,83 | 5,05 |
| **Barlow Twins** | **224** | **LDA** | **13,23** | **51,47** | **19,12** | **91,56** | **16,61** | **74,72** | **38,44** | **17,42** |

Toutes les valeurs en %. Fichiers : `results/etape2/`. Sans ACP (2 048 dimensions),
le brut est à peine meilleur (+0,1 à +0,3 point de MAP@R) : réduire à 128 ne
détruit presque rien.

Repère : un tirage au hasard donnerait un P@1 d'environ 0,1 % (≈ 20 photos de la
bonne personne parmi 19 720).

### Décision (règle fixée et commitée avant tout calcul, commit `edad5b0`)

1. Résolution : **224 px** pour les deux backbones (meilleure LDA).
2. Backbone : LDA Barlow Twins − supervisé = **+3,81 points de MAP@R**,
   IC 95 % **[+3,44 ; +4,18]** (bootstrap apparié sur les identités, 1 000
   tirages) → écart solide → **Barlow Twins**.
3. Marge LDA − brut : 13,23 − 4,01 = **9,22 points** ≥ 5 → marge confortable.
4. LDA ≥ 5 % : oui.

**Retenu pour l'expérience principale : ResNet-50 Barlow Twins, 224 px.**

### Planche des plus proches voisins (contrôle du style)

10 requêtes de validation, 5 voisins, Barlow Twins 224 px. Planches avec des
visages : `runs/20261109-182611_2f_planche_voisins/` (Mac seulement, licence
CelebA : pas de redistribution).

- Brut : 3/50 bons voisins ; voisins semblables par la pose, le couvre-chef, la
  coiffure, l'éclairage.
- LDA : 16/50 ; nettement plus centré sur le visage ; erreurs restantes
  surtout démographiques (sexe, âge, origine, barbe), un peu de style.

### Figures (sans visages, utilisables dans le rapport)

- `docs/figures/histogrammes_brut.png` : similarités brutes, 4 combinaisons.
- `docs/figures/histogrammes_brut_vs_lda.png` : brut à gauche, LDA à droite.
  La bosse « personnes différentes » devient étroite et centrée sur 0 après
  LDA ; le chevauchement diminue nettement.

---

## 3. Messages pour le rapport

1. **Cas favorable** : brut bas, LDA nettement plus haute. L'information
   d'identité est présente dans les vecteurs gelés mais mal rangée ; une
   transformation linéaire en récupère beaucoup (MAP@R ×3, TAR@10⁻² de 16 à
   38 %). Les pertes ont 9 points de marge pour se distinguer.
2. **Résultat propre, sans entraînement** : à architecture (ResNet-50) et
   données (ImageNet) identiques, l'objectif **auto-supervisé** (Barlow Twins)
   préserve mieux l'identité que l'objectif **supervisé**, en brut comme après
   LDA, et l'écart est solide statistiquement.
3. **Résolution** : 224 px bat 112 px partout ; le réseau préfère sa taille
   d'entraînement même pour un visage agrandi.
4. **Style** : le brut range surtout par apparence globale ; la LDA s'appuie bien
   plus sur le visage, mais le confondant ne disparaît pas.

---

## 4. Écarts au protocole initial et incidents (à mentionner)

- **Extraction sur le Mac (GPU Apple M2) au lieu de Colab** : GPU gratuits de
  Colab indisponibles. Vérifié sans effet : écart ≤ 0,0005 avec un calcul CPU,
  cosinus 1,000000. Durée 94 min. Révision `145fe16`.
- **LDA recodée** : la LDA de scikit-learn (solveur eigen) calcule une covariance
  par classe, trop lent pour 7 689 classes en 2 048 dimensions ; un premier
  passage a été interrompu (`runs/20261109-180556_2d_reference_lda`, vide).
  Recodée par produits de matrices, **résultat identique** à scikit-learn
  (test d'équivalence), 23 s par ajustement.
- **Centrage** : la projection LDA est centrée sur la moyenne d'entraînement,
  comme l'ACP, car le solveur eigen ne centre pas et le cosinus y est sensible.
- **Figure refaite** : le premier tracé des histogrammes bruts coupait les
  valeurs sous −0,2 ; supprimé et refait, chiffres identiques.
- **Écarts assumés à Musgrave** : backbone gelé (pas de fine-tuning), pas
  d'augmentation, un seul découpage de validation (pas de validation croisée),
  10 essais aléatoires au lieu de 50 bayésiens, 5 graines au lieu de 10, 4
  pertes au lieu de 13.

## 5. Limites déjà visibles

- Le MAP@R reste bas en absolu (13 % au mieux après LDA) : la tâche « distinguer
  des individus » est loin de celle pour laquelle les backbones ImageNet ont
  été entraînés. Nos chiffres ne sont comparables à aucun chiffre publié.
- Planche sur 10 requêtes seulement : constat qualitatif.
- Possibles erreurs d'étiquetage ou sosies dans CelebA, non vérifiables.
- Les intervalles de l'étape 2 viennent du bootstrap sur les identités de
  validation ; les intervalles sur graines viendront à l'étape 3.

## 6. Où sont les fichiers

| Quoi | Où | Sauvegardé ailleurs ? |
|---|---|---|
| Code, protocole, ce document, tableaux, figures sans visages | dépôt GitHub | oui |
| Vecteurs (3,1 Go) | `~/Projets/SRD/features/` | **non : à envoyer sur Drive** |
| Dossiers de résultats complets, planches avec visages | `~/Projets/SRD/runs/` | **non : à envoyer sur Drive** |
| CelebA | `~/Projets/SRD/data/` et Drive | oui (Drive) |
