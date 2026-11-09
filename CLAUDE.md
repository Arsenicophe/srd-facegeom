# CLAUDE.md — Projet SRD sujet 09

Contexte court, chargé à chaque session. Le détail est ailleurs :

- **`docs/protocole.md`** — le protocole expérimental. **Source de vérité** :
  en cas de doute ou de contradiction, c'est lui qui fait foi.
- `references/musgrave2020_fiche.md` — fiche de lecture de l'article de référence
  (PDF à côté).
- `docs/archive/` — anciens documents, à ne pas suivre.

## Le projet

Module SRD, M1 informatique, Aix-Marseille Université, groupe de 4. Sujet 09
« Algorithmique de la reconnaissance faciale, avant et après IA ».
Livrables en anglais : synthèse d'avancement (2 pages), poster A0 soutenu à l'oral.

**Question** : qu'est-ce que la fonction de coût change à la façon dont les
visages sont rangés dans l'espace des vecteurs, et est-ce que ça aide à
reconnaître des personnes jamais vues ?

**Dispositif** : ResNet-50 pré-entraîné **gelé**, tête linéaire 2048 → 128,
CelebA, 4 pertes (`nsoftmax`, `arcface`, `contrastive`, `triplet`). Seule la
perte change. Méthode inspirée de Musgrave et al. (2020), **sans** la reprendre
en entier : c'est un projet universitaire, on vise l'essentiel.

## Où tourne quoi

Tout le calcul tourne sur **Google Colab (version gratuite)** : la machine est
effacée à chaque déconnexion, les sessions sont limitées en durée.

| Quoi | Où |
|---|---|
| Code | GitHub `Arsenicophe/srd-facegeom` (public) ; Colab le clone à chaque session |
| CelebA | Google Drive (compte d'Arsène), **un seul zip**, copié et décompressé sur Colab à chaque session |
| Vecteurs extraits, résultats des runs | Google Drive |
| Copie de travail du code | `~/Projets/SRD` sur le Mac d'Arsène, **hors d'iCloud** |

Conséquences pour le code : les notebooks ne font que monter Drive, cloner,
installer et lancer un script ; les traitements longs écrivent par morceaux et
savent reprendre après une coupure ; chaque run enregistre les versions des
bibliothèques.

## Façon de travailler

- Découper chaque demande en **petites étapes**. Faire la première, s'arrêter,
  attendre la validation d'Arsène, puis passer à la suivante.
- Proposer des options avec une recommandation, ne pas décider à la place du
  groupe. Les choix non discutés sont marqués **[à valider]**.
- Une décision prise est écrite dans `docs/protocole.md`, avec une ligne dans son
  journal.
- Expliquer simplement, en français.

## Règles qu'on ne viole jamais

1. Aucune personne du test n'apparaît à l'entraînement ni en validation. Un test
   automatique le vérifie et n'est jamais désactivé.
2. Le test est consulté **une seule fois**, à la fin. Tous les choix (backbone,
   résolution, réglages, seuils) se font sur la validation.
3. Mêmes conditions pour les 4 pertes, même nombre d'essais de réglage.
4. Mêmes graines pour toutes les pertes, intervalle de confiance sur chaque chiffre.
5. Tout est consigné, runs ratés compris. Un dossier par run, jamais écrasé.
6. Aucun entraînement de perte avant les mesures préalables (brute + LDA).

## Ce qu'on ne fait pas

- Dégeler le backbone ou entraîner un réseau depuis zéro.
- Utiliser des modèles publics entraînés sur visages comme terme de comparaison.
- Implémenter la détection ou l'alignement.
- Ajouter une cinquième perte.
- Comparer nos chiffres à des chiffres publiés.
- Utiliser des photos produites par nous ou nos proches.

## Conventions de code

- Python ≥ 3.10, layout `src/`, paquet `facegeom`.
- Tous les réglages dans des fichiers de configuration, aucune valeur en dur.
- Scripts numérotés dans l'ordre d'exécution, orchestration seulement ; la logique
  vit dans le paquet.
- Graines fixées pour `random`, `numpy`, `torch`.
- Pertes via `pytorch-metric-learning`, pas d'implémentation maison.
- Docstrings en français, noms de symboles en anglais.
- Linter et tests avant chaque commit.

**Tests prioritaires** : non-recouvrement des identités ; `arcface` avec m = 0
égale `nsoftmax` ; métriques à 1,0 sur des vecteurs synthétiques séparables ;
même graine, même résultat.

## Répartition

| Qui | Travail |
|---|---|
| Arsène | partie mathématique, conception du dispositif, garantie d'équité, relecture |
| Membre 2 | extraction, entraînements des pertes |
| Membre 3 | mesures de géométrie, courbes, figures |
| Membre 4 | Eigenfaces, Fisherfaces, LBP, robustesse |

## État du projet

- [x] Bibliographie annotée (1 page, anglais)
- [x] Protocole expérimental rédigé (`docs/protocole.md`), 4 points [à valider]
- [x] Étape 0 : dépôt git, notebook Colab, CelebA sur Drive, partition (`splits/`),
      test de non-recouvrement (`tests/test_splits.py`)
- [ ] Étape 1 : extraction des vecteurs
- [ ] Étape 2 : mesures préalables, choix du backbone et de la résolution
- [ ] Étapes 3 à 5 : réglage, entraînements finaux, test

Mettre cette liste à jour quand une étape est terminée.
