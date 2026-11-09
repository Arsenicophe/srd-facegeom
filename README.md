# srd-facegeom

Projet SRD (M1 informatique, Aix-Marseille Université), sujet 09 :
« Algorithmique de la reconnaissance faciale, avant et après IA ».

Question : qu'est-ce que la fonction de coût change à la façon dont les visages
sont rangés dans l'espace des vecteurs, et est-ce que ça aide à reconnaître des
personnes jamais vues ?

- Protocole expérimental : [`docs/protocole.md`](docs/protocole.md)
- Fiche de lecture de l'article de référence (Musgrave et al., 2020) :
  [`references/musgrave2020_fiche.md`](references/musgrave2020_fiche.md)

## Organisation

| Dossier | Contenu |
|---|---|
| `src/facegeom/` | toute la logique (paquet Python) |
| `scripts/` | scripts numérotés dans l'ordre d'exécution, orchestration seulement |
| `configs/` | fichiers de configuration (aucune valeur en dur dans le code) |
| `notebooks/` | notebooks Colab courts : montage de Drive, clone, lancement d'un script |
| `splits/` | listes d'identités par partition (versionnées) |
| `tests/` | tests automatiques |

Les données (CelebA), les vecteurs extraits et les résultats des runs sont sur
Google Drive, pas dans ce dépôt.

## Lancer sur Colab

Chaque session commence par [`notebooks/00_demarrage_colab.ipynb`](notebooks/00_demarrage_colab.ipynb) :
montage de Drive, récupération du code, installation, versions, et copie de
CelebA si besoin.

Ouvrir un notebook du dépôt dans Colab : Fichier → Ouvrir un notebook → onglet
GitHub → `Arsenicophe/srd-facegeom`. Le dépôt est public : aucun jeton n'est
nécessaire.

### Dossier Drive

Le notebook crée `MyDrive/srd-facegeom/{data,features,runs}` sur le Drive du
compte connecté. Pour le groupe, Arsène partage ce dossier, et chaque membre
ajoute un raccourci vers « Mon Drive » pour retrouver le même chemin.

## Installation locale

```bash
pip install -e ".[dev]"
pytest
```
