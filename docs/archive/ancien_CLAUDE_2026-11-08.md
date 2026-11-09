# CLAUDE.md — Projet SRD sujet 09

Contexte permanent du projet. À lire avant d'écrire la moindre ligne.
Si une instruction ponctuelle contredit ce fichier, demander confirmation avant d'agir.

---

## 1. Le projet

**Module** SRD, M1 informatique, Aix-Marseille Université. Groupe de 4.
**Coach** Stéphane Ayache (LIS, équipe Qarma, Luminy).
**Sujet 09** « Algorithmique de la reconnaissance faciale, avant et après IA ».

**Livrables** : bibliographie annotée (1 page, anglais — *faite*), synthèse
d'avancement (2 pages, anglais), poster A0 portrait en anglais soutenu à l'oral.

**Charge réaliste** : 3 à 4 heures par semaine et par personne. Environ
200 heures-personne au total, dont une bonne partie partira dans le pipeline
de données.

### Le titre est inexact, et on l'exploite

L'ACP fait déjà de l'apprentissage ; « avant l'IA » est un raccourci. La vraie
coupure est entre **descripteurs conçus à la main** et **descripteurs appris**,
avec DeepFace en 2014. On ouvre le poster là-dessus au lieu d'attendre qu'on nous
le reproche.

### La question de recherche

> Qu'est-ce que la **fonction de coût** fait à la **géométrie des représentations
> d'identité** ?

Ce n'est pas un panorama historique. C'est une comparaison contrôlée.

**Critère** : généralisation à des identités *jamais vues* à l'entraînement
(ensemble ouvert). C'est la situation réelle d'un système : reconnaître des gens
qui n'existaient pas dans les données.

### Périmètre

La chaîne complète est **détection → alignement → description → comparaison**.
On ne traite que la **description** (image → vecteur), sur des recadrages déjà
alignés fournis avec le jeu de données.

Détection et alignement sont hors sujet : **ne pas les implémenter**. À écrire
explicitement dans le rapport, sinon on nous demandera pourquoi on a sauté la
moitié de la chaîne.

### Ce que le projet n'est pas

Ce n'est pas une recherche originale. C'est une **réplication à petite échelle de
Musgrave et al. (2020), transposée aux visages**. Le rapport doit le dire. Nos
chiffres ne seront comparables à aucun chiffre publié, et toute comparaison de ce
genre serait trompeuse.

---

## 2. Le dispositif expérimental

Un backbone pré-entraîné **gelé**, une tête de projection **linéaire** identique,
mêmes données, mêmes partitions, même prétraitement, même protocole d'évaluation.

**Seule la fonction de coût varie.**

### Pourquoi pas les modèles publics déjà entraînés sur visages

On trouve des ArcFace et des FaceNet téléchargeables. Les utiliser comme termes de
comparaison serait une faute : ils diffèrent par les données, l'architecture, la
durée d'entraînement et les réglages. Quatre variables changent à la fois, aucune
conclusion n'est possible.

Ils peuvent apparaître comme **repère**, clairement étiqueté comme tel, jamais
dans la comparaison contrôlée.

Conséquence assumée : nos modèles seront nettement moins performants que ceux du
commerce. Ce n'est pas le but. Un écart mesuré proprement entre modèles moyens
vaut mieux qu'un classement invalide entre modèles excellents.

---

## 3. Les quatre fonctions de coût

| Nom | Type | Paramètres |
|---|---|---|
| `nsoftmax` | classification | `scale` |
| `contrastive` | paires | `margin` |
| `triplet` | triplets | `margin` + mining semi-difficile |
| `arcface` | marge angulaire | `scale`, `margin` |

### `nsoftmax` est la softmax **normalisée**, pas la softmax ordinaire

Poids et vecteurs normalisés $L_2$. C'est exactement ArcFace avec $m = 0$.

Choix délibéré, trois raisons :

1. les quatre pertes vivent alors sur la même hypersphère, donc les mesures de
   géométrie s'appliquent partout de la même façon ;
2. l'écart `arcface` − `nsoftmax` isole **exactement la marge**, rien d'autre ;
3. ça nous rend comparables au « N. Softmax » de Musgrave.

Prix à payer : on perd le lien direct avec DeepFace, qui utilisait la version
ordinaire. Une ligne dans le rapport et c'est réglé.

### Une seule implémentation pour la famille angulaire

Formulation unifiée d'ArcFace :

$$\mathcal{L} = -\log \frac{e^{s\,(\cos(m_1\theta_y + m_2) - m_3)}}
{e^{s\,(\cos(m_1\theta_y + m_2) - m_3)} + \sum_{j \neq y} e^{s\cos\theta_j}}$$

- softmax normalisée : $m_1 = 1,\; m_2 = 0,\; m_3 = 0$
- SphereFace : $m_1 > 1$ — **ne pas implémenter**, instable à l'entraînement
- CosFace : $m_3 > 0$ — implémentable, **hors expérience principale**
- ArcFace : $m_2 > 0$

### Pas de cinquième perte

Chaque perte supplémentaire consomme le budget de réglage et les graines, et les
graines sont ce qui rend les conclusions valides. Center Loss et CosFace restent
en bibliographie.

Si malgré tout une cinquième est ajoutée, c'est **Center Loss** : c'est un
mécanisme différent (compacité intra-classe explicite), pas une autre position de
marge. Mais elle ajoute deux hyperparamètres, ce qui alourdit la règle du budget
identique.

### Sur la baseline softmax et les filiations

Il n'existe pas d'article dédié à la softmax comme perte de reconnaissance
faciale : c'est l'entraînement par classification de DeepFace et DeepID.

À savoir pour le rapport : **deux des quatre pertes ne viennent pas du monde des
visages**. Contrastive vient de Hadsell, Chopra & LeCun (2005–2006) et triplet de
l'apprentissage métrique général (LMNN de Weinberger & Saul) ; FaceNet les importe
dans le domaine des visages. ArcFace, CosFace, SphereFace et Center Loss sont
natives du monde des visages.

C'est une **convergence de deux lignées**, et c'est une meilleure histoire qu'une
filiation unique. C'est aussi l'argument qui justifie de transposer la
méthodologie de Musgrave : si les pertes circulent dans les deux sens, le
protocole peut circuler aussi.

---

## 4. Règles non négociables

Elles viennent de Musgrave, Belongie & Lim (2020), *A Metric Learning Reality
Check*, arXiv:2003.08505. Toute violation invalide le résultat.

1. **Partition par identité.** Aucune personne du test n'apparaît à
   l'entraînement ni en validation. Un test automatique le vérifie. Ce test ne
   doit jamais être désactivé.
2. **Budget de réglage identique** pour chaque perte : même nombre de
   configurations, tirées dans des plages fixées à l'avance.
3. **Réglage sur la validation uniquement.** Le test est consulté **une seule
   fois**, à la toute fin.
4. **Plusieurs graines** sur la meilleure configuration — viser 5, minimum 3 — et
   **les mêmes graines pour toutes les pertes**.
5. **Intervalle de confiance sur chaque chiffre.** Deux méthodes dont les
   intervalles se chevauchent largement **ne sont pas** déclarées différentes.
6. **Tout consigner**, y compris les expériences ratées. Un fichier de résultats
   par run, jamais écrasé. Graines, configuration et révision du dépôt dedans.

### Comparaison appariée

Puisque toutes les pertes tournent sur les mêmes graines, comparer les
**différences** graine par graine plutôt que les moyennes : la variance commune
(ordre des données, initialisation) s'annule, ce qui resserre nettement les
intervalles sans un seul entraînement de plus.

### Sur le nombre de graines

Le multiplicateur de Student explose quand il y a peu de mesures : 4,30 à
3 graines, 2,78 à 5, 2,26 à 10 (95 %). Avec un écart-type de 1,5 point, la
demi-largeur de l'intervalle passe de ±3,7 points à 3 graines à ±1,9 à 5.

Or les écarts entre pertes sont typiquement de cet ordre. **À 3 graines on peut
déclarer indistinguables deux pertes qui diffèrent réellement** — et c'est le pire
cas : une conclusion prudente et fausse.

**Faire un pilote** : une seule perte, 5 graines, même configuration, pour mesurer
l'écart-type réel du dispositif. Deux heures, et ça dimensionne tout le reste.
Payer les graines en coupant les configurations d'hyperparamètres : 3 configs × 5
graines vaut mieux que 4 × 3.

---

## 5. Ce que Musgrave a fait, exactement

Utile parce qu'on s'en inspire et qu'il faut savoir où on s'en écarte.

**Les trois défauts qu'il dénonce**

*Comparaisons inéquitables.* Le tableau 1 de l'article donne la performance des
réseaux **avant tout apprentissage métrique**, juste pré-entraînés ImageNet :
GoogleNet 41,1 / BN-Inception 51,1 / ResNet50 48,7 sur CUB200. Dix points d'écart
avant d'avoir rien fait. Un article de 2017 a utilisé ResNet50 là où ses
concurrents utilisaient GoogleNet, puis annoncé de gros gains.

Et le détail qui résume tout : dans le code public d'un article de 2019, les
BatchNorm sont gelés, les auteurs écrivent **dans un commentaire** que ça vaut 2
points, et l'article ne le mentionne pas.

*Métriques faibles.* Recall@1 donne 100 % à quatre situations très différentes.
NMI donne 81,2 à des réseaux **initialisés au hasard** sur SOP (20 000 classes)
contre 23,6 sur CUB200 (200 classes) : elle mesure le nombre de classes, pas la
qualité.

*Retour du jeu de test.* La majorité des articles surveillent la performance de
test pendant l'entraînement et rapportent la meilleure. Pas de validation.

**Son protocole**

- BN-Inception pré-entraîné ImageNet, embedding 128 dimensions, BatchNorm gelés.
- Lot de 32. **Composition adaptée au type de perte** : 8 classes × 4 images pour
  les pertes d'embedding, 32 classes × 1 image pour les pertes de classification.
- Recadrage aléatoire redimensionné : côté court à 256, crop entre 40 et 256,
  rapport d'aspect 3/4 à 4/3, redimensionné en 227×227, miroir une fois sur deux.
  En évaluation : 256 puis crop centré 227.
- RMSprop, taux 1e-6. Pour les pertes à poids apprenables (ArcFace), le taux
  devient un hyperparamètre.
- Embeddings normalisés $L_2$.
- **50 tours d'optimisation bayésienne** par perte, chacun avec validation croisée
  à 4 plis sur la première moitié des classes, plis disjoints par classe.
- Seconde moitié des classes = test.
- **10 entraînements** avec la meilleure configuration, moyenne et IC à 95 %.
- Métrique principale **MAP@R** (autocorrélation d'ordre 1 de 0,81 contre 0,73
  pour Recall@1, donc moins bruitée).

**Son résultat.** 13 pertes testées. Sur CUB200 en MAP@R concaténé : contrastive
(2006) 26,53 ; ArcFace (2019) 26,45 ; CosFace 26,70. Les intervalles se
chevauchent. Sur Cars196 le meilleur est MS à 28,07 contre 24,89 pour contrastive ;
sur SOP ArcFace à 47,41 contre 44,39.

**D'où venaient les faux gains.** L'article de 2016 sur la lifted structure loss
implémentait les baselines en n'échantillonnant que N/2 paires et N/3 triplets par
lot, et fixait la marge du triplet à 1 quand l'optimum est vers 0,1. Tout le monde
a ensuite recopié ces chiffres bas.

**Ce qu'on reprend** : plis disjoints par classe, pas de retour du test, graines
multiples, intervalles de confiance, et surtout la composition des lots adaptée au
type de perte.

**Ce qu'on ne peut pas reprendre** : 50 tours bayésiens × 13 pertes × 10 relances.
À écrire comme un écart assumé.

**La nuance la plus importante.** Ses jeux de données sont CUB200 (oiseaux),
Cars196 (voitures) et SOP (produits). Les classes y sont des **catégories
d'objets** — exactement ce qu'un pré-entraînement ImageNet conserve et amplifie.
Notre tâche est de distinguer deux individus de la même catégorie, c'est-à-dire de
l'**information d'instance**, que l'objectif supervisé est entraîné à écraser.

On transpose donc sa méthodologie sur une tâche où son point de départ est
beaucoup moins favorable. C'est la raison précise pour laquelle l'étape 2
ci-dessous passe avant tout le reste.

À noter aussi : il a inclus CosFace et ArcFace parce que les articles
d'apprentissage métrique laissaient systématiquement de côté les pertes de
visages, alors qu'elles n'ont rien de facial. Il les a donc entraînées sur des
oiseaux et des voitures. **Aucun visage nulle part dans son article.**

---

## 6. Ordre de travail

### Étape 0 — infrastructure

Dépôt, environnement, partition par identité, test de non-recouvrement.
*Livrable : les fichiers d'identités versionnés dans le dépôt.*

### Étape 1 — extraction

Un tableau de vecteurs par (backbone × résolution). Une à trois heures, une seule
fois, réutilisé par tout le monde. Ensuite tous les entraînements partent de ce
fichier et ne retouchent jamais aux images.

### Étape 2 — LES DEUX MESURES QUI DÉCIDENT

Sans aucun entraînement :

**(a) La mesure brute.** Similarité cosinus entre paires de test sur les vecteurs
tels quels. Histogrammes, ROC, P@1. C'est le **plancher** : ce que le backbone
donne sans aucun travail de notre part.

**(b) La référence linéaire.** Analyse discriminante linéaire ajustée sur les
identités **d'entraînement**, projection appliquée aux identités de **test**,
mêmes métriques. C'est un calcul en forme fermée, quelques secondes, aucun
hyperparamètre, aucune boucle d'apprentissage.

Ce n'est pas un plafond mathématique strict — c'est un **repère fort et gratuit**.
Il dit jusqu'où une transformation linéaire peut aller sur ces vecteurs. Or la
tête de projection est linéaire. Donc c'est le niveau que les quatre pertes
peuvent viser.

**Lecture du résultat :**

| brute | référence linéaire | conclusion |
|---|---|---|
| basse | **haute** | cas idéal : information présente, mal rangée, grande marge |
| haute | haute | vecteurs déjà organisés, les pertes vont se tasser |
| basse | basse | rien à exploiter → le dispositif gelé ne tient pas, en discuter |

Une mesure brute basse n'est pas mauvaise en soi. Elle l'est seulement si la
référence linéaire est basse aussi.

**Ne lancer aucun entraînement de perte avant d'avoir ces deux chiffres.**

### Étape 3 — comparaison des quatre pertes

Sur **un seul** backbone, choisi à l'étape 2.

### Étape 4 — géométrie, robustesse, méthodes classiques

---

## 7. Le choix du backbone

Trois options, toutes gelées :

| Option | Nature | Réserve |
|---|---|---|
| ResNet-50 supervisé ImageNet | le défaut déclaré | objectif par catégories : écrase l'information d'instance, donc peut-être l'identité |
| ResNet-50 auto-supervisé (SwAV, Barlow Twins, MoCo v3) | même architecture, mêmes données, **seul l'objectif change** | invariances héritées, non inspectables |
| DINOv2 | borne haute, « meilleure représentation gelée disponible » | change architecture **et** objectif : un écart ne peut pas être attribué |

### Ce qui ne demande aucune décision

**Extraire avec les deux ResNet-50** coûte le même script lancé deux fois. Mesurer
la similarité brute et la référence linéaire sur les deux, pareil. À faire sans
arbitrer.

Ce n'est pas une étape préparatoire, c'est **un résultat** : à architecture et
données identiques, quel objectif de pré-entraînement préserve le mieux
l'information d'identité dans des vecteurs gelés ? Personne n'a posé la question
pour les visages, et ça ne coûte aucun entraînement.

### La vraie décision

Sur quel backbone tourne la comparaison des quatre pertes. La faire sur les deux
double l'expérience : huit entraînements, et autant de réglages et de graines.
Hors budget.

**Critère : la référence linéaire la plus haute.** C'est mesuré, pas préféré.

**Si les deux sont proches**, prendre le supervisé pour l'expérience principale.
Trois raisons pratiques : il se charge en une ligne de torchvision et personne ne
demandera de le justifier ; les checkpoints auto-supervisés demandent de retirer
la tête de projection et de renommer les clés, donc plus d'occasions de se tromper
silencieusement ; et si les chiffres paraissent bizarres, il existe des dizaines
de résultats publiés de sondes linéaires sur ResNet-50 supervisé pour se situer,
alors que sur des features auto-supervisées appliquées à des visages il n'y a rien.

**Si l'auto-supervisé est nettement devant**, le prendre, et le poster ouvre sur un
résultat au lieu d'une convention.

Dans les deux cas, la comparaison des deux backbones bruts reste dans le rapport.

---

## 8. Données

| Jeu | Contenu | Rôle |
|---|---|---|
| **CelebA** | ~200 000 images, ~10 000 identités, recadrages alignés fournis | entraînement et évaluation principale |
| **LFW** | 13 000 images, 5 749 personnes, 6 000 paires standard | évaluation secondaire |
| **Extended Yale B** | 38 personnes × 64 éclairages | expérience sur l'éclairage, optionnelle |

### Règles

**Jeux publics uniquement.** Aucune photo produite par nous ou par des proches :
ce sont des données biométriques, encadrées par le RGPD.

**Partition par identité**, avec test automatique de non-recouvrement.

**Réserve à écrire.** CelebA et LFW viennent tous deux de photos de célébrités, et
les identités de CelebA sont anonymisées. Un recouvrement entre nos données
d'entraînement et LFW est **possible et invérifiable**. D'où : évaluation
principale sur le test CelebA, LFW rapporté en secondaire avec la réserve énoncée.

### Échelle, à mettre sur le poster

| | Images | Identités |
|---|---|---|
| Eigenfaces (1991) | quelques centaines | dizaines |
| DeepFace (2014) | 4,4 M | 4 000 |
| FaceNet (2015) | 200 M | 8 M |
| **Nous** | **200 k** | **10 k** |

Avec la phrase : *nos chiffres mesurent l'écart entre quatre pertes à données
constantes, pas une performance absolue.*

---

## 9. Ce qu'on mesure

Tout sort de la même chose — la matrice des similarités cosinus entre vecteurs de
test — donc une fois celle-ci calculée, le reste est de la dérivation.

### Vérification — la tâche principale

Deux images, même personne ou non. Seuil choisi **sur la validation**.

- courbe ROC et son aire ;
- **TAR à FAR fixé** ($10^{-1}$, $10^{-2}$, $10^{-3}$) — la mesure qui compte
  opérationnellement ;
- EER ;
- précision au seuil de validation.

**Limite de taille d'échantillon, à rapporter.** Règle de trois : avec 0 fausse
acceptation sur $n$ paires négatives, la borne supérieure à 95 % est $\approx 3/n$.
Certifier un FAR de $10^{-4}$ exige donc $n \geq 30\,000$ paires négatives. Avec
les ~3 000 de LFW, **on ne peut rien affirmer en dessous de $10^{-3}$**. Ce n'est
pas une opinion, c'est un calcul — et c'est un résultat en soi.

### Identification — le pont vers Musgrave

Une seule photo de référence par personne (*one-shot*), puis pour chaque autre
image : le plus proche voisin est-il la bonne personne ?

- **P@1**, exactement sa métrique ;
- courbe CMC (rangs 1, 5, 10) ;
- **MAP@R** si possible — on a les étiquettes, donc on connaît $R$ pour chaque
  requête.

Rappel de ce que ces métriques veulent dire. Chaque image de test devient tour à
tour la requête, les autres sont les références, triées par distance.
P@1 : le premier voisin est-il correct ?
R-Precision : parmi les $R$ premiers, combien sont corrects, où $R$ est le nombre
d'images de la même personne.
MAP@R $= \frac{1}{R}\sum_{i=1}^{R} P(i)$, où $P(i)$ est la précision au rang $i$
si la $i$-ème réponse est correcte, 0 sinon — donc elle tient compte de l'**ordre**,
là où les deux autres n'en tiennent pas compte.

### Géométrie — et ce n'est pas un supplément

C'est littéralement la question posée.

- **Les deux histogrammes de similarité superposés**, positives et négatives. La
  zone de chevauchement, ce sont les erreurs. Figure centrale du poster.
- **Alignement et uniformité** (Wang & Isola, 2020) :
  $\mathcal{L}_{\text{align}} = \mathbb{E}_{\text{pos}} \lVert f(x)-f(y) \rVert_2^2$
  et
  $\mathcal{L}_{\text{unif}} = \log \mathbb{E}_{x,y} e^{-2\lVert f(x)-f(y) \rVert_2^2}$.
  Chaque perte devient un point dans un plan.
- **Spectre des valeurs singulières** de la matrice apprise : combien de directions
  chaque perte conserve réellement.

**Réserve à écrire.** La décomposition de Wang & Isola est établie pour la famille
contrastive / InfoNCE. Appliquer ces deux mesures à une perte à marge est
**descriptif**, pas théoriquement fondé.

### Robustesse

Dégradations appliquées **au test seulement**, plusieurs intensités : masque bas du
visage, bande sur les yeux, flou gaussien, résolution 112 → 56 → 28 → 14,
variation de luminosité.

Nuance : on mesure la robustesse du **backbone + tête**, et le backbone domine.
C'est donc surtout la robustesse du backbone qu'on observe, pas celle de la perte.

### Priorité si tout doit être réduit

Les histogrammes de similarité et le TAR à FAR fixé. Le premier répond à la
question, le second rend les résultats lisibles par quelqu'un du domaine. P@1 juste
après, parce qu'il coûte dix lignes et qu'il branche sur la littérature.

---

## 10. Pièges connus — déjà identifiés, ne pas les reprendre

**Composition des lots.** Les pertes par paires et triplets ont besoin de plusieurs
images de la même identité dans le lot. Un lot tiré uniformément sur 8 000
identités n'en contient quasiment aucune, et contrastive comme triplet n'apprennent
rien. → Échantillonneur PK ($P$ identités × $K$ images) pour `contrastive` et
`triplet`, tirage uniforme pour `nsoftmax` et `arcface`, et **l'écrire dans le
rapport**.

**Mining des triplets.** Des triplets tirés au hasard sont presque tous déjà
satisfaits, donc le gradient est nul. Sélection **semi-difficile** obligatoire :
$\lVert f_a - f_p \rVert^2 < \lVert f_a - f_n \rVert^2 < \lVert f_a - f_p \rVert^2 + \alpha$.
Sans elle, triplet ne converge pas et on conclurait à tort qu'elle est mauvaise.
Prendre les négatifs les plus difficiles ne marche pas non plus : gradients bruités,
mauvais minima.

**Triplet est structurellement désavantagée** dans un dispositif à petits lots.
FaceNet utilisait des lots d'environ 1800 images. À rapporter comme limite du
dispositif, pas comme un verdict sur la perte.

**Facteur d'échelle d'ArcFace.** Sous un certain seuil la softmax ne peut pas
produire une probabilité élevée et l'entraînement stagne. Condition nécessaire,
centres en simplexe régulier :

$$s \;\geq\; \frac{C-1}{C}\,\log\!\left(\frac{(C-1)P}{1-P}\right)$$

Pour $C = 8000$ et $P = 0{,}9$ : $s \geq 11{,}2$. Les articles utilisent $s = 64$.
La borne est donc peu discriminante — à garder comme exercice de démonstration,
pas comme prédiction forte. Et elle suppose un simplexe régulier, qui demande au
moins $C-1$ dimensions : avec 8 000 identités en 512 dimensions, l'idéalisation ne
tient pas.

**Montée progressive de la marge.** Marge à 0 sur les premières époques puis montée
linéaire, sinon ArcFace part mal depuis une initialisation aléatoire.

**Parité de prétraitement entre backbones.** Le protocole standard de sonde
linéaire applique une normalisation particulière aux vecteurs. L'appliquer à un
backbone et pas à l'autre biaise la comparaison en faveur du supervisé. → Un seul
prétraitement, mêmes réglages partout, consigné dans le fichier de résultats.

**Résolution d'entrée.** Un ResNet-50 ImageNet attend du 224 px représentant une
scène entière. On lui donne un recadrage serré de visage en 112 px : doublement
hors distribution. → Extraire en 112 **et** en 224 suréchantillonné, garder celle
qui donne la meilleure référence linéaire, écrire le choix. Dix minutes de calcul,
et ça peut changer le diagnostic.

**Le confondant « style ».** CelebA, ce sont des célébrités. Même recadré serré, il
reste la coiffure, la couleur de cheveux, le maquillage, le grain et l'époque de la
photo. Un ResNet ImageNet capte très bien tout ça.

Un bon score brut pourrait donc venir de « blonde aux cheveux longs, photo de
tapis rouge des années 2000 » et non du visage. Ce serait un **faux positif de
diagnostic** : on conclurait que l'extracteur contient de l'information d'identité
alors qu'il contient de l'information de style.

→ Produire une planche des cinq plus proches voisins pour une dizaine de requêtes,
et **la regarder**. Si les « bonnes » réponses sont la bonne personne, tant mieux.
Si ce sont des gens différents avec la même coiffure, on sait à quoi s'en tenir.
C'est la seule partie du diagnostic qui demande des images et non des nombres.
Ne pas la sauter.

**Inégalité du budget de réglage.** `nsoftmax` a un hyperparamètre, `arcface` en a
deux ou trois. Explorer un espace à trois dimensions avec le même nombre de tirages
qu'un espace à une dimension défavorise mécaniquement les pertes les plus
paramétrées. C'est un problème ouvert dans la littérature, pas une faute — mais il
faut le dire, pas le cacher derrière « budget identique ».

---

## 11. Ce qu'on peut mesurer, et ce qu'on ne peut pas

À écrire dans le rapport. C'est ce qui distingue un travail honnête d'un travail
qui surinterprète.

### Mesurable

Sans entraînement : l'information d'identité déjà présente dans le backbone ; la
référence linéaire ; la comparaison supervisé contre auto-supervisé à architecture
identique.

Avec la tête entraînée : l'effet de la fonction de coût **sur une réorganisation
linéaire d'une représentation figée** — c'est cet énoncé précis, pas « l'effet de
la fonction de coût » tout court ; la géométrie produite ; la généralisation à des
identités inconnues ; les balayages d'hyperparamètres ; les intervalles de
confiance.

Indépendamment du backbone : Eigenfaces, Fisherfaces, LBP sur les mêmes données ;
le sous-espace d'éclairage sur Yale B.

### Non mesurable

- **Ce que la perte ferait si elle pouvait façonner le réseau.** Limite principale.
  On mesure la perte comme réorganisateur linéaire, pas comme constructeur de
  représentation. Les classements publiés viennent du second régime.
- **Donc on ne peut ni confirmer ni réfuter les classements publiés.**
- **Le régime pour lequel les marges ont été conçues.** ArcFace vise des dizaines
  de milliers d'identités ; avec 8 000 classes en 512 dimensions la géométrie
  diffère.
- L'effet de l'échelle des données, de l'architecture, de l'alignement : fixés ou
  hors périmètre.
- **Les invariances héritées du pré-entraînement.** On peut observer leurs
  conséquences, pas les attribuer.
- Les FAR sous $10^{-3}$ : taille d'échantillon insuffisante.
- L'équité démographique sur CelebA : pas d'annotations fiables.
- Les performances absolues comparables à la littérature.

### Les six moteurs du domaine, et celui qu'on isole

Le poster raconte six moteurs : (1) représentation conçue puis apprise, (2)
objectif de classification puis structuration de l'espace, (3) géométrie
euclidienne puis sphérique, (4) échelle des données, (5) alignement, (6) protocoles
d'évaluation.

**Un seul est testable à nos moyens** : la fonction de coût. Les autres
demanderaient des millions d'images, une chaîne de détection, des entraînements
complets ou des jeux de test qu'on n'a pas.

Phrase pour la soutenance : *des six moteurs identifiés, nous isolons celui qui est
testable à ressources constantes.* C'est aussi la meilleure réponse si on demande
pourquoi on n'a pas étudié l'effet des données ou de l'architecture.

---

## 12. Colonne « avant » — les méthodes classiques

### Ce qu'on code

- **Eigenfaces** (Turk & Pentland, 1991) — ACP, ~80 lignes de numpy.
- **Fisherfaces** (Belhumeur et al., 1997) — même code, critère de Fisher au lieu
  de la variance. ACP préalable vers $n-C$ dimensions obligatoire, car la matrice
  intra-classe est singulière quand $n < d$.
- **LBP** (Ahonen et al., 2006) — `skimage.feature.local_binary_pattern`, plus le
  découpage en cases et la distance du khi-deux. ~100 lignes.

### Pourquoi LBP et pas seulement Eigenfaces

La colonne « avant » a **trois générations**, pas une : sous-espace global (1991),
descripteurs locaux (2006), descripteurs locaux + encodage statistique (2013).

Eigenfaces est le bas de l'échelle. Si on ne compare les réseaux qu'à la méthode de
1991, on gagne de 30 points et ça ne mesure rien. Il faut au moins une méthode
locale.

Ordre de grandeur sur LFW : Eigenfaces 60–70 %, Fisher Vector Faces ~93 %, DeepFace
97,35 %. **L'écart à l'intérieur de la colonne « avant » est plus grand que l'écart
entre sa meilleure méthode et la première méthode « après ».**

### Note de protocole

LBP se compare par **distance du khi-deux sur histogrammes**, pas par cosinus.
C'est une entorse à la règle « même métrique de décision partout » : à écrire.

### Optionnel

**Fisher Vector Faces** (Simonyan et al., 2013) : SIFT dense sur grille fixe,
mélange de gaussiennes, encodage de Fisher. L'adversaire le plus crédible, mais
trois étapes d'implémentation. LBP remplit le même rôle pour un tiers du travail.

À savoir : dans ce contexte, SIFT n'est **pas** utilisé comme détecteur de points
d'intérêt. Un visage est lisse, les points détectés sont peu nombreux et tombent à
des endroits différents d'un visage à l'autre, donc les vecteurs ne seraient pas
comparables. On place une grille régulière et on calcule un descripteur à chaque
case, quelle que soit la texture. SURF n'a pratiquement jamais été utilisé en
reconnaissance faciale — aucune référence à citer.

---

## 13. Conventions de code

- Python ≥ 3.10, layout `src/`, paquet `facegeom`.
- Tout passe par des fichiers de configuration. **Aucune valeur en dur dans un
  script.**
- Chaque run écrit son dossier avec configuration, métriques, journal. **Jamais
  écrasé.**
- Graines fixées et consignées partout : `random`, `numpy`, `torch`.
- Les scripts sont numérotés dans l'ordre d'exécution et ne contiennent **que** de
  l'orchestration ; la logique vit dans le paquet.
- Linter et tests avant chaque commit.
- Docstrings en français, noms de symboles en anglais.
- **Regarder `pytorch-metric-learning`** (Kevin Musgrave) avant de coder les pertes
  à la main : il implémente contrastive, triplet, ArcFace et les stratégies de
  mining avec une interface commune. Quatre implémentations maison de qualité
  inégale, ce serait exactement le biais que son article dénonce.

### Les tests qui comptent

1. **Non-recouvrement des identités** entre partitions. Jamais désactivé.
2. Les pertes : valeur attendue sur un cas construit à la main, gradient non nul.
3. `arcface` avec $m = 0$ doit donner exactement `nsoftmax`.
4. L'évaluation : ROC et P@1 sur des vecteurs synthétiques séparables doivent
   donner 1,0.
5. Déterminisme : même graine, même résultat.

---

## 14. Ce qu'il ne faut pas faire

- Entraîner un réseau depuis zéro (sauf décision explicite après l'étape 2).
- Dégeler le backbone dans l'expérience principale.
- Utiliser des modèles publics entraînés sur visages comme termes de comparaison.
- Implémenter la détection ou l'alignement.
- Ajouter une cinquième perte.
- Comparer nos chiffres à des chiffres publiés.
- Produire des photos de visages nous-mêmes.
- Toucher au jeu de test avant la fin.
- Lancer un entraînement avant d'avoir les deux mesures de l'étape 2.

---

## 15. Répartition

| Rôle | Travail |
|---|---|
| Arsène | partie mathématique, conception du dispositif et garantie d'équité, relecture des analyses |
| Membre 2 | extraction, entraînements des pertes |
| Membre 3 | mesures de géométrie, courbes ROC et figures |
| Membre 4 | Eigenfaces, Fisherfaces, LBP, robustesse |

Tout le monde lit l'introduction et la conclusion de Musgrave : la soutenance ne
doit pas reposer sur une seule personne.

---

## 16. Références

- **Musgrave, Belongie & Lim (2020)**, *A Metric Learning Reality Check*, ECCV,
  [arXiv:2003.08505](https://arxiv.org/abs/2003.08505) — le protocole vient de là.
  Code : [powerful-benchmarker](https://github.com/KevinMusgrave/powerful-benchmarker).
  Bibliothèque : [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning).
- **Wang & Isola (2020)**, alignement et uniformité, ICML,
  [arXiv:2005.10242](https://arxiv.org/abs/2005.10242).
- **Deng et al. (2019)**, *ArcFace*, CVPR,
  [arXiv:1801.07698](https://arxiv.org/abs/1801.07698) — formulation unifiée.
- **Schroff et al. (2015)**, *FaceNet*, CVPR,
  [arXiv:1503.03832](https://arxiv.org/abs/1503.03832) — triplet, mining semi-difficile.
- **Chopra, Hadsell & LeCun (2005)**, CVPR — contrastive sur visages.
  Formulation générale : Hadsell et al. (2006).
- **Wen et al. (2016)**, *Center Loss*, ECCV.
- **Taigman et al. (2014)**, *DeepFace*, CVPR — la bascule.
- **Sun et al. (2014)**, *DeepID2*, NIPS,
  [arXiv:1406.4773](https://arxiv.org/abs/1406.4773) — l'ancrage historique.
- **Parkhi et al. (2015)**, *VGGFace*, BMVC — les données comptent autant que l'objectif.
- **Turk & Pentland (1991)**, J. Cognitive Neuroscience 3(1), 71–86,
  doi:10.1162/jocn.1991.3.1.71.
- **Belhumeur et al. (1997)**, IEEE TPAMI 19(7), 711–720, doi:10.1109/34.598228.
- **Ahonen et al. (2006)**, IEEE TPAMI 28(12), 2037–2041.
- **Liu et al. (2015)**, *CelebA*, ICCV.
- **Huang et al. (2007)**, *LFW*, UMass Amherst TR 07-49.
- **Caron et al. (2020)**, *SwAV*, NeurIPS,
  [arXiv:2006.09882](https://arxiv.org/abs/2006.09882) ;
  **Zbontar et al. (2021)**, *Barlow Twins*, ICML,
  [arXiv:2103.03230](https://arxiv.org/abs/2103.03230).

---

## 17. Dette connue

Les annotations de la bibliographie ont été rédigées à partir d'une connaissance
du domaine, et seuls les éléments bibliographiques (lieux de publication, volumes,
pages, trois DOI) ont été vérifiés contre les registres des éditeurs. Une passe de
vérification de chaque annotation contre l'article d'origine reste à faire.
