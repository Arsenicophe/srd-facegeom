# Synthèse d'avancement — brouillon

Sections validées une par une. Le fichier Word sera produit à la fin.
Règles de style : pas de tirets longs, parenthèses seulement si nécessaires.

En-tête : noms du groupe et date à compléter.

---

## 1. Context and research question — VALIDÉE

A face recognition system maps each face image to a vector and decides whether
two images show the same person by comparing their vectors. Over the history of
the field, the way this vector is obtained has changed, from hand-designed
descriptors to representations learned with a training objective, and many
objectives have been proposed for learning it. We focus on this last ingredient
and ask: **what does the training objective do to the geometry of identity
representations, and does it help recognise people never seen during
training?** To isolate the effect of the objective, we compare four losses,
namely normalised softmax, contrastive, triplet and ArcFace, trained on the same
frozen pre-trained backbone with identical data, splits, preprocessing and
evaluation. We study only the description step, which turns an image into a
vector, on the aligned face crops provided with the dataset; face detection and
alignment are out of scope.

## 2. Coach's guidance and how we applied it — VALIDÉE

Our coach, Stéphane Ayache, advised us to start from a pre-trained ResNet, to
train a linear head on top of it with each of the chosen losses, and to keep
every other factor constant so that the comparison stays fair. We applied this
advice as follows. The ResNet-50 backbone stays frozen, and only a single linear
layer, mapping its 2,048-dimensional output to a 128-dimensional embedding, is
trained. Image vectors are extracted once and shared by all experiments, so
every loss sees exactly the same data, splits and preprocessing. Each loss
receives the same tuning budget of ten configurations, drawn from ranges fixed
in advance and compared on validation identities only. The selected
configuration of each loss will then be trained with the same five random
seeds, and every result will come with a confidence interval. Test identities
are never used for training or tuning, and the test set will be evaluated only
once, at the end. To make these rules precise, we drew on the evaluation
practices of Musgrave et al. [11], who showed that unequal protocols can create
apparent differences between losses.

Note pour la section 4 : mentionner en une demi-phrase que le « brut » est
réduit à 128 dimensions par ACP pour comparer à dimension égale, les 2 048
dimensions donnant des résultats presque identiques.

## 3. Experimental setup — VALIDÉE

We use CelebA [14] with its official partition, which we verified to be
disjoint in identities, and keep only identities with at least four images so
that every loss sees the same data. This leaves 7,689 training, 924 validation
and 951 test identities, with 161,602, 19,720 and 19,850 images respectively.
Each aligned image is cropped to a 128-pixel square that keeps the face from
forehead to chin and removes most of the hair and background, then resized to
112 or 224 pixels. We compare two ResNet-50 backbones pre-trained on ImageNet:
the supervised torchvision model and the self-supervised Barlow Twins model
[15]. They share the same architecture and training images and differ only in
how they were pre-trained. Normalised softmax is the softmax loss with
L2-normalised weights and embeddings, that is ArcFace without margin, so that
all four losses work on the same hypersphere and the gap between ArcFace and
normalised softmax isolates the margin. Batches contain eight identities with
four images each for the contrastive and triplet losses, and 32 identities for
the two classification losses. Retrieval is measured with MAP@R, our main
metric, and precision at one; verification with the ROC curve and the true
acceptance rate at fixed false acceptance rates.

## 4. Progress and preliminary results — VALIDÉE

Before training any loss, we measured two references on validation identities:
the raw backbone features, reduced to 128 dimensions by PCA to match the head,
and a linear discriminant analysis fitted on training identities, which shows
how far a linear head can go. Our metrics match the reference implementation in
pytorch-metric-learning exactly, and the selection rule was fixed before these
results were computed.

As shown in Table 1, raw features rank identities poorly, with a MAP@R of 4% at
best, but LDA more than triples it: identity information is present but badly
organised, which leaves a margin of 9.2 points for the losses. Figure 1 shows
how LDA concentrates the similarities of different people around zero. With the
same architecture and training images, Barlow Twins preserves identity better
than the supervised model, by 3.8 MAP@R points with a 95% bootstrap confidence
interval of 3.4 to 4.2, so we selected it at 224 pixels. A visual check of
nearest neighbours shows that raw neighbours mostly share pose, hair or
lighting, while LDA neighbours are more often the same person.

**Table 1.** Results on validation identities, in %. TAR: true acceptance rate
at a false acceptance rate of 1%.

| Backbone | Input | MAP@R raw | MAP@R LDA | TAR raw | TAR LDA |
|---|---|---|---|---|---|
| Supervised | 112 | 2.3 | 6.1 | 11.2 | 28.0 |
| Supervised | 224 | 2.8 | 9.4 | 11.2 | 33.4 |
| Barlow Twins | 112 | 3.3 | 8.5 | 13.9 | 31.8 |
| **Barlow Twins** | **224** | **4.0** | **13.2** | **15.8** | **38.4** |

**Figure 1** (`docs/figures/report_hist_barlow224.png`). Cosine similarity of
same-person and different-people pairs on validation identities, Barlow Twins at
224 pixels, before and after LDA.
