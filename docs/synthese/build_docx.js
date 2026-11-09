// Construit la synthèse d'avancement en Word à partir du texte validé
// (docs/synthese/brouillon.md). Mise en page reprise de la bibliographie
// annotée : Calibri, A4, marges 1,3 cm / 1,5 cm.
//
//   NODE_PATH=$(npm root -g) node docs/synthese/build_docx.js [--refs B]
//
// --refs A : liste des 7 références citées ; --refs B : une seule ligne de renvoi.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, BorderStyle, WidthType, ShadingType,
} = require("docx");

const REFS = process.argv.includes("--refs") ? process.argv[process.argv.indexOf("--refs") + 1] : "A";
const OUT = path.join(__dirname, REFS === "A" ? "synthese_avancement.docx" : "synthese_avancement_refsB.docx");
const FIG = path.join(__dirname, "..", "figures", "report_hist_barlow224.png");

const BODY = 20;      // 10 pt
const SMALL = 17;     // 8,5 pt, comme les annotations de la bibliographie
const CONTENT_W = 11906 - 850 - 850; // largeur utile A4 en DXA

// Paragraphe de texte ; les segments entre ** ** sont en gras.
function para(text, opts = {}) {
  const runs = text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((t) =>
    t.startsWith("**")
      ? new TextRun({ text: t.slice(2, -2), bold: true, size: opts.size || BODY })
      : new TextRun({ text: t, size: opts.size || BODY, italics: opts.italics }));
  return new Paragraph({
    children: runs,
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after ?? 100, line: 252 },
  });
}

function heading(text) {
  return new Paragraph({
    children: [new TextRun({ text, bold: true, size: 22 })],
    spacing: { before: 140, after: 60 },
    keepNext: true,
  });
}

function caption(label, text) {
  return new Paragraph({
    children: [new TextRun({ text: label, bold: true, size: SMALL }), new TextRun({ text, size: SMALL })],
    spacing: { before: 60, after: 60 },
    keepNext: true,
  });
}

// --- Tableau 1 ---------------------------------------------------------
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
const colW = [2400, 1100, 1300, 1300, 1150, 1150];
const tableW = colW.reduce((a, b) => a + b, 0);
const rows = [
  ["Backbone", "Input", "MAP@R raw", "MAP@R LDA", "TAR raw", "TAR LDA"],
  ["Supervised", "112", "2.3", "6.1", "11.2", "28.0"],
  ["Supervised", "224", "2.8", "9.4", "11.2", "33.4"],
  ["Barlow Twins", "112", "3.3", "8.5", "13.9", "31.8"],
  ["Barlow Twins", "224", "4.0", "13.2", "15.8", "38.4"],
];
const table = new Table({
  width: { size: tableW, type: WidthType.DXA },
  columnWidths: colW,
  alignment: AlignmentType.CENTER,
  rows: rows.map((r, i) => new TableRow({
    children: r.map((c, j) => new TableCell({
      borders,
      width: { size: colW[j], type: WidthType.DXA },
      shading: i === 0 ? { fill: "EDEDED", type: ShadingType.CLEAR } : undefined,
      margins: { top: 30, bottom: 30, left: 100, right: 100 },
      children: [new Paragraph({
        alignment: j < 2 ? AlignmentType.LEFT : AlignmentType.RIGHT,
        children: [new TextRun({ text: c, size: SMALL, bold: i === 0 || i === rows.length - 1 })],
      })],
    })),
  })),
});

// --- Figure 1 ----------------------------------------------------------
const figWidthPx = 600; // ~15,9 cm à 96 px/pouce
const figure = new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 80, after: 0 },
  keepNext: true,
  children: [new ImageRun({
    type: "png",
    data: fs.readFileSync(FIG),
    transformation: { width: figWidthPx, height: Math.round(figWidthPx / 3) },
    altText: {
      title: "Figure 1",
      description: "Histograms of cosine similarity for same-person and different-people pairs, raw features and after LDA",
      name: "report_hist_barlow224",
    },
  })],
});

// --- Références ---------------------------------------------------------
const refsA = [
  "[1] Turk, M. & Pentland, A. (1991). Eigenfaces for Recognition. Journal of Cognitive Neuroscience, 3(1), 71–86. doi:10.1162/jocn.1991.3.1.71",
  "[2] Belhumeur, P., Hespanha, J. & Kriegman, D. (1997). Eigenfaces vs. Fisherfaces. IEEE TPAMI, 19(7), 711–720. doi:10.1109/34.598228",
  "[3] Ahonen, T., Hadid, A. & Pietikäinen, M. (2006). Face Description with Local Binary Patterns. IEEE TPAMI, 28(12), 2037–2041.",
  "[11] Musgrave, K., Belongie, S. & Lim, S.-N. (2020). A Metric Learning Reality Check. ECCV 2020. arXiv:2003.08505",
  "[12] Wang, T. & Isola, P. (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. ICML 2020. arXiv:2005.10242",
  "[14] Liu, Z., Luo, P., Wang, X. & Tang, X. (2015). Deep Learning Face Attributes in the Wild. ICCV 2015.",
  "[15] Zbontar, J. et al. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. ICML 2021. arXiv:2103.03230",
];
const refsBlock = REFS === "A"
  ? [heading("References"),
     para("Numbers follow our annotated bibliography.", { size: SMALL, italics: true, after: 40 }),
     ...refsA.map((r) => para(r, { size: SMALL, align: AlignmentType.LEFT, after: 20 }))]
  : [para("Reference numbers follow our annotated bibliography.", { size: SMALL, italics: true })];

// --- Document ----------------------------------------------------------
const children = [
  new Paragraph({
    children: [new TextRun({ text: "Progress report", bold: true, size: 26 })],
    spacing: { after: 40 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Topic 09: Face recognition algorithms, before and after deep learning · MINF1-SRD", size: SMALL })],
    spacing: { after: 20 },
  }),
  new Paragraph({
    children: [
      new TextRun({ text: "[Group members]", size: SMALL, shading: { type: ShadingType.CLEAR, fill: "FFF2A8" } }),
      new TextRun({ text: " · Coach: Stéphane Ayache · ", size: SMALL }),
      new TextRun({ text: "[Date]", size: SMALL, shading: { type: ShadingType.CLEAR, fill: "FFF2A8" } }),
    ],
    spacing: { after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "BFBFBF", space: 4 } },
  }),

  heading("1. Context and research question"),
  para("A face recognition system maps each face image to a vector and decides whether two images show the same person by comparing their vectors. Over the history of the field, the way this vector is obtained has changed, from hand-designed descriptors to representations learned with a training objective, and many objectives have been proposed for learning it. We focus on this last ingredient and ask: **what does the training objective do to the geometry of identity representations, and does it help recognise people never seen during training?** To isolate the effect of the objective, we compare four losses, namely normalised softmax, contrastive, triplet and ArcFace, trained on the same frozen pre-trained backbone with identical data, splits, preprocessing and evaluation. We study only the description step, which turns an image into a vector, on the aligned face crops provided with the dataset; face detection and alignment are out of scope."),

  heading("2. Coach's guidance and how we applied it"),
  para("Our coach, Stéphane Ayache, advised us to start from a pre-trained ResNet, to train a linear head on top of it with each of the chosen losses, and to keep every other factor constant so that the comparison stays fair. We applied this advice as follows. The ResNet-50 backbone stays frozen, and only a single linear layer, mapping its 2,048-dimensional output to a 128-dimensional embedding, is trained. Image vectors are extracted once and shared by all experiments, so every loss sees exactly the same data, splits and preprocessing. Each loss receives the same tuning budget of ten configurations, drawn from ranges fixed in advance and compared on validation identities only. The selected configuration of each loss will then be trained with the same five random seeds, and every result will come with a confidence interval. Test identities are never used for training or tuning, and the test set will be evaluated only once, at the end. To make these rules precise, we drew on the evaluation practices of Musgrave et al. [11], who showed that unequal protocols can create apparent differences between losses."),

  heading("3. Experimental setup"),
  para("We use CelebA [14] with its official partition, which we verified to be disjoint in identities, and keep only identities with at least four images so that every loss sees the same data. This leaves 7,689 training, 924 validation and 951 test identities, with 161,602, 19,720 and 19,850 images respectively. Each aligned image is cropped to a 128-pixel square that keeps the face from forehead to chin and removes most of the hair and background, then resized to 112 or 224 pixels. We compare two ResNet-50 backbones pre-trained on ImageNet: the supervised torchvision model and the self-supervised Barlow Twins model [15]. They share the same architecture and training images and differ only in how they were pre-trained. Normalised softmax is the softmax loss with L2-normalised weights and embeddings, that is ArcFace without margin, so that all four losses work on the same hypersphere and the gap between ArcFace and normalised softmax isolates the margin. Batches contain eight identities with four images each for the contrastive and triplet losses, and 32 identities for the two classification losses. Retrieval is measured with MAP@R, our main metric, and precision at one; verification with the ROC curve and the true acceptance rate at fixed false acceptance rates."),

  heading("4. Progress and preliminary results"),
  para("Before training any loss, we measured two references on validation identities: the raw backbone features, reduced to 128 dimensions by PCA to match the head, and a linear discriminant analysis fitted on training identities, which shows how far a linear head can go. Our metrics match the reference implementation in pytorch-metric-learning exactly, and the selection rule was fixed before these results were computed."),
  para("As shown in Table 1, raw features rank identities poorly, with a MAP@R of 4% at best, but LDA more than triples it: identity information is present but badly organised, which leaves a margin of 9.2 points for the losses. Figure 1 shows how LDA concentrates the similarities of different people around zero. With the same architecture and training images, Barlow Twins preserves identity better than the supervised model, by 3.8 MAP@R points with a 95% bootstrap confidence interval of 3.4 to 4.2, so we selected it at 224 pixels. A visual check of nearest neighbours shows that raw neighbours mostly share pose, hair or lighting, while LDA neighbours are more often the same person."),
  caption("Table 1. ", "Results on validation identities, in %. TAR: true acceptance rate at a false acceptance rate of 1%."),
  table,
  figure,
  caption("Figure 1. ", "Cosine similarity of same-person and different-people pairs on validation identities, Barlow Twins at 224 pixels, before and after LDA."),

  heading("5. Next steps"),
  para("We will now train the linear head with each of the four losses on the Barlow Twins features. Each loss receives ten configurations, the best one is retrained with five seeds, and losses are compared seed by seed. Beyond accuracy, we will compare the geometry each loss produces, using similarity histograms and the alignment and uniformity measures of Wang and Isola [12]. Test identities will be evaluated once, at the end. We will also run the classical baselines of our bibliography, namely Eigenfaces [1], Fisherfaces [2] and local binary patterns [3], on the same splits, to place learned representations next to hand-designed ones. Robustness to image degradations is planned as an optional extension."),
  para("Two points will guide how we read these results. Because the backbone is frozen, we measure what each loss does as a linear reorganisation of fixed features, not how it would shape a network trained end to end. And the two backbones differ in their whole pre-training recipe, including data augmentation, so the gap between them cannot be attributed to the objective alone."),

  ...refsBlock,
];

const doc = new Document({
  creator: "SRD sujet 09",
  title: "Progress report, topic 09",
  styles: { default: { document: { run: { font: "Calibri", size: BODY } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 737, right: 850, bottom: 737, left: 850 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("écrit :", OUT, "| largeur utile", CONTENT_W, "DXA");
});
