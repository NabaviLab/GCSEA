# Gabor-Constrained Semantic Evidence Aggregation for Prostate Cancer Grading

Research implementation scaffold for weakly supervised, four-class prostate Grade Group prediction from WSIs.

## Main modules

- A: WSI → tissue patches → frozen ResNet50 patch features; independent frozen UNI patch features.
- B: Trainable UNI-based patch importance and Top-M weighted ResNet50 refinement.
- C: Frozen CONCH concept vectors imported from the text encoder; trainable ResNet→concept embedding projection; cosine evidence.
- D: Trainable directional Gabor filters on original selected RGB image patches, coordinate/Gabor-based spatial kNN graph.
- E: Row-stochastic adjacency and configurable concept-evidence propagation; patch attention, slide classifier.
- F: Four outputs: label 0 GG1-2, label 1 GG3, label 2 GG4, label 3 GG5.



## Layout

```
config.yaml                    Hyperparameters
concepts/concept_anchors.txt   Example anchors;
scripts/prepare_patches.py     OpenSlide tissue tiling and tissue filtering (baseline)
scripts/extract_features.py    ResNet50 and UNI feature extraction
scripts/export_conch_concepts.py  CONCH export (upstream API may vary)
scripts/check_data.py          Dataset validation
scripts/train.py               Supervised slide training / early stopping
scripts/evaluate.py            Slide-level metrics / CSV predictions
scripts/crossvalidate.py       Five-fold stratified CV (slide-level)
gsea/{data,model,gabor,graph,losses,metrics,util}.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
pytest -q
```

OpenSlide requires the corresponding operating-system OpenSlide shared library. Install official CONCH per its repository instructions and provide your licensed model checkpoint. UNI must be loaded from a compatible timm architecture and pretrained checkpoint; the `--uni-model` name must match the checkpoint. Standard ImageNet normalization in the example extractor may need replacing with model-specific UNI preprocessing.

## Inputs

First create `slides.csv` with `slide_id,wsi_path,label` and grades remapped as **GG1-2→0, GG3→1, GG4→2, GG5→3**.

```bash
python scripts/prepare_patches.py --slides slides.csv --out-dir data/patches --size 224
```

This is baseline tile extraction and NOT an exact CLAM/stain-normalization implementation. For the preprocessing, replace with your validated CLAM feature/patch workflow and appropriate stain normalization. A `patches.csv` must contain `slide_id,patch_path,x,y,label` with coordinates in level-0 WSI pixels.

```bash
python scripts/extract_features.py \
  --patch-csv data/patches/patches.csv \
  --uni-model YOUR_VALID_TIMM_UNI_ARCHITECTURE \
  --uni-checkpoint /secure/path/to/uni_checkpoint.pt \
  --out-dir data/bags
```

Feature files have keys `resnet: [N,2048]`, `uni: [N,D_uni]`, `patches: [N,3,224,224]`, `coords: [N,2]`. Patch order is identical across both encoders and raw images. Slide bag files can be large. Adjust max-patches memory cap and use cached embeddings.

Produce the *actual* concept list from your research and export using your verified CONCH installation:

```bash
python scripts/export_conch_concepts.py \
  --checkpoint /secure/path/to/conch_checkpoint.pt \
  --concepts concepts/concept_anchors.txt \
  --out concepts/conch_text_embeddings.pt
```

The exported `.pt` file contains `embeddings: [K,D_text]`. The export helper targets a commonly published CONCH API, not all forks, and may need to be adapted. Do not substitute random vectors.

## Train / evaluate

```bash
python scripts/check_data.py --manifest data/bags/manifest.csv
python scripts/train.py \
  --config config.yaml --train train.csv --val val.csv \
  --concept-embeddings concepts/conch_text_embeddings.pt --out runs/experiment01
python scripts/evaluate.py --checkpoint runs/experiment01/best.pt \
  --manifest test.csv --out runs/experiment01/test_predictions.csv
python scripts/crossvalidate.py --manifest data/bags/manifest.csv \
  --concept-embeddings concepts/conch_text_embeddings.pt --config config.yaml --out runs/cv
```

Use **patient-grouped**, stratified folds if multiple slides come from one patient. The included CV example only performs **slide-stratified** folds; it is not leakage-safe for multi-slide patients. Test sets must remain strictly independent of model selection. AUC is multiclass one-vs-rest; F1 is macro; CK defaults to quadratic-weighted Cohen's kappa. The exact metric conventions should match your experiments.

✅ Four-class slide grading, differentiable Gabor filters, IRM-style learned scoring, original-image Gabor inputs, graph evidence propagation, cosine concept similarity, weighted patch attention, focal+smoothing objective, checkpointing, validation, prediction CSV, five-fold CV script.
