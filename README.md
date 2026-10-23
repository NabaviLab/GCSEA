# Gabor-Constrained Semantic Evidence Aggregation for Prostate Cancer Grading

Research implementation scaffold for weakly supervised, four-class prostate Grade Group prediction from WSIs.

**Status: functional PyTorch reference pipeline, NOT a validated reproduction of the paper's numerical results.** All slide labels, concept text, UNI and CONCH checkpoints, stain processing, exact loss definition, architecture initialization and clinical split must be verified against the authors' experiment records prior to declaring a reproducible release. The 16 entries in `concepts/concept_anchors.txt` are **illustrative placeholders**, not asserted to be the exact Gemini-generated prompts from the study.

## Main modules

- A: WSI → tissue patches → frozen ResNet50 patch features; independent frozen UNI patch features.
- B: Trainable UNI-based patch importance and Top-M weighted ResNet50 refinement.
- C: Frozen CONCH concept vectors imported from the **authentic** text encoder; trainable ResNet→concept embedding projection; cosine evidence.
- D: Trainable directional Gabor filters on **original selected RGB image patches**, coordinate/Gabor-based spatial kNN graph.
- E: Row-stochastic adjacency and configurable concept-evidence propagation; patch attention, slide classifier.
- F: Four outputs: label 0 GG1-2, label 1 GG3, label 2 GG4, label 3 GG5.

**Critical distinctions:** `gsea/losses.py` implements **standard focal loss**, not a mathematically ordinal focal loss (the manuscript term requires the actual ordinal-aware loss definition). CONCH/UNI weights are **not bundled**; do not report fake results. ResNet features are projected into CONCH text space with slide classification supervision; this alone does not prove clinical concept alignment. Gabor parameter initialization is one configurable reference choice, not a claimed implementation fact. This is a useful baseline suitable for further implementation reconciliation.

## Layout

```
config.yaml                    Hyperparameters
concepts/concept_anchors.txt   Example anchors; REPLACE with actual study anchors
scripts/prepare_patches.py     OpenSlide tissue tiling and tissue filtering (baseline, not CLAM)
scripts/extract_features.py    Authentic ResNet50 and UNI feature extraction
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

OpenSlide requires the corresponding operating-system OpenSlide shared library. Install official CONCH per its repository instructions and provide your licensed model checkpoint. UNI must be loaded from a compatible timm architecture and **real** pretrained checkpoint; the `--uni-model` name must match the checkpoint. Standard ImageNet normalization in the example extractor may need replacing with model-specific UNI preprocessing.

## Inputs

First create `slides.csv` with `slide_id,wsi_path,label` and grades remapped as **GG1-2→0, GG3→1, GG4→2, GG5→3**. Confirm patient-level uniqueness and grade harmonization yourself.

```bash
python scripts/prepare_patches.py --slides slides.csv --out-dir data/patches --size 224
```

This is **simple tile extraction** and NOT an exact CLAM/stain-normalization implementation. For paper-faithful preprocessing, replace with your validated CLAM feature/patch workflow and appropriate stain normalization. A `patches.csv` must contain `slide_id,patch_path,x,y,label` with coordinates in level-0 WSI pixels.

```bash
python scripts/extract_features.py \
  --patch-csv data/patches/patches.csv \
  --uni-model YOUR_VALID_TIMM_UNI_ARCHITECTURE \
  --uni-checkpoint /secure/path/to/uni_checkpoint.pt \
  --out-dir data/bags
```

Feature files have keys `resnet: [N,2048]`, `uni: [N,D_uni]`, `patches: [N,3,224,224]`, `coords: [N,2]`. Patch order is identical across both encoders and raw images. **Slide bag files can be large.** Adjust max-patches memory cap and use cached embeddings.

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

## What is and is not implemented

✅ Four-class slide grading, differentiable Gabor filters, IRM-style learned scoring, original-image Gabor inputs, graph evidence propagation, cosine concept similarity, weighted patch attention, focal+smoothing objective, checkpointing, validation, prediction CSV, five-fold CV script.

⚠️ Not paper-exact: IRM is a **single scoring/Top-M selection**, not a genuinely iterative algorithm; graph kNN is directed with row-normalized adjacency; attention is **patch attention per concept vector**, not independently estimated per-concept attention; loss is **not ordinal-aware**; no verified Gemini anchor prompt, pathologist validation, trained image/text contrastive alignment, exact CLAM stain normalization, reliable patient-grouped splitting, or independent statistical significance tests are provided. Avoid presenting these substitutions as the paper's original experiment until reconciled.

## Reproducibility checklist

1. Insert and release the exact 16 anchors, Gemini prompt, concept encoder ID and checkpoint hash.
2. Verify whether IRM is iterative or single-stage, and whether ResNet50 is frozen.
3. Replace the focal loss with the **actual** ordinal-aware objective if one was used.
4. Verify the actual concept aggregation rule and image-to-text alignment strategy.
5. Document the Gabor initialization, parameter constraints, kNN symmetrization, coordinate scale and parameter/FLOP accounting.
6. Match cohort-specific CLAM preprocessing, patient-level splits, seed control and actual AUC/F1/CK definitions.
7. Run sensitivity tests for Top-M values 16,32,64,128; evaluate concept prompts; repeat baselines and statistical tests.

## Citation

Please cite the eventual published paper once bibliographic information is available. This repository does not include private slides or restricted model checkpoints.
