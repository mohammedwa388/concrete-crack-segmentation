# Concrete Crack Segmentation — CSE281 Course Project

**Team Crack Hunters** · Digital Image Processing (CSE281) · Alamein International University

A classical image-processing pipeline (no machine learning) that takes a raw photo of a
concrete surface, marks the cracked pixels, measures the cracked area and decides whether
the wall needs inspection.

> Status: **Week 1 — foundation in progress.** Sections marked *TBD* are filled in as the
> project advances.

---

## 1. Problem statement

The user is a building inspector who photographs concrete walls and needs to know where the
cracks are and how large they are. The pipeline takes a raw colour photo of a concrete
surface and outputs a binary crack mask, the crack area as a percentage of the image, and a
decision: "needs inspection" or "no action". A missed crack (false negative) is the costly
error, because a dangerous wall may be left unchecked. A false alarm only costs an extra
site visit.

### Input, output and metric

| Item | Definition |
|---|---|
| Input | RGB photo, `uint8` 0–255, converted to grayscale and resized (size *TBD* from the resolution study) |
| Output | Binary crack mask (0/1, same size as the input), crack area (%), decision |
| Primary metric | **IoU** = \|A ∩ B\| / \|A ∪ B\| between predicted and ground-truth crack pixels |
| Secondary metrics | Crack recall (share of true crack pixels found), absolute area error (%) |

**Why IoU:** cracks cover a small fraction of each image, so pixel accuracy is misleading
(predicting "no crack" everywhere would still score above 90 %). IoU ignores the true
background and only rewards overlap with the real crack.

## 2. Dataset

**Concrete Crack Segmentation Dataset** — 458 high-resolution images (4032 × 3024) of
concrete surfaces from buildings at Middle East Technical University, each with a
black-and-white mask marking the crack pixels.

- Source: <https://data.mendeley.com/datasets/jwsn7tfbrp/1>
- Licence: CC BY 4.0
- Citation: Özgenel, Ç. F. (2019). *Concrete Crack Segmentation Dataset*, Version 1.
  Mendeley Data. DOI: 10.17632/jwsn7tfbrp.1

Raw data is **not** stored in this repository. See [`data/README.md`](data/README.md) for
download instructions.

## 3. Team

| Member | Role | Work package | Main files |
|---|---|---|---|
| Mohamed Waly | Tech Lead + Data & Baselines | Problem formulation, repo, integration, loading, split | `src/data.py`, `configs/`, `README.md` |
| Ahmed Ashraf | Enhancement Engineer | Histograms, point transforms, smoothing & sharpening | `src/enhancement.py` |
| Mohamed Hosny | Restoration & Frequency Engineer | Noise models, restoration filters, DFT filters | `src/restoration.py`, `src/frequency.py` |
| Yahia | Segmentation Engineer | Edges, thresholding, baselines | `src/segmentation.py`, `src/pipeline.py` |
| Mahmoud | Morphology & Evaluation Engineer | Morphology, measurement, metrics, evaluation, demo | `src/morphology.py`, `src/metrics.py`, `src/evaluate.py` |

How we work together: see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 4. Repository structure

```
concrete-crack-segmentation/
├── README.md            problem, results table, setup and run instructions
├── CONTRIBUTING.md      branches, reviews, coding rules
├── requirements.txt     pinned package versions
├── configs/             one file per pipeline configuration
├── data/                download instructions (no raw data in git)
├── notebooks/           01_image_analysis.ipynb, 02_results.ipynb
├── src/
│   ├── data.py          loading, splitting, controlled degradation
│   ├── metrics.py       MSE, PSNR, IoU, recall, area error
│   ├── enhancement.py   point operations, histograms, spatial filters
│   ├── restoration.py   noise models and restoration filters
│   ├── frequency.py     DFT, spectrum, frequency-domain filters
│   ├── segmentation.py  edge detection, thresholding, Hough transform
│   ├── morphology.py    morphological operations, hole filling, measurement
│   ├── pipeline.py      baselines and Pipelines A, B, C
│   ├── evaluate.py      test-set metrics and failure analysis
│   └── run_pipeline.py  full pipeline on a raw image
├── tests/               unit tests (run automatically on every push)
├── experiments/         experiment log (CSV) and saved figures
├── app/                 demo application
└── report/              report PDF and slides
```

## 5. Setup

```bash
git clone https://github.com/mohammedwa388/concrete-crack-segmentation.git
cd concrete-crack-segmentation
pip install -r requirements.txt
```

Then download the dataset as described in [`data/README.md`](data/README.md) and set the
data path in `configs/default.yaml` (or with the `CRACK_DATA_DIR` environment variable).

### Run the tests

```bash
pytest
ruff check .
```
i am inivtable

## 6. Results

| Pipeline | IoU (mean ± std) | Recall | Area error | Time / image |
|---|---|---|---|---|
| Trivial baseline (threshold 127) | *TBD* | *TBD* | *TBD* | *TBD* |
| Simple baseline (Otsu + 3×3 opening) | *TBD* | *TBD* | *TBD* | *TBD* |
| Pipeline A | *TBD* | *TBD* | *TBD* | *TBD* |
| Pipeline B | *TBD* | *TBD* | *TBD* | *TBD* |
| Pipeline C | *TBD* | *TBD* | *TBD* | *TBD* |
