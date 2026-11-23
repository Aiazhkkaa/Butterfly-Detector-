# 🦋 Butterfly Detector: YOLOv8n on the Leeds Butterfly Dataset

Detects a butterfly in a photo and classifies it into one of **5 species**, using a fine-tuned, pretrained **YOLOv8n**. The project also runs a controlled experiment comparing two input sizes (640 px vs 416 px).

*CSCI1409 Deep Learning, bonus project (KBTU)*

| | |
|---|---|
| **Notebook** | `butterfly_detector_improved.ipynb` (run top to bottom in Google Colab, T4 GPU) |
| **Weights** | `results/best.pt` (6.5 MB): <PASTE LINK: Google Drive "anyone with the link" or GitHub Release> |
| **Final model** | `baseline_640` (YOLOv8n, `imgsz=640`) |
| **Test result** | mAP50 **0.986**, mAP50-95 **0.954** (60 held-out images, evaluated once) |

---

## Table of contents
1. [Results at a glance](#results-at-a-glance)
2. [Task and data](#task-and-data)
3. [Method](#method)
4. [Experiment: 640 px vs 416 px](#experiment-640-px-vs-416-px)
5. [Final test results](#final-test-results)
6. [Error analysis](#error-analysis)
7. [Quick start](#quick-start)
8. [Reproduce the experiments](#reproduce-the-experiments)
9. [Repository contents](#repository-contents)
10. [Limitations](#limitations)
11. [Data and license notes](#data-and-license-notes)

---

## Results at a glance

- The model finds the butterfly and names the species with **precision 0.976 / recall 0.982** on the held-out test set.
- **416 px trains ~36% faster** (1.75 vs 2.74 min) with **no measurable quality loss on validation**. The validation set is too small to separate the two runs, so the pre-declared rule kept the 640 px baseline.
- On all 60 test images, **59 are correct at top-1** (conf ≥ 0.25). The one miss is a **wrong class with a well-placed box** (IoU 0.95): a *Vanessa cardui* predicted as *Vanessa atalanta* (see [Error analysis](#error-analysis)).

## Task and data

- **Task:** object detection with classification (one bounding box plus a species label per image).
- **Dataset:** [Leeds Butterfly Dataset](https://zenodo.org/records/7559420) (Zenodo record 7559420, archive MD5-verified in the notebook). We use **5 of the 10 species**, giving **410 images**.
- **Labels:** the dataset ships segmentation masks, not boxes. Boxes are derived from the masks (pixel values 1 and 3 = butterfly; min/max of the foreground pixels).

| Class | Species | Images |
|---|---|---|
| 0 | *Danaus plexippus* (monarch) | 82 |
| 1 | *Heliconius charitonius* (zebra longwing) | 93 |
| 2 | *Heliconius erato* (red postman) | 61 |
| 3 | *Vanessa atalanta* (red admiral) | 90 |
| 4 | *Vanessa cardui* (painted lady) | 84 |

**Split (stratified by class, seed 42):** train **288** / val **62** / test **60** (≈ 70 / 15 / 15).
Near-duplicate images are found with perceptual hashing (pHash, Hamming distance ≤ 4; 5 pairs found) and **kept in the same split** to reduce train/test leakage. Per-split class counts are in `results/dataset_stats.csv`.

## Method

| Item | Value |
|---|---|
| Model | YOLOv8n (3.0 M parameters, 8.2 GFLOPs), COCO-pretrained `yolov8n.pt`, head replaced for 5 classes |
| Framework | Ultralytics 8.4.216, PyTorch 2.14.1 (full list in `requirements.txt`) |
| Training | 30 epochs, batch 16, seed 42, AMP on, default Ultralytics augmentations (mosaic closed for the last 10 epochs), early-stopping patience 10 (not triggered; both runs completed 30 epochs) |
| Optimizer | `optimizer=auto`: Ultralytics selected **AdamW (lr ≈ 0.0011)** and ignored the `lr0=0.01` argument. Both runs used the same optimizer settings. |
| Evaluation | `conf=0.001`, NMS `iou=0.6` for every val/test call, so all numbers are comparable |
| Hardware | Google Colab, Tesla T4 |

**Model selection rule (declared in advance):** pick the run with the highest **validation mAP50**. Differences below **0.005** count as a tie, and on a tie the 640 px baseline is kept. The **test set is not used for selection** and is evaluated exactly once, on the chosen model.

## Experiment: 640 px vs 416 px

**Hypothesis:** a smaller input (`imgsz=416`) trains faster, possibly with a small quality drop because fine wing patterns become less visible.
**Control:** only `imgsz` changes. Model, data split, seed, epochs, batch size and optimizer settings are identical.

| Run | imgsz | Val precision | Val recall | Val mAP50 | Val mAP50-95 | Train time (min) | Epochs |
|---|---|---|---|---|---|---|---|
| `baseline_640` | 640 | ≈ 0.99 | ≈ 1.00 | 0.995 | 0.959 | 2.74 | 30 |
| `experiment_416` | 416 | ≈ 0.99 | ≈ 1.00 | 0.995 | 0.960 | 1.75 | 30 |

**Conclusion.** Validation mAP50 is identical (0.995) and mAP50-95 differs by only +0.001, so the hypothesised quality drop was **not observed**. Training was **~36% faster** at 416 px. With only 62 validation images and one seed, a small real difference can be neither confirmed nor ruled out. mAP50 is saturated near its ceiling, so this comparison cannot separate the runs. The rule therefore kept the baseline.

> Exact values (including precision and recall) are in `results/validation_comparison.csv`. If you re-run the notebook, **trust the CSV files over the tables here**, since timings in particular vary between runs on a shared GPU.

Plots: `results/validation_comparison.png`, `results/training_time_comparison.png`, `results/training_curves_baseline_640.png`, `results/training_curves_experiment_416.png`.

## Final test results

Selected model `baseline_640`, evaluated once on the 60 held-out test images:

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.976 | 0.982 | 0.986 | 0.954 |

Per-class numbers: `results/test_per_class.csv`. Further plots: `results/test_confusion_matrix.png`, `results/test_confusion_matrix_normalized.png`, `results/test_PR_curve.png`.

**Qualitative check:** on five random test images the predicted box overlaps the ground-truth box well (IoU 0.89 to 0.98) and the class is correct (`results/test_predictions.png`). Green boxes are ground truth.

## Error analysis

Every test image was checked (`results/test_per_image_analysis.csv`): the top-confidence box at conf ≥ 0.25 is compared with the ground truth for both class and IoU.

- **59 of 60 images are correct** at top-1.
- **The one error is image `0100029.png`:** ground truth *Vanessa cardui*, predicted *Vanessa atalanta* (confidence 0.84). The box itself is good (IoU 0.95), so this is a **classification** error, not a localization one.
- **Likely cause (hypothesis, not tested):** the two *Vanessa* species share a dark-brown, orange-banded wing pattern, and this photo shows an insect that looks very much like a red admiral (the caption inside the photo reads "Indian Red Admiral"). This may be a look-alike or a label-noise case.
- **Loosest boxes among correct predictions** (IoU ≈ 0.89): `0090132.png` and `0010076.png`. The predicted boxes are slightly wider or shifted compared with the mask-derived ground truth.

Figure of the hardest cases: `results/hardest_test_cases.png`.

**Suggested next step:** check `0100029.png` and similar *V. cardui* images for label noise, then collect or add more *Vanessa* examples with varied backgrounds, and re-test with a second seed and a larger test set.

## Quick start

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("results/best.pt")          # download the weights first (link at the top)
results = model.predict("butterfly.jpg", imgsz=640, conf=0.25)

for box in results[0].boxes:
    species = model.names[int(box.cls)]
    print(f"{species}: {float(box.conf):.2f}, box (xyxy) = {box.xyxy[0].tolist()}")

results[0].save("prediction.jpg")        # image with the box drawn
```

Class names: `danaus_plexippus`, `heliconius_charitonius`, `heliconius_erato`, `vanessa_atalanta`, `vanessa_cardui`.
Inference speed on a T4 at 640 px is about 2 ms per image (see the validation log in the notebook).

> The model was trained on single-butterfly photos of these 5 species only. On other species it will still output one of the 5 labels.

## Reproduce the experiments

1. Open the notebook in Google Colab and choose **Runtime → Change runtime type → T4 GPU**.
2. **Runtime → Run all.** The dataset (≈ 476 MB) is loaded from Google Drive `leeds.zip` if present, otherwise downloaded from Zenodo and MD5-checked.
3. All outputs are written to `results/`. Library versions are saved in `requirements.txt`. Total run time is roughly 10 minutes, including the download.

**Reproducibility note:** all seeds are fixed to 42, but images are cached in RAM (`cache=True`), which Ultralytics warns can make training slightly non-deterministic. Expect small run-to-run differences in the metrics and the timings.

## Repository contents

```
├── butterfly_detector_improved.ipynb   # full pipeline: data → train → compare → test
├── requirements.txt                    # pinned library versions
├── README.md
└── results/
    ├── best.pt                         # final model weights
    ├── dataset_stats.csv               # images per class and split
    ├── validation_comparison.csv/.png  # 640 vs 416 on validation
    ├── training_time_comparison.png
    ├── training_curves_*.png           # loss and metric curves per run
    ├── training_runs.json              # time, epochs, best epoch per run
    ├── selected_model.json             # which run was chosen and why
    ├── test_results.csv                # final test metrics
    ├── test_per_class.csv
    ├── test_confusion_matrix*.png, test_PR_curve.png
    ├── test_predictions.png            # sample predictions vs ground truth
    ├── test_per_image_analysis.csv     # outcome for every test image
    ├── hardest_test_cases.png
    ├── ground_truth_examples.png
    └── conclusion.md                   # auto-generated summary
```

## Limitations

- **Boxes come from masks** (pixel values 1 and 3) and can be imperfect.
- **Easy task:** only 5 of 10 species, one butterfly per image, mostly clear views. The results should not be generalised to cluttered scenes or other species.
- **Small val/test sets** (62 and 60 images, 9 to 14 per class). Validation mAP50 saturates at ≈ 0.995, so the two experiments cannot be told apart on it.
- **One seed per run**, so small metric differences may be training variability.
- **Timings** come from a single run on a shared Colab T4 and include caching and CUDA warm-up. Only a large gap is meaningful.
- **Near-duplicate detection** (pHash ≤ 4) found only 5 pairs and may miss others, so some leakage is possible.
- **Possible label noise:** at least one test image (`0100029.png`) looks ambiguous between two *Vanessa* species.

## Data and license notes

Dataset: Leeds Butterfly Dataset, Zenodo record 7559420. The images were collected from Google Images. **Check the dataset terms on Zenodo before redistributing the images.** This repository does not include the images, and the notebook downloads them from the original source.
