**Experiment.** Reducing `imgsz` from 640 to 416 (all else fixed) changed validation mAP50 by +0.000 and mAP50-95 by +0.000: no meaningful difference in validation quality. Training took 1.74 min instead of 2.70 min (36% less, single run). The hypothesis predicted a small quality drop; on this validation set it was not observed, but 62 images and one seed cannot confirm or rule out a small effect.

**Selection.** Final model: `baseline_640` (tie within 0.005 on val mAP50 -> keep baseline).

**Test (held-out, evaluated once).** Precision 0.976, recall 0.982, mAP50 0.986, mAP50-95 0.954. Test-set numbers were not used to choose between the runs.

**Error analysis.** On all 60 test images: 59 correct at top-1 (conf >= 0.25); mean top-1 IoU 0.960. See section 4.3 for the hardest cases.
