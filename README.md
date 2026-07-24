Live Demo: https://image-forgery-detection-9ukjkmguvtksnil5vfcuz3.streamlit.app/
# Image Forgery / Tamper Detection

A binary image classifier that detects whether an image has been digitally
manipulated (spliced, copy-moved, or otherwise tampered) vs. an authentic,
untouched image. Relevant to identity verification and document fraud use
cases, where detecting whether an uploaded ID, receipt, or photo has been
tampered with is a core problem.

## Problem Statement

Given an input image, predict:
- `0` = Authentic (untouched)
- `1` = Tampered (spliced / copy-moved / edited)

Framed as binary image classification, fine-tuned on top of a pretrained
backbone rather than trained from scratch, since forgery datasets are small
relative to what's needed to train a CNN from zero.

## Dataset

**CASIA v2** — ~7,400 authentic images + ~5,100 tampered images. Tampered
images include splicing (pasting a region from another image) and copy-move
forgery (copying/moving a region within the same image).

## Approach

- Fine-tuned a pretrained **ResNet-18** (ImageNet weights) with its final
  layer replaced for 2-class output.
- Backbone frozen initially, training only the new classification head —
  faster and less prone to overfitting on a relatively small dataset.
- Partway through training, unfroze the last two backbone layers and
  continued fine-tuning at a lower learning rate, letting the model adapt
  its features specifically to forgery artifacts.
- Mild data augmentation only (flips, small rotations, slight color
  jitter) — aggressive augmentation risks destroying the subtle
  lighting/compression artifacts the model needs to detect.
- Data split into train / validation / test (70/15/15), with the best
  checkpoint selected by validation accuracy.

## Tech Stack

Python, PyTorch, torchvision, scikit-learn (metrics), PIL.

## Results

Evaluated on a held-out test set of 224 images (122 authentic, 102 tampered),
never seen during training.

| Metric | Value |
|---|---|
| Accuracy | 0.8884 |
| Precision (Tampered) | 0.9231 |
| Recall (Tampered) | 0.8235 |
| F1 (Tampered) | 0.8705 |
| Precision (Authentic) | 0.8647 |
| Recall (Authentic) | 0.9426 |
| F1 (Authentic) | 0.9020 |

**Confusion matrix:**

|  | Predicted Authentic | Predicted Tampered |
|---|---|---|
| **True Authentic** | 115 | 7 |
| **True Tampered** | 18 | 84 |

**Error analysis:** Precision on the tampered class (0.92) is notably
higher than recall (0.82) — when the model flags an image as tampered it's
usually right, but it misses roughly 1 in 5 actual forgeries (18 of 102),
classifying them as authentic. In a real fraud-detection setting this is
the costlier failure mode, since a missed forgery is generally worse than
a false alarm on a genuine document. The authentic class shows the
opposite pattern (recall 0.94, precision 0.86) — the model rarely misses a
genuine image, but produces some false positives (7 cases), likely
flagging authentic images with unusual lighting or compression as
tampered.

## Notes / Next Steps

- Try a stronger backbone (EfficientNet, ViT) and compare.
- Add Error Level Analysis (ELA) as an additional input channel, since
  forgery detection often benefits from compression-artifact features, not
  just raw pixels.
- Test robustness against re-compressed/re-saved images.
- Expand to localization (predicting which pixels were tampered, not just
  whether the image was tampered).
- CASIA v2 is a relatively clean academic benchmark — real-world tampered
  documents look different, so this is a proof of concept of the
  fine-tuning + evaluation workflow rather than a production-ready detector.
