# Image Forgery Detection Classifier

An end-to-end ML project that fine-tunes a ResNet-18 to detect digitally
tampered images (spliced or copy-moved) versus authentic ones, then ships
the model as a live interactive demo — simulating a real-world identity
verification / document fraud detection pipeline.

## Problem

Platforms that verify identity documents, receipts, or user-uploaded photos
can't manually inspect every image for tampering. This project builds a
classifier that takes an image the moment it's uploaded and predicts
whether it has been digitally manipulated, flagging likely forgeries for
closer review.

## Approach

1. **Data exploration** — analyzed the CASIA v2 dataset (~7,400 authentic +
   ~5,100 tampered images), checked class balance and image formats before
   touching any model.
2. **Preprocessing** — resized images to 224×224, applied ImageNet
   normalization, and used mild augmentation (flips, small rotations, slight
   color jitter) — kept deliberately light since aggressive augmentation
   can destroy the subtle lighting/compression artifacts forgery detection
   depends on.
3. **Fine-tuning** — fine-tuned a pretrained ResNet-18 as a binary
   classifier, starting with the backbone frozen and only training the new
   classification head, then unfreezing the last two backbone layers at a
   lower learning rate for further adaptation.
4. **Evaluation** — measured per-class precision, recall, and F1 alongside
   a full confusion matrix, since precision and recall diverge in
   meaningfully different directions for this task.
5. **Deployment** — shipped the trained model as a live Streamlit app, with
   the checkpoint pulled from Google Drive at runtime to keep the repo
   lightweight.

## Results

- **Accuracy: 0.8884** on a held-out test set of 224 images (122 authentic,
  102 tampered)
- **Tampered class:** precision 0.9231, recall 0.8235, F1 0.8705
- **Authentic class:** precision 0.8647, recall 0.9426, F1 0.9020
- Precision and recall diverge in opposite directions per class — the
  model rarely misses genuine images (recall 0.94) but produces some false
  positives (7 cases), while for tampered images it's usually right when it
  flags one (precision 0.92) but misses roughly 1 in 5 actual forgeries
  (18 of 102) — the costlier failure mode in a fraud-detection context
- Manually tested with additional images outside the CASIA v2 test set to
  sanity-check generalization beyond the benchmark

## Dataset

[CASIA v2 Image Tampering Detection Dataset](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)
(Kaggle) — ~12,500 images labeled as authentic or tampered (splicing /
copy-move forgery).

## Tech Stack

Python, PyTorch, torchvision, scikit-learn, PIL, Streamlit, Google Colab

## Live Demo

🔗 [Try it here](https://image-forgery-detection-9ukjkmguvtksnil5vfcuz3.streamlit.app)

## Notes / Next Steps

- Try a stronger backbone (EfficientNet, ViT) and compare against ResNet-18
- Add Error Level Analysis (ELA) as an additional input channel, since
  forgery detection often benefits from compression-artifact features, not
  just raw pixels
- Test robustness against re-compressed/re-saved images, a common
  real-world evasion tactic
- Expand to localization (predicting which pixels were tampered, not just
  whether the image was tampered)
- CASIA v2 is a relatively clean academic benchmark — real-world tampered
  documents look different, so this is a proof of concept of the
  fine-tuning + evaluation workflow rather than a production-ready detector
