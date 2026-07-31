# Exercise 3 checkpoints

This directory contains the canonical classification checkpoint used to
initialize the ResNet-50 body of Faster R-CNN runs B, C and D.

Expected generated files:

```text
gtsrb_resnet50_full_linear.pt
gtsrb_resnet50_full_linear.json
training_runs/
```

The `.pt` file and the `training_runs/` directory are generated locally or on
the server and must not be committed to Git. The JSON file records the source
run, validation-based checkpoint selection, Git commit and file checksum.

Create or validate the canonical checkpoint with:

```bash
python -m Exercise3.main prepare-backbone
python -m Exercise3.main prepare-backbone --validate-only
```
