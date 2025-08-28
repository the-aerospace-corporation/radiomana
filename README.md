# Radio Mana

*radiomana* is an open-source PyTorch library developed by The Aerospace Corporation for manipulating radio signals.

## Installation

```bash
pip install --editable .
```

- Download the [Spectrum Highway Dataset 2](https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/fiot_highway2), and set the `DSET_FIOT_HIGHWAY2` environment variable your .bashrc file to point to the location of the dataset.

## Usage Example

### Loading the Highway2 Dataset

```python
import radiomana

# inspect a single item
dset = radiomana.Highway2Dataset()
some_psd, some_label = dset[0]

# inspect a whole batch from loader
dmodule = radiomana.HighwayDataModule()
dmodule.setup()
some_batch = next(iter(dmodule.train_datamodule()))
```

### Training Example

```bash
./examples/train_baseline.py
```

### Highway2 Model Performance

With provided basic models and augmentations, we achieve the following performance.
Observe that when un-augmented we overfit rapidly to the training set and our model doesn't generalize well.

| Model                | Submodel           | Augmentations | # Params (M) | Memory (Mb) |  GFlops | Test Loss | F1    | Acc% |
|----------------------|--------------------|---------------|--------------|-------------|---------|-----------|-------|------|
| HighwayBaselineModel | resnet18           | None          |         11.7 |          46 |   1.81  | 0.535     | 0.634 | 79.2 |
| HighwayBaselineModel | resnet18           | VerticalFlip  |         11.7 |          46 |   1.81  | 0.506     | 0.694 | 80.1 |
| HighwayBaselineModel | resnet18           | Noise @ -90dB |         11.7 |          46 |   1.81  | 0.521     | 0.633 | 79.6 |
| HighwayBaselineModel | resnet18           | VFlip & Noise |         11.7 |          46 |   1.81  | 0.507     | 0.662 | 80.2 |
| HighwayBaselineModel | mobilenet_v3_large | None          |          5.5 |          12 |   0.22  | 0.617     | 0.582 | 75.6 |
| HighwayBaselineModel | mobilenet_v3_large | VerticalFlip  |          5.5 |          12 |   0.22  | 0.578     | 0.625 | 77.2 |
| HighwayBaselineModel | mobilenet_v3_large | Noise @ -90dB |          5.5 |          12 |   0.22  | 0.593     | 0.607 | 78.4 |
| HighwayBaselineModel | mobilenet_v3_large | VFlip & Noise |          5.5 |          12 |   0.22  | 0.527     | 0.611 | 79.5 |
