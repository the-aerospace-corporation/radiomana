# Radio Mana

*radiomana* is an open-source PyTorch library developed by The Aerospace Corporation for
**GNSS jamming detection and classification** using deep learning on radio frequency (RF) spectrum data.

## What it does

This library provides:
- **Neural network models** (HighwayBaselineModel,NanoGRU) for jamming classification
- **Data loading utilities** for the Fraunhofer GNSS Jamming Highway2 Dataset
- **Signal processing transforms** (noise augmentation, time cropping) optimized for RF data
- **Training pipelines** using PyTorch Lightning for jamming detection research

## Quick Start

1. Install

    ```bash
    # install in development mode
    pip install --editable .
    ```
2. Download the [Spectrum Highway Dataset 2](https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/fiot_highway2)
3. Set environment variable in your `.bashrc` file:
   ```bash
   export DSET_FIOT_HIGHWAY2=/path/to/highway2/dataset
   ```

## Usage Examples

### Basic Data Loading

```python
import radiomana

# inspect a sigle sample and label
dataset = radiomana.Highway2Dataset()
psd_sample, label = dataset[0]  # power spectral density + jamming classification

# inspect a training batch
datamodule = radiomana.HighwayDataModule(batch_size=32, num_workers=4)
datamodule.setup()
train_batch = next(iter(datamodule.train_dataloader()))
```

### Model Inference

```python
# create and train your own nano model
model = radiomana.NanoGRU(num_classes=9)

# or load from checkpoint after training
# model.load_state_dict(torch.load("path/to/your/trained_model.pt"))

# classify RF spectrum
with torch.no_grad():
    predictions = model(psd_sample.unsqueeze(0))  # add batch dimension
    jamming_class = predictions.argmax(dim=1)
```

### Training Your Own Model

```bash
# train baseline model (with resnet18 or mobilenet_v3_large submodel)
python examples/train_baseline.py

# train nanogru model
python examples/train_nano.py

# benchmark model inference speed
python examples/bench_model.py
```

## Model Performance

Performance benchmarks on the Highway2 GNSS jamming detection dataset. Models predict jamming type from RF power spectral density data.

| Model                | Submodel           | Augmentations | Params (M) | Memory (Mb) | multadds (G) | Test Loss | F1    | Acc% |
|----------------------|--------------------|---------------|------------|-------------|--------------|-----------|-------|------|
| HighwayBaselineModel | resnet18           | None          |       11.7 |          46 |        1.81  | 0.535     | 0.634 | 79.2 |
| HighwayBaselineModel | resnet18           | VFlip & Noise |       11.7 |          46 |        1.81  | 0.507     | 0.662 | 80.2 |
| HighwayBaselineModel | mobilenet_v3_large | None          |        5.5 |          12 |        0.22  | 0.617     | 0.582 | 75.6 |
| HighwayBaselineModel | mobilenet_v3_large | VFlip & Noise |        5.5 |          12 |        0.22  | 0.527     | 0.611 | 79.5 |

## Open Source

### Release

This project is approved for public release with unlimited distribution by Aerospace under OSS Project Ref #OSS25-0006.

### Contributing

Do you have code you would like to contribute to this Aerospace project?

We are excited to work with you. We are able to accept small changes
immediately and require a Contributor License Agreement (CLA) for larger
changesets. Generally documentation and other minor changes less than 10 lines
do not require a CLA. The Aerospace Corporation CLA is based on the well-known
[Harmony Agreements CLA](http://harmonyagreements.org/) created by Canonical,
and protects the rights of The Aerospace Corporation, our customers, and you as
the contributor. [You can find our CLA here](https://aerospace.org/sites/default/files/2020-12/Aerospace-CLA-2020final.pdf).

Please complete the CLA and send us the executed copy. Once a CLA is on file we
can accept pull requests on GitHub or GitLab. If you have any questions, please
e-mail us at [oss@aero.org](mailto:oss@aero.org).

### Licensing

The Aerospace Corporation supports Free & Open Source Software and we publish
our work with GPL-compatible licenses. If the license attached to the project
is not suitable for your needs, our projects are also available under an
alternative license. An alternative license can allow you to create proprietary
applications around Aerospace products without being required to meet the
obligations of the GPL. To inquire about an alternative license, please get in
touch with us at [oss@aero.org](mailto:oss@aero.org).
