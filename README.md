<div align="center">
    <a href="https://github.com/liris-xr/kineo">
      <img alt="Kineo banner." src="docs/static/images/kineo.png">
    </a>
</div>
<p align="center">
    Charles Javerliat, Pierre Raimbaud, Guillaume Lavoué
    <br />
    <a href="https://liris-xr.github.io/kineo/"><strong>Project Page »</strong></a>&emsp;
    <a href="https://arxiv.org/abs/2510.24464"><strong>Paper »</strong></a>&emsp;
    <br />
</p>

Kineo is a calibration-free metric motion capture system that reconstructs 3D motion from sparse RGB cameras. It leverages existing 2D keypoints detectors to estimate 3D poses without requiring complex calibration procedures, making motion capture more accessible and flexible.

![StoneQuarry](docs/static/images/stone_quarry.gif)

## Installation

To install Kineo, please follow the procedure described in [INSTALL.md](./INSTALL.md).

## Inference

### Using Docker

If you installed Kineo using Docker, run the inference script directly through the docker container:
```sh
docker run --gpus all -it --rm \
  -v ./cache:/app/cache \
  -v ./outputs:/app/outputs \
  -v ./checkpoints:/app/checkpoints \
  -v ./body_models:/app/body_models/ \
  -v ./data:/app/data \
  kineo:latest python3 infer.py \
  --config-file configs/infer_nlf_single_person.yaml \
  --sequence-name MY_SEQUENCE \
  /app/data/view1.avi \
  /app/data/view2.avi \
  /app/data/view3.avi \
  /app/data/view4.avi \
  /app/data/view5.avi
```
This mounts your local folders (cache, outputs, checkpoints, body_models, data) inside the container, so the inference results and models persist outside the container.

### Using Manual Installation

If you installed Kineo manually (not using Docker), run the inference script directly in your Python environment:

```sh
python3 infer.py \
  --config-file configs/infer_nlf_single_person.yaml \
  --sequence-name MY_SEQUENCE \
  /app/data/view1.avi \
  /app/data/view2.avi \
  /app/data/view3.avi \
  /app/data/view4.avi \
  /app/data/view5.avi
```

## Evaluation

We evaluated Kineo on the EgoHumans and Human3.6M datasets. To reproduce our results, first download and preprocess the datasets using the provided scripts:
```sh
python scripts/download_h36m_dataset.py <path-to-h36m-dataset>
python scripts/preprocess_h36m_dataset.py <path-to-h36m-dataset>

python scripts/download_egohumans_dataset.py <path-to-egohumans-dataset>
python scripts/preprocess_egohumans_dataset.py <path-to-egohumans-dataset>
```

Once the datasets are prepared, you can run evaluation using the corresponding evaluation scripts:
```sh
python experiments/h36m_eval.py <path-to-h36m-dataset> configs/experiments/benchmarks/h36m_benchmark_nlf_estRt_estK_estD.yaml

python experiments/egohumans_eval.py <path-to-egohumans-dataset> configs/experiments/benchmarks/egohumans_benchmark_nlf_estRt_estK_estD.yaml
```
All configurations used in the paper are available in the `configs` directory.

## Acknowledgments

This work was supported by the Auvergne-Rhône-Alpes region as part of the PROMESS project. This work was granted access to the HPC resources of IDRIS under the allocation 2025-AD010614830 made by GENCI. We also express our gratitude to the Guédelon Castle for kindly welcoming us and permitting the captures that were essential to this study.

## BibTeX

```
@article{javerliat2025kineo,
  title={Kineo: Calibration-Free Metric Motion Capture From Sparse RGB Cameras}, 
  author={Charles Javerliat and Pierre Raimbaud and Guillaume Lavoué},
  year={2025},
  eprint={2510.24464},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2510.24464}, 
}
```