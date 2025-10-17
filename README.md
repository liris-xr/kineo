# Kineo

Kineo is a toolkit for 3D human pose estimation. It is built on top of MMPretrain, MMEngine, MMCV, MMDet, and MMPose.

## Installation

### Minimal installation

```bash
conda create -n kineo python=3.10
conda activate kineo
```

To compile CUDA kernels, make sure to set the following environment variables:
```bash
export MMCV_WITH_OPS=1
export SAM2_BUILD_ALLOW_ERRORS=0
export SAM2_BUILD_CUDA=1
```

If you want to install in headless mode, make sure to set the `AITVIEWER_HEADLESS` environment variable to `1`.
```bash
export AITVIEWER_HEADLESS=1
```

When building MMCV with CUDA support, start by [installing torch](https://pytorch.org/get-started/locally/) in your environment, then install the package without build isolation so that the correct version of torch is used to build MMCV kernels. Otherwise you might encounter "undefined symbol" errors when running the pipeline.
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # <- Replace with the correct version of torch you want to use
pip install --no-build-isolation -v -e .
```

### Download the keypoints detector

If you use MMDet and MMPose, you can either specify the model name alias, for example:
```yaml
mmlab_bbox_keypoints_detection:
    _target_: kineo.pipeline.stages.mmlab_bbox_keypoints_detection.MMLabBboxKeypointsDetectionStage
      name: "MMLab Bbox Keypoints Detection"
    [...]
    det_model: "rtmdet-t"
    keypoints_model: "rtmpose-l_8xb256-420e_coco-256x192"
    [...]
```
In which case the model weights will be downloaded automatically. Otherwise, you can specify the model config and weights path manually, for example:
```yaml
mmlab_bbox_keypoints_detection:
    _target_: kineo.pipeline.stages.mmlab_bbox_keypoints_detection.MMLabBboxKeypointsDetectionStage
      name: "MMLab Bbox Keypoints Detection"
    [...]
    det_model: "rtmdet-t"
    det_model_weights: "./checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
    keypoints_model: "./configs/rtmpose-l_8xb256-420e_h36m-384x288.py"
    keypoints_model_weights: "./checkpoints/rtmpose_h36m.pth"
    [...]
```

### Download the body models

If you use SMPL global scaling, you need to download the SMPL model from [here](https://smpl.is.tue.mpg.de/) and place it in the `body_models/smpl` directory, along with the joint regressor which outputs the same joints format as the 2D detector.

### Download the MoGe model

If you use MoGe global scaling, you need to download the MoGe model from [here](https://huggingface.co/Ruicheng/moge-2-vitl-normal) and place it in the `checkpoints` directory after renaming it to `moge-2-vitl-normal.pt`.