from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

##---------------------------------------
_C = CN()
_C.SEQUENCE = "001_tagging"
_C.INVALID_ARIAS = []
_C.INVALID_EXOS = []
_C.SEQUENCE_TOTAL_TIME = -1
_C.EXO_CALIBRATION_ROOT = ""

_C.GEOMETRY = CN()
_C.GEOMETRY.MANUAL_GROUND_PLANE_POINTS = ""

_C.CALIBRATION = CN()
_C.CALIBRATION.MANUAL_EXO_CAMERAS = []
_C.CALIBRATION.MANUAL_EGO_CAMERAS = []
_C.CALIBRATION.MANUAL_INTRINSICS_OF_EXO_CAMERAS = []
_C.CALIBRATION.MANUAL_INTRINSICS_FROM_EXO_CAMERAS = []
_C.CALIBRATION.ANCHOR_EGO_CAMERA = (
    "aria01"  ## all cameras are transformed to this camera's world
)

_C.BBOX = CN()
_C.BBOX.MIN_VERTICES = 40
_C.BBOX.ROI_CYLINDER_RADIUS = 0.3
_C.BBOX.VIS_CAMERAS = []
_C.BBOX.HUMAN_HEIGHT = None

_C.BBOX.EGO = CN()
_C.BBOX.EGO.MIN_AREA_RATIO = 0.005
_C.BBOX.EGO.CLOSE_BBOX_DISTANCE = 2.0
_C.BBOX.EGO.CLOSE_BBOX_MIN_AREA_RATIO = 0.01
_C.BBOX.EGO.MAX_ASPECT_RATIO = 3
_C.BBOX.EGO.MIN_ASPECT_RATIO = 0.3
_C.BBOX.SAVE_OFFSHELF_BOX_TO_DISK = False

_C.BBOX.EXO = CN()
_C.BBOX.EXO.MIN_AREA_RATIO = 0.001
_C.BBOX.EXO.MAX_ASPECT_RATIO = 4
_C.BBOX.EXO.MIN_ASPECT_RATIO = 0.4

_C.BBOX.CONFIDENCE_THRESHOLD_FOR_SEGMENTATION = 0.7

_C.POSE2D = CN()
_C.POSE2D.VIS_CAMERAS = []
_C.POSE2D.USE_BBOX_DETECTOR = False
_C.POSE2D.DETECTOR_CONFIG_FILE = (
    "/home/rawalk/Desktop/ego/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
)
_C.POSE2D.DETECTOR_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
_C.POSE2D.DETECTOR_MIN_IOU = 0.1

_C.POSE2D.DEBUG = False
_C.POSE2D.DUMMY_RGB_CONFIG_FILE = "/home/rawalk/Desktop/ego/vitpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py"
_C.POSE2D.DUMMY_RGB_CHECKPOINT = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"
_C.POSE2D.RGB_CONFIG_FILE = "/home/rawalk/Desktop/ego/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
_C.POSE2D.RGB_CHECKPOINT = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

_C.POSE2D.RGB_SEG_CONFIG_FILE = "/home/rawalk/Desktop/ego/vitpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seg_ViTPose_large_coco_256x192.py"
_C.POSE2D.RGB_SEG_CHECKPOINT = (
    "/media/rawalk/disk1/rawalk/vitpose/seg_checkpoints/large_epoch_140.pth"
)

_C.POSE2D.GRAY_CONFIG_FILE = "/home/rawalk/Desktop/ego/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody-grayscale/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
_C.POSE2D.GRAY_CHECKPOINT = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

_C.POSE2D.BBOX_THRES = 0.5  ## keypoint thres
_C.POSE2D.RGB_THRES = 0.5  ## keypoint thres
_C.POSE2D.RGB_VIS_THRES = 0.5  ## keypoint thres
_C.POSE2D.GRAY_THRES = 0.5  ## keypoint thres
_C.POSE2D.GRAY_VIS_THRES = 0.5  ## keypoint thres
_C.POSE2D.MIN_VIS_KEYPOINTS = 5
_C.POSE2D.OVERLAP_OKS_THRES = (
    0.8  ## if overlap thres greater than this, both poses are removed
)

_C.POSE2D.VIS = CN()
_C.POSE2D.VIS.RADIUS = CN()
_C.POSE2D.VIS.RADIUS.EXO_RGB = 10
_C.POSE2D.VIS.RADIUS.EGO_RGB = 5
_C.POSE2D.VIS.RADIUS.EGO_LEFT = 2
_C.POSE2D.VIS.RADIUS.EGO_RIGHT = 2

_C.POSE2D.VIS.THICKNESS = CN()
_C.POSE2D.VIS.THICKNESS.EXO_RGB = 10
_C.POSE2D.VIS.THICKNESS.EGO_RGB = 5
_C.POSE2D.VIS.THICKNESS.EGO_LEFT = 2
_C.POSE2D.VIS.THICKNESS.EGO_RIGHT = 2

_C.SEGMENTATION = CN()
_C.SEGMENTATION.VIS_CAMERAS = []
_C.SEGMENTATION.MODEL_TYPE = "vit_h"
_C.SEGMENTATION.CHECKPOINT = (
    "/media/rawalk/disk1/rawalk/segment_anything/sam_vit_h_4b8939.pth"
)
_C.SEGMENTATION.ONNX_CHECKPOINT = (
    "/home/rawalk/Desktop/ego/segment_anything/models/sam.onnx"
)
_C.SEGMENTATION.ONNX_QUANTIZED_CHECKPOINT = (
    "/home/rawalk/Desktop/ego/segment_anything/models/sam_quantized.onnx"
)

_C.POSE3D = CN()
_C.POSE3D.VIS_CAMERAS = []
_C.POSE3D.KEYPOINTS_THRES = 0.5
_C.POSE3D.BBOX_AREA_THRES = 0.003
_C.POSE3D.NUM_ITERS = 800
_C.POSE3D.REPROJECTION_ERROR_EPSILON = 0.01
_C.POSE3D.MIN_VIEWS = 2  ## min views for triangulation
_C.POSE3D.MIN_INLIER_VIEWS = 4  ## min views for triangulation
_C.POSE3D.SECONDARY_MIN_VIEWS = 3  ## min views for triangulation
_C.POSE3D.INCLUDE_CONFIDENCE = False  ## include confidence in triangulation
_C.POSE3D.OVERRIDE = CN()
_C.POSE3D.OVERRIDE.TIMESTAMPS = []  ## ignore the human keypoints at this timestamp from exo cameras
_C.POSE3D.OVERRIDE.HUMAN_NAMES = []
_C.POSE3D.OVERRIDE.EXO_CAMERAS = []
_C.POSE3D.OVERRIDE.KEYPOINT_IDXS = []
_C.POSE3D.USE_SEGPOSE2D = False

_C.REFINE_POSE3D = CN()
_C.REFINE_POSE3D.DEBUG = False
_C.REFINE_POSE3D.STD_THRES = 10  ##+- std deviation allowed for inliers
_C.REFINE_POSE3D.WINDOW_LENGTH = 10  ##window length to interpolate
_C.REFINE_POSE3D.IQR_THRES = 4  ## iqr thres
_C.REFINE_POSE3D.MOTION_THRES = (
    300  ## motion thres between consectutive frames from average
)

_C.FIT_POSE3D = CN()
_C.FIT_POSE3D.VIS_CAMERAS = []
_C.FIT_POSE3D.DEBUG = True
_C.FIT_POSE3D.NUM_EPOCHS = 10
_C.FIT_POSE3D.NUM_ITERS = 500
_C.FIT_POSE3D.LR = 0.1
_C.FIT_POSE3D.MAX_ITER = 20
_C.FIT_POSE3D.INIT_POSE_LOSS_WEIGHT = 1
_C.FIT_POSE3D.SYMMETRY_LOSS_WEIGHT = 1
_C.FIT_POSE3D.TEMPORAL_LOSS_WEIGHT = 1
_C.FIT_POSE3D.LIMB_LENGTH_LOSS_WEIGHT = 1
_C.FIT_POSE3D.FTOL = 1e-4
_C.FIT_POSE3D.GLOBAL_ITERS = 1

_C.INIT_SMPL = CN()
_C.INIT_SMPL.VIS = False
_C.INIT_SMPL.VIS_CAMERAS = []

_C.SMPL = CN()
_C.SMPL.VIS_CAMERAS = []
_C.SMPL.DEBUG = False
_C.SMPL.VERBOSE = True
_C.SMPL.CONFIG_FILE = "smplify3d_temporal.py"
_C.SMPL.ARIA_NAME_LIST = ["aria01", "aria02", "aria03", "aria04"]
_C.SMPL.NUM_EPOCHS_LIST = [10, 10, 10, 10]
_C.SMPL.STAGE1_ITERS_LIST = [50, 50, 50, 50]
_C.SMPL.STAGE2_ITERS_LIST = [20, 20, 20, 20]
_C.SMPL.STAGE3_ITERS_LIST = [120, 120, 120, 120]

# _C.SMPL.ARIA_GENDER_LIST = ['male', 'male', 'male', 'male']
_C.SMPL.ARIA_GENDER_LIST = ["neutral", "neutral", "neutral", "neutral"]

_C.SMPL.JOINT_WEIGHT_OVERRIDE = CN()
_C.SMPL.JOINT_WEIGHT_OVERRIDE.ARIA_NAME_LIST = []
_C.SMPL.JOINT_WEIGHT_OVERRIDE.JOINT_NAMES = []  ## unlike mmhuman3d, we use left-hip and right-hip for COCO and not left-hip-extra and right-hip-extra
_C.SMPL.JOINT_WEIGHT_OVERRIDE.JOINT_WEIGHTS = []


_C.SMPL_COLLISION = CN()
_C.SMPL_COLLISION.VIS_CAMERAS = []
_C.SMPL_COLLISION.CONFIG_FILE = "smplify3d_collision_temporal.py"
_C.SMPL_COLLISION.NUM_EPOCHS = 10

_C.BLENDER = CN()
_C.BLENDER.SCENE_FILE = "tagging/tagging.blend"
_C.BLENDER.COLORS = "blue###green###red###orange"
_C.BLENDER.MAX_OFFSET = 0.25
_C.BLENDER.TOLERANCE = 1e-5
_C.BLENDER.OVERLAY = True
_C.BLENDER.OVERLAY_CAMERA = "cam08"

_C.BLENDER.CAMERA_RELATIVE = CN()
_C.BLENDER.CAMERA_RELATIVE.SCENE_FILE = "scene_camera_relative.blend"
_C.BLENDER.CAMERA_RELATIVE.EGO_SCENE_FILE = (
    "ego_scene_camera_relative.blend"  ## different lighting setup
)
_C.BLENDER.CAMERA_RELATIVE.COLORS = "blue###green###red###orange"


##---------------------------------------
def update_config(cfg, config_file):
    cfg.defrost()
    cfg.merge_from_file(config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
