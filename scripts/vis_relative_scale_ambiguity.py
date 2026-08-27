import pickle
import numpy as np
import rerun as rr
import os


# COCO-17 skeleton connectivity (DWPose body keypoints)
# fmt: off
COCO_CONNECTIVITY = [
    (0, 1), (0, 2),               # nose → eyes
    (1, 3), (2, 4),               # eyes → ears
    (5, 6),                       # shoulders
    (5, 7), (7, 9),               # left  arm
    (6, 8), (8, 10),              # right arm
    (5, 11), (6, 12),             # shoulders → hips
    (11, 12),                     # hips
    (11, 13), (13, 15),           # left  leg
    (12, 14), (14, 16),           # right leg
]
# fmt: on

rr.init("scale_ambiguity")
rr.spawn()

# Declare OpenCV world convention (X=Right, Y=Down, Z=Forward)
rr.log("/", rr.ViewCoordinates.BLU, static=True)

n_cams = 4
data_dir = "D:/Charles_JAVERLIAT/Technique/kineo_lirisxr/outputs/infer_dwpose_single_person_pairwise_calib/offline_demo/annotations/walking"
image_plane_dist = 0.2

# ── Single pair ───────────────────────────────────────────────────────────────
i, j = 0, 1
PAIR_COLOR   = (255, 100,  80)   # coral   — animated pair


# ── Animation: scale oscillates 1 → 2 → 1 (ping-pong) ───────────────────────
n_anim       = 60    # frames for one half-cycle (1→2 or 2→1)
n_hold_end   = 0    # frames to hold at each extreme before reversing
n_cycles     = 1     # number of full back-and-forth cycles
n_total      = n_cycles * 2 * (n_anim + n_hold_end)

LINE_RADIUS          = 0.03

SKELETON_JOINTS_RADIUS = LINE_RADIUS + 0.01
SKELETON_BONES_RADIUS = LINE_RADIUS

SCALE_MIN = 1.0
SCALE_MAX = 2.0


def ease_in_out(x: float) -> float:
    """Smooth cubic ease-in-out: 0→0, 1→1."""
    return 3 * x**2 - 2 * x**3


# ── Load data ──────────────────────────────────────────────────────────────────
with open(os.path.join(data_dir, "keypoints_3d.pkl"), "rb") as f:
    data = pickle.load(f)
    kps3d = np.asarray(data["annotations"][0]["xyz"])  # (N, 3) world coordinates

with open(os.path.join(data_dir, "cameras_intrinsics.pkl"), "rb") as f:
    data = pickle.load(f)["annotations"]
    K      = np.asarray([data[c]["K"]             for c in range(n_cams)])
    res_hw = [data[c]["resolution_hw"]             for c in range(n_cams)]

with open(os.path.join(data_dir, "cameras_extrinsics.pkl"), "rb") as f:
    data = pickle.load(f)["annotations"]
    R = np.asarray([data[c]["R"] for c in range(n_cams)])  # (n_cams,3,3) world→cam
    t = np.asarray([data[c]["t"] for c in range(n_cams)])  # (n_cams,3)

# ── Absolute camera centres and rotations in world frame ─────────────────────
R_c2w       = np.transpose(R, (0, 2, 1))                        # (n_cams, 3, 3)
cam_centers = (-R_c2w @ t[..., None]).squeeze(-1)               # (n_cams, 3)

# ── Keypoints in camera-i local frame ────────────────────────────────────────
kps3d_in_cam_i = (R[i] @ kps3d.T).T + t[i]                     # (N, 3)

# ── Relative pose (i→j) ───────────────────────────────────────────────────────
# R_rel @ p_i + s * t_rel = p_j   (with s=1 being the true scale)
R_rel = R[j] @ R[i].T
t_rel = t[j] - R_rel @ t[i]


def world_kps(s: float) -> np.ndarray:
    """3-D skeleton in world frame for a given scale s."""
    # Place cam-i at its true position; scale the baseline.
    return (R_c2w[i] @ (s * kps3d_in_cam_i).T).T + cam_centers[i]


def cam_j_center(s: float) -> np.ndarray:
    """World position of camera j at scale s (cam-i anchored at truth)."""
    return R_c2w[i] @ (-R_rel.T @ (s * t_rel)) + cam_centers[i]


# ── Per-frame animated log ────────────────────────────────────────────────────
half_period = n_anim + n_hold_end   # frames for one direction (e.g. 1→2)

for frame_idx in range(n_total):
    rr.set_time("frame", sequence=frame_idx)

    # Position within the current half-cycle
    phase_frame = frame_idx % half_period
    going_up    = (frame_idx // half_period) % 2 == 0  # True: 1→2, False: 2→1

    # Raw alpha [0..1] with easing, held at 0 or 1 during the hold phase
    if phase_frame < n_anim:
        alpha = ease_in_out(phase_frame / (n_anim - 1))
    else:
        alpha = 1.0

    if not going_up:
        alpha = 1.0 - alpha   # reverse direction

    s = SCALE_MIN + alpha * (SCALE_MAX - SCALE_MIN)

    # ── Animated cam-i (fixed) + cam-j ───────────────────────────────────────
    rr.log(f"pair/cam_{i}",
           rr.Pinhole(image_from_camera=K[i], width=res_hw[i][1],
                      height=res_hw[i][0], image_plane_distance=image_plane_dist))
    rr.log(f"pair/cam_{i}",
           rr.Transform3D(translation=cam_centers[i], mat3x3=R_c2w[i]),
           rr.TransformAxes3D(axis_length=0))

    cj = cam_j_center(s)
    rr.log(f"pair/cam_{j}",
           rr.Pinhole(image_from_camera=K[j], width=res_hw[j][1],
                      height=res_hw[j][0], image_plane_distance=image_plane_dist))
    rr.log(f"pair/cam_{j}",
           rr.Transform3D(translation=cj, mat3x3=R_c2w[j]),
           rr.TransformAxes3D(axis_length=0))

    # ── Baseline ─────────────────────────────────────────────────────────────
    rr.log("pair/baseline",
           rr.LineStrips3D(strips=[np.stack([cam_centers[i], cj])],
                           colors=[PAIR_COLOR], radii=LINE_RADIUS))

    # ── Skeleton ─────────────────────────────────────────────────────────────
    kps_world = world_kps(s)
    rr.log("pair/skeleton/joints",
           rr.Points3D(kps_world, colors=[PAIR_COLOR], radii=SKELETON_JOINTS_RADIUS))
    bones = [[kps_world[a], kps_world[b]]
             for a, b in COCO_CONNECTIVITY
             if a < len(kps_world) and b < len(kps_world)]
    if bones:
        rr.log("pair/skeleton/bones",
               rr.LineStrips3D(strips=np.array(bones), colors=[PAIR_COLOR], radii=SKELETON_BONES_RADIUS))

    # ── Scalar monitors ───────────────────────────────────────────────────────
    rr.log("pair/scale",          rr.Scalars(s))
    rr.log("pair/baseline_norm",  rr.Scalars(np.linalg.norm(cj - cam_centers[i])))

print(
    "Done. Scrub the 'frame' timeline in Rerun.\n"
    "Scale oscillates 1 → 2 → 1: all frames are equally consistent with\n"
    "the image observations — that is the scale ambiguity."
)