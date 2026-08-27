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

# Mask of keypoints that appear in at least one bone connection
CONNECTED_KPS = np.array(sorted({idx for pair in COCO_CONNECTIVITY for idx in pair}))

LINE_RADIUS          = 0.02

SKELETON_JOINTS_RADIUS = LINE_RADIUS + 0.01
SKELETON_BONES_RADIUS = LINE_RADIUS

rr.init("kps3d_relative_pairs")
rr.spawn()

# Declare OpenCV world convention (X=Right, Y=Down, Z=Forward)
rr.log("/", rr.ViewCoordinates.BLU, static=True)

n_cams = 4
data_dir = "D:/Charles_JAVERLIAT/Technique/kineo_lirisxr/outputs/infer_dwpose_single_person_pairwise_calib/offline_demo/annotations/walking"
image_plane_dist = 0.2

# ── Animation timing ──────────────────────────────────────────────────────────
# Phase 0: pair 0 appears at correct scale, holds alone.
# Phase k (k>0): all previous pairs frozen at s=1, pair k animates unit→correct.
n_hold        = 10   # frames to display the first pair before any animation
n_delay_first = 10   # frames to wait before the second pair appears
n_delay       = 10   # frames to wait before each subsequent pair appears
n_hold_new    = 20   # frames a new pair stays at unit-norm before animating
n_anim        = 40   # frames for each scale animation
n_pop         = 12   # frames for the camera pop-in appearance animation
n_hold_last   = 40   # frames to hold after the last pair has converged

# ── Fail simulation (applied to pair index 1) ─────────────────────────────────
fail_pair_idx      = 1          # which pair index simulates a failure
fail_scale_offset  = -0.1       # added to s=1 target (e.g. 0.9 instead of 1.0)
fail_noise_std     = 0.08       # std of gaussian noise added to skeleton keypoints (metres)
n_fail_delay       = 40         # frames to hold cleanly before the fail effect starts
n_fail_anim        = 60         # frames over which the fail effect is revealed
rng = np.random.default_rng(0)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(os.path.join(data_dir, "keypoints_3d.pkl"), "rb") as f:
    data = pickle.load(f)
    kps3d = np.asarray(data["annotations"][0]["xyz"])  # (N, 3) world coordinates

with open(os.path.join(data_dir, "cameras_intrinsics.pkl"), "rb") as f:
    data = pickle.load(f)["annotations"]
    K = np.asarray([data[i]["K"] for i in range(n_cams)])
    res_hw = [data[i]["resolution_hw"] for i in range(n_cams)]

with open(os.path.join(data_dir, "cameras_extrinsics.pkl"), "rb") as f:
    data = pickle.load(f)["annotations"]
    R = np.asarray([data[i]["R"] for i in range(n_cams)])   # (n_cams,3,3) world→cam
    t = np.asarray([data[i]["t"] for i in range(n_cams)])   # (n_cams,3)

# ── Absolute camera centres and orientations in world frame ───────────────────
# cam_center[i] = -R[i].T @ t[i]    (camera position in world)
# R_c2w[i]      = R[i].T            (rotation: cam → world)
R_c2w      = np.transpose(R, (0, 2, 1))                               # (n_cams, 3, 3)
cam_centers = (-R_c2w @ t[..., None]).squeeze(-1)                     # (n_cams, 3)

# ── Keypoints in each camera's local frame ────────────────────────────────────
# p_cam_i = R[i] @ p_world + t[i]
kps3d_in_cam = (R @ kps3d.T).transpose(0, 2, 1) + t[:, None, :]      # (n_cams, N, 3)

# ── 3 pairs: 0→1, then 0→2, then 2→3 ────────────────────────────────────────
pairs = [(0, 1), (0, 2), (2, 3)]

# ── One distinct color per pair ───────────────────────────────────────────────
PAIR_COLORS = {
    (0, 1): (255, 100,  80),   # coral
    (0, 2): ( 80, 200, 120),   # mint green
    (2, 3): ( 80, 160, 255),   # sky blue
}

# ── Scale-independent relative rotation / translation ─────────────────────────
# R_rel = R[j] @ R[i].T
# t_rel = t[j] - R_rel @ t[i]   (true baseline; scale is factored out at render time)
rel_R = {(i, j): R[j] @ R[i].T                   for i, j in pairs}
rel_t = {(i, j): t[j] - rel_R[(i, j)] @ t[i]     for i, j in pairs}

# Initial scale = 1 / ‖t_rel‖ so each pair starts with a unit-norm baseline.
# Animates to s=1 (true scale) so ‖t_rel‖ converges to the correct value.
unit_scales = {(i, j): 1.0 / np.linalg.norm(rel_t[(i, j)]) for i, j in pairs}


def ease_in_out(x: float) -> float:
    """Smooth cubic ease-in-out: 0→0, 1→1."""
    return 3 * x**2 - 2 * x**3


def ease_pop(x: float) -> float:
    """Elastic overshoot: grows past 1 then settles, giving a pop feel."""
    return 1 - (1 - x) ** 3 * np.cos(x * np.pi * 2.5)


# ── Pre-compute static noise ──────────────────────────────────────────────────
# Noise is fixed (not per-frame) so the skeleton is stable but wrong.
# It is blended in gradually via noise_weight after the scale animation ends.
kps_noise = rng.normal(0.0, fail_noise_std, kps3d.shape) if fail_noise_std > 0.0 else np.zeros_like(kps3d)

TARGET_COLOR = (128, 128, 128)  # mid gray — ground-truth target

# ── Static logs: target config (ground truth) ─────────────────────────────────
for cam_idx in range(n_cams):
    rr.log(
        f"target/cam_{cam_idx}",
        rr.Pinhole(
            image_from_camera=K[cam_idx],
            width=res_hw[cam_idx][1],
            height=res_hw[cam_idx][0],
            image_plane_distance=image_plane_dist,
        ),
        static=True,
    )
    rr.log(
        f"target/cam_{cam_idx}",
        rr.Transform3D(translation=cam_centers[cam_idx], mat3x3=R_c2w[cam_idx]),
        rr.TransformAxes3D(axis_length=0),
        static=True,
    )

bones_target = [
    [kps3d[a], kps3d[b]]
    for a, b in COCO_CONNECTIVITY
    if a < len(kps3d) and b < len(kps3d)
]
rr.log("target/skeleton/joints", rr.Points3D(kps3d, colors=[TARGET_COLOR], radii=0.015), static=True)
rr.log("target/skeleton/bones", rr.LineStrips3D(strips=np.array(bones_target), colors=[TARGET_COLOR], radii=0.008), static=True)

# ── Timeline helpers ──────────────────────────────────────────────────────────
SLOT = n_delay + n_hold_new + n_anim   # frames consumed per pair after the second

def pair_start_frame(pair_idx: int) -> int:
    """Frame at which pair first appears (at unit-norm scale)."""
    if pair_idx == 0:
        return 0
    if pair_idx == 1:
        return n_hold + n_delay_first
    return n_hold + n_delay_first + (pair_idx - 1) * SLOT + n_delay

def pair_anim_frame(pair_idx: int) -> int:
    """Frame at which pair starts its scale animation."""
    return pair_start_frame(pair_idx) if pair_idx == 0 else pair_start_frame(pair_idx) + n_hold_new


n_total = pair_start_frame(len(pairs) - 1) + n_hold_new + n_anim + n_fail_delay + n_fail_anim + n_hold_last


def cam_j_world_pos(i: int, j: int, s_eff: float, anchor: np.ndarray) -> np.ndarray:
    """World position of camera j given anchor of camera i and effective scale."""
    return R_c2w[i] @ (-rel_R[(i, j)].T @ (s_eff * rel_t[(i, j)])) + anchor


def log_pair(pair_idx: int, i: int, j: int, s: float, pop_scale: float = 1.0,
             scale_offset: float = 0.0, noise_weight: float = 0.0,
             anchor: np.ndarray | None = None) -> None:
    """Log all entities for one pair at effective scale (s + scale_offset).

    scale_offset and noise_weight are both 0 during the scale animation and
    blend smoothly to their target values only *after* it completes, so the
    viewer first sees a clean convergence and then the failure effect.
    """
    pair_name = f"pair_{i}_{j}"
    color     = PAIR_COLORS[(i, j)]
    anchor    = anchor if anchor is not None else cam_centers[i]

    s_eff          = s + scale_offset
    kps_world      = (R_c2w[i] @ (s_eff * kps3d_in_cam[i]).T).T + anchor
    if noise_weight > 0.0:
        kps_world[CONNECTED_KPS] += kps_noise[CONNECTED_KPS] * noise_weight
    cam_j_center   = cam_j_world_pos(i, j, s_eff, anchor)

    rr.log(f"pairs/{pair_name}/cam_{j}",
           rr.Transform3D(translation=cam_j_center, mat3x3=R_c2w[j]),
           rr.TransformAxes3D(axis_length=0))
    rr.log(f"pairs/{pair_name}/cam_{j}",
           rr.Pinhole(image_from_camera=K[j], width=res_hw[j][1],
                      height=res_hw[j][0],
                      image_plane_distance=image_plane_dist * pop_scale))
    rr.log(f"pairs/{pair_name}/baseline",
           rr.LineStrips3D(strips=[np.stack([anchor, cam_j_center])],
                           colors=[color], radii=LINE_RADIUS))
    rr.log(f"pairs/{pair_name}/skeleton/joints",
           rr.Points3D(kps_world, colors=[color], radii=SKELETON_JOINTS_RADIUS))
    bones = [[kps_world[a], kps_world[b]]
             for a, b in COCO_CONNECTIVITY
             if a < len(kps_world) and b < len(kps_world)]
    if bones:
        rr.log(f"pairs/{pair_name}/skeleton/bones",
               rr.LineStrips3D(strips=np.array(bones), colors=[color], radii=SKELETON_BONES_RADIUS))

    rr.log(f"pairs/{pair_name}/scale",       rr.Scalars(s))
    rr.log(f"pairs/{pair_name}/scale_error", rr.Scalars(abs(s_eff - 1.0)))


logged_cams = set()

for frame_idx in range(n_total):
    rr.set_time("frame", sequence=frame_idx)

    # ── Per-frame fail-reveal alpha (0 → 1, starts only after ALL pairs settled) ─
    # The fail pair looks perfectly correct while the full sequence is playing out.
    # Only once the last pair's scale animation finishes do the scale_offset and
    # noise blend in, so the viewer first sees clean convergence for every pair.
    _last_pair_idx = len(pairs) - 1
    _fail_anim_end = pair_anim_frame(_last_pair_idx) + n_anim + n_fail_delay  # frame where last scale anim ends
    if frame_idx <= _fail_anim_end:
        fail_alpha = 0.0
    elif frame_idx < _fail_anim_end + n_fail_anim:
        raw_alpha  = (frame_idx - _fail_anim_end) / (n_fail_anim - 1)
        fail_alpha = ease_in_out(raw_alpha)
    else:
        fail_alpha = 1.0

    # ── Dynamically track where cam_j of the fail pair currently sits ─────────
    # Subsequent pairs (e.g. (2, 3)) anchor off this camera, so their position
    # naturally follows the drift introduced by the fail reveal.
    _fi, _fj = pairs[fail_pair_idx]
    _s_fail_current    = 1.0 + fail_alpha * fail_scale_offset   # s=1 post-anim
    _current_fail_camj = cam_j_world_pos(_fi, _fj, _s_fail_current, cam_centers[_fi])

    # Map camera index → its current world position (updated each frame).
    # cam_0 and cam_1 are always at their ground-truth positions.
    dynamic_cam_centers = dict(enumerate(cam_centers))
    dynamic_cam_centers[_fj] = _current_fail_camj

    for pair_idx, (i, j) in enumerate(pairs):
        start      = pair_start_frame(pair_idx)
        anim_start = pair_anim_frame(pair_idx)

        if frame_idx < start:
            continue

        # ── On the very first frame of this pair, log cam_i (if new) + cam_j ──
        if frame_idx == start:
            pair_name = f"pair_{i}_{j}"
            if i not in logged_cams:
                rr.log(f"pairs/{pair_name}/cam_{i}",
                       rr.Pinhole(image_from_camera=K[i], width=res_hw[i][1],
                                   height=res_hw[i][0], image_plane_distance=image_plane_dist))
                rr.log(f"pairs/{pair_name}/cam_{i}",
                       rr.Transform3D(translation=cam_centers[i], mat3x3=R_c2w[i]),
                       rr.TransformAxes3D(axis_length=0))
                logged_cams.add(i)
            logged_cams.add(j)

        # ── Scale s: unit-norm → 1.0 during anim window, frozen outside it ────
        if pair_idx == 0 or frame_idx >= anim_start + n_anim:
            s = 1.0
        elif frame_idx < anim_start:
            s = unit_scales[(i, j)]
        else:
            raw_alpha = (frame_idx - anim_start) / (n_anim - 1)
            alpha     = ease_in_out(raw_alpha)
            s         = np.exp(np.log(unit_scales[(i, j)]) * (1.0 - alpha))

        # ── Pop scale: animate cam_j frustum size on appearance ───────────────
        frames_since_start = frame_idx - start
        pop_scale = ease_pop(frames_since_start / (n_pop - 1)) if pair_idx > 0 and frames_since_start < n_pop else 1.0

        # ── Fail effect: only applied to the designated pair, only post-anim ──
        # scale_offset and noise_weight are both 0 while the scale is converging
        # and only blend in smoothly once the convergence animation finishes.
        if pair_idx == fail_pair_idx:
            scale_offset  = fail_alpha * fail_scale_offset
            noise_weight  = fail_alpha
        else:
            scale_offset  = 0.0
            noise_weight  = 0.0

        # ── Anchor: use the dynamically tracked position for downstream pairs ─
        anchor = dynamic_cam_centers[i]

        log_pair(pair_idx, i, j, s, pop_scale, scale_offset, noise_weight, anchor)

print("Done. Scrub the 'frame' timeline in Rerun to watch the scale converge, "
      "then the fail effect reveal itself.")