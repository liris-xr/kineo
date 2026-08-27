import pickle
import numpy as np
import rerun as rr
import os


# ── Configuration ─────────────────────────────────────────────────────────────
data_dir = "D:/Charles_JAVERLIAT/Technique/kineo_lirisxr/outputs/infer_dwpose_single_person_pairwise_calib/offline_demo/annotations/walking"
n_cams = 4

# Triplet: three cameras forming a directed loop  0 → 1 → 2 → 0
PAIRS = [(0, 1), (1, 2), (2, 0)]

# ── Colors ────────────────────────────────────────────────────────────────────

# Single color for the whole triplet chain (baselines, ghost cam, gap).
TRIPLET_COLOR = (40, 160, 210)   # green

# ── Line / frustum appearance ─────────────────────────────────────────────────
image_plane_dist = 0.2    # camera frustum depth
LINE_RADIUS      = 0.015  # radius of baseline lines
GAP_RADIUS       = 0.015  # radius of the dashed closure-gap line
DASH_PERIOD      = 0.08   # world-space length of one dash + gap
DASH_FRAC        = 0.3    # fraction of each period that is solid

# ── Scale animation ───────────────────────────────────────────────────────────
# Each pair animates from its START_SCALE → 1.0 (true scale).
START_SCALE_01: float = 0.5
START_SCALE_12: float = 0.6
START_SCALE_20: float = 0.3

# ── Animation timing ──────────────────────────────────────────────────────────
n_hold      = 20
n_anim      = 90
n_hold_last = 10
n_total     = n_hold + n_anim + n_hold_last


def ease_in_out(x: float) -> float:
    return 3 * x**2 - 2 * x**3


def make_dashed_strips(p0: np.ndarray, p1: np.ndarray,
                       dash_frac: float = DASH_FRAC) -> list:
    """Return dashes with constant world-space spacing regardless of length."""
    length   = float(np.linalg.norm(p1 - p0))
    n_dashes = max(1, round(length / DASH_PERIOD))
    strips   = []
    for k in range(n_dashes):
        t_start = (k + 0.0) / n_dashes
        t_end   = (k + dash_frac) / n_dashes
        strips.append(np.stack([
            p0 + t_start * (p1 - p0),
            p0 + t_end   * (p1 - p0),
        ]))
    return strips


# ── Load data ─────────────────────────────────────────────────────────────────
with open(os.path.join(data_dir, "cameras_intrinsics.pkl"), "rb") as f:
    data   = pickle.load(f)["annotations"]
    K      = np.asarray([data[i]["K"]            for i in range(n_cams)])
    res_hw = [data[i]["resolution_hw"]            for i in range(n_cams)]

with open(os.path.join(data_dir, "cameras_extrinsics.pkl"), "rb") as f:
    data = pickle.load(f)["annotations"]
    R    = np.asarray([data[i]["R"] for i in range(n_cams)])
    t    = np.asarray([data[i]["t"] for i in range(n_cams)])

# ── Camera geometry ───────────────────────────────────────────────────────────
R_c2w       = np.transpose(R, (0, 2, 1))
cam_centers = (-R_c2w @ t[..., None]).squeeze(-1)

# ── Relative transforms ───────────────────────────────────────────────────────
rel_R = {(i, j): R[j] @ R[i].T                for i, j in PAIRS}
rel_t = {(i, j): t[j] - rel_R[(i, j)] @ t[i]  for i, j in PAIRS}


def loop_loss(s_dict: dict) -> float:
    """‖t₂₀·s + R₂₀·(s·t₁₂) + R₂₀·R₁₂·(s·t₀₁)‖ — zero at true scale."""
    t01 = s_dict[(0, 1)] * rel_t[(0, 1)]
    t12 = s_dict[(1, 2)] * rel_t[(1, 2)]
    t20 = s_dict[(2, 0)] * rel_t[(2, 0)]
    residual = t20 + rel_R[(2, 0)] @ t12 + rel_R[(2, 0)] @ rel_R[(1, 2)] @ t01
    return float(np.linalg.norm(residual))


def log_camera(path: str, cam_idx: int, world_pos: np.ndarray,
               ipd: float = image_plane_dist,
               color: tuple = None) -> None:
    """Log a camera frustum."""
    rr.log(path,
           rr.Transform3D(translation=world_pos, mat3x3=R_c2w[cam_idx]),
           rr.TransformAxes3D(axis_length=0.05))
    rr.log(path,
           rr.Pinhole(image_from_camera=K[cam_idx], width=res_hw[cam_idx][1],
                      height=res_hw[cam_idx][0], image_plane_distance=ipd,
                      color=color))


# ── Rerun init ────────────────────────────────────────────────────────────────
rr.init("triplet_loop_closure")
rr.spawn()
rr.log("/", rr.ViewCoordinates.BLU, static=True)

# ── Animation loop ────────────────────────────────────────────────────────────
for frame_idx in range(n_total):
    rr.set_time("frame", sequence=frame_idx)

    # Global animation alpha
    if frame_idx < n_hold:
        alpha = 0.0
    elif frame_idx < n_hold + n_anim:
        alpha = ease_in_out((frame_idx - n_hold) / (n_anim - 1))
    else:
        alpha = 1.0

    # All pairs interpolate from their start_scale to 1.0 in log-space.
    # s = start_scale^(1-alpha)  →  start_scale at alpha=0, 1.0 at alpha=1
    scales = {
        (0, 1): float(START_SCALE_01 ** (1.0 - alpha)),
        (1, 2): float(START_SCALE_12 ** (1.0 - alpha)),
        (2, 0): float(START_SCALE_20 ** (1.0 - alpha)),
    }

    # ── Chained camera world positions ────────────────────────────────────────
    anchor_0   = cam_centers[0]
    cam1_world = R_c2w[0] @ (-rel_R[(0, 1)].T @ (scales[(0, 1)] * rel_t[(0, 1)])) + anchor_0
    cam2_world = R_c2w[1] @ (-rel_R[(1, 2)].T @ (scales[(1, 2)] * rel_t[(1, 2)])) + cam1_world
    # Ghost cam 0: where cam 0 lands when closing the loop from cam 2
    cam0_ghost = R_c2w[2] @ (-rel_R[(2, 0)].T @ (scales[(2, 0)] * rel_t[(2, 0)])) + cam2_world

    # ── Log cameras (triplet color) ───────────────────────────────────────────
    log_camera("chain/cam_0",       0, anchor_0,   color=TRIPLET_COLOR)
    log_camera("chain/cam_1",       1, cam1_world, color=TRIPLET_COLOR)
    log_camera("chain/cam_2",       2, cam2_world, color=TRIPLET_COLOR)
    log_camera("chain/cam_0_ghost", 0, cam0_ghost, color=TRIPLET_COLOR)

    # ── Chain lines (all in TRIPLET_COLOR) ────────────────────────────────────
    for name, p_start, p_end in [
        ("chain/baseline_01", anchor_0,   cam1_world),
        ("chain/baseline_12", cam1_world, cam2_world),
        ("chain/baseline_20", cam2_world, cam0_ghost),
    ]:
        rr.log(name,
               rr.LineStrips3D(strips=[np.stack([p_start, p_end])],
                               colors=[TRIPLET_COLOR], radii=LINE_RADIUS))

    # ── Closure gap: dashed line from ghost cam 0 → true cam 0 ───────────────
    gap = np.linalg.norm(cam0_ghost - anchor_0)
    if gap > 1e-6:
        rr.log("chain/closure_gap",
               rr.LineStrips3D(strips=make_dashed_strips(cam0_ghost, anchor_0),
                               colors=[TRIPLET_COLOR], radii=GAP_RADIUS))
    else:
        rr.log("chain/closure_gap",
               rr.LineStrips3D(strips=[], colors=[], radii=GAP_RADIUS))

    # ── Scalar metrics ────────────────────────────────────────────────────────
    rr.log("loop/loss", rr.Scalars(loop_loss(scales)))
    rr.log("loop/loss", rr.SeriesLines(widths=2))
    for i, j in PAIRS:
        rr.log(f"scales/pair_{i}_{j}", rr.Scalars(scales[(i, j)]))

print(f"Done — {n_total} frames.")
print("Set START_SCALE_01 / START_SCALE_12 / START_SCALE_20 != 1.0 to visualise scale drift.")