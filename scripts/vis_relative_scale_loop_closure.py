import pickle
import numpy as np
import rerun as rr
import os
from itertools import combinations

# ── Configuration ──────────────────────────────────────────────────────────────
data_dir = "D:/Charles_JAVERLIAT/Technique/kineo_lirisxr/outputs/infer_dwpose_single_person_pairwise_calib/offline_demo/annotations/walking"
n_cams = 4

all_pairs = list(combinations(range(n_cams), 2))  # 6 pairs
all_triplets = list(combinations(range(n_cams), 3))  # 4 triplets

FIXED_PAIR = (0, 1)  # scale locked at unit_scale[(0,1)] throughout

# ── Colors (chosen for visibility on a white background) ───────────────────
CAM_COLORS = {
    0: (200, 30, 30),  # red
    1: (20, 150, 20),  # green
    2: (20, 80, 200),  # blue
    3: (180, 130, 0),  # dark yellow
}

PAIR_COLORS = {
    (0, 1): (120, 120, 120),  # grey   (fixed pair)
    (0, 2): (210, 80, 0),  # burnt orange
    (0, 3): (160, 120, 0),  # dark gold
    (1, 2): (0, 150, 60),  # dark mint
    (1, 3): (0, 150, 170),  # dark cyan
    (2, 3): (120, 20, 210),  # violet
}

TRIPLET_COLORS = {
    (0, 1, 2): (180, 0, 120),  # magenta
    (0, 1, 3): (200, 80, 0),  # orange
    (0, 2, 3): (0, 160, 140),  # teal
    (1, 2, 3): (80, 160, 0),  # olive green
}

# ── Line / frustum appearance ─────────────────────────────────────────────
image_plane_dist = 0.2  # camera frustum depth
LINE_RADIUS = 0.015  # base radius of chain lines
GAP_RADIUS = 0.015  # radius of the dashed closure-gap line
DASH_PERIOD = 0.08  # world-space length of one dash + gap
DASH_FRAC = 0.3  # fraction of each period that is solid
IDLE_THICKNESS_SCALE = 0.4  # line/frustum scale for non-active triplets

# ── Intro animation timing ────────────────────────────────────────────────
N_POP = 12  # frames for the camera pop-in animation
N_INTRO_PER_TRIPLET = 30  # total frames allocated per triplet
N_INTRO_HOLD = 5  # frames to hold once all triplets are shown
n_intro = len(all_triplets) * N_INTRO_PER_TRIPLET + N_INTRO_HOLD

# ── Optimisation / animation timing ──────────────────────────────────────
N_OPT_STEPS = 100  # gradient descent steps to compute
N_SLOW_STEPS = 0  # first N steps get the slow frame rate
FRAMES_PER_STEP_SLOW = 2  # frames per step for the slow portion
FRAMES_PER_STEP_FAST = 2  # frames per step for the rest
n_hold = 20  # hold frames at the start (initial scales)
n_hold_last = 50  # hold frames after convergence

# Precompute cumulative frame boundaries for each optimisation step.
# frame_to_frac[f] gives the fractional step index at animation frame f.
_frames_per_step = [FRAMES_PER_STEP_SLOW] * N_SLOW_STEPS + [FRAMES_PER_STEP_FAST] * (
    N_OPT_STEPS - N_SLOW_STEPS
)
_step_start_frame = [0]  # _step_start_frame[k] = anim frame where step k begins
for _fps in _frames_per_step:
    _step_start_frame.append(_step_start_frame[-1] + _fps)
n_anim_frames = _step_start_frame[-1]  # total animation frames (excl. hold)
n_total_opt = n_hold + n_anim_frames + n_hold_last
n_total = n_intro + n_total_opt  # intro then full optimisation


def ease_pop(x: float) -> float:
    """Elastic overshoot: grows past 1 then settles, giving a pop feel."""
    return 1 - (1 - x) ** 3 * np.cos(x * np.pi * 2.5)


def anim_frame_to_frac_step(anim_frame: int) -> float:
    """Map an animation frame (0-based, after hold) to a fractional step index."""
    anim_frame = min(anim_frame, n_anim_frames)
    # Binary search for which step interval this frame falls in
    lo, hi = 0, N_OPT_STEPS
    while lo < hi:
        mid = (lo + hi) // 2
        if _step_start_frame[mid + 1] <= anim_frame:
            lo = mid + 1
        else:
            hi = mid
    step = lo
    if step >= N_OPT_STEPS:
        return float(N_OPT_STEPS)
    span = _step_start_frame[step + 1] - _step_start_frame[step]
    frac = (anim_frame - _step_start_frame[step]) / span
    return min(step + frac, N_OPT_STEPS)


def make_dashed_strips(p0: np.ndarray, p1: np.ndarray, f: float = DASH_FRAC) -> list:
    """Constant world-space dash spacing regardless of segment length."""
    length = float(np.linalg.norm(p1 - p0))
    n = max(1, round(length / DASH_PERIOD))
    return [
        np.stack([p0 + (k / n) * (p1 - p0), p0 + ((k + f) / n) * (p1 - p0)])
        for k in range(n)
    ]


# ── Load data ──────────────────────────────────────────────────────────────────
with open(os.path.join(data_dir, "cameras_intrinsics.pkl"), "rb") as f:
    data = pickle.load(f)["annotations"]
    K = np.asarray([data[i]["K"] for i in range(n_cams)])
    res_hw = [data[i]["resolution_hw"] for i in range(n_cams)]

with open(os.path.join(data_dir, "cameras_extrinsics.pkl"), "rb") as f:
    data = pickle.load(f)["annotations"]
    R = np.asarray([data[i]["R"] for i in range(n_cams)])
    t = np.asarray([data[i]["t"] for i in range(n_cams)])

# ── Camera geometry ────────────────────────────────────────────────────────────
R_c2w = np.transpose(R, (0, 2, 1))
cam_centers = (-R_c2w @ t[..., None]).squeeze(-1)

rel_R = {(i, j): R[j] @ R[i].T for i, j in all_pairs}
rel_t = {(i, j): t[j] - rel_R[(i, j)] @ t[i] for i, j in all_pairs}
unit_scales = {(i, j): 1.0 / np.linalg.norm(rel_t[(i, j)]) for i, j in all_pairs}

# ── Optimization setup ─────────────────────────────────────────────────────────
# Pair (0,1) is fixed at its unit-norm scale throughout.
# The remaining 5 pairs are optimized to minimise the cumulative triplet loss.

s_fixed = 1.0  # true scale from the file
free_pairs = [p for p in all_pairs if p != FIXED_PAIR]  # 5 free scales
n_free = len(free_pairs)
pidx = {p: k for k, p in enumerate(free_pairs)}  # pair → index in x


def build_s(x: np.ndarray) -> dict:
    """Build a full scale dict from the optimisation vector x."""
    s = {FIXED_PAIR: s_fixed}
    for k, p in enumerate(free_pairs):
        s[p] = float(x[k])
    return s


def triplet_residual(s: dict, a: int, b: int, c: int) -> np.ndarray:
    """Loop-closure residual for triplet a→b→c→a.

    r = R_bc @ (s_ab * t_ab) + s_bc * t_bc − s_ac * t_ac
    r == 0  iff all scales are mutually consistent.
    """
    return (
        rel_R[(b, c)] @ (s[(a, b)] * rel_t[(a, b)])
        + s[(b, c)] * rel_t[(b, c)]
        - s[(a, c)] * rel_t[(a, c)]
    )


def total_loss(x: np.ndarray) -> float:
    """Sum of squared residual norms across all 4 triplets."""
    s = build_s(x)
    total = 0.0
    for a, b, c in all_triplets:
        r = triplet_residual(s, a, b, c)
        total += float(np.dot(r, r))
    return total


def gradient(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of total_loss w.r.t. the free scales."""
    s = build_s(x)
    g = np.zeros(n_free)
    for a, b, c in all_triplets:
        r = triplet_residual(s, a, b, c)
        # ∂r/∂s_ab = R_bc @ t_ab,  ∂r/∂s_bc = t_bc,  ∂r/∂s_ac = −t_ac
        for pair, dr in [
            ((a, b), rel_R[(b, c)] @ rel_t[(a, b)]),
            ((b, c), rel_t[(b, c)]),
            ((a, c), -rel_t[(a, c)]),
        ]:
            if pair in pidx:
                g[pidx[pair]] += 2.0 * float(r @ dr)
    return g


# Compute the Hessian (constant for this quadratic loss) to pick a stable LR.
# H = 2 * J^T J  where J is the residual Jacobian w.r.t. x.
eps = 1e-6
x_ref = np.ones(n_free)
J_cols = []
for k in range(n_free):
    e = np.zeros(n_free)
    e[k] = eps
    r_p = np.concatenate(
        [triplet_residual(build_s(x_ref + e), a, b, c) for a, b, c in all_triplets]
    )
    r_m = np.concatenate(
        [triplet_residual(build_s(x_ref - e), a, b, c) for a, b, c in all_triplets]
    )
    J_cols.append((r_p - r_m) / (2 * eps))
J = np.column_stack(J_cols)
H = 2.0 * J.T @ J
pos_eigs = np.linalg.eigvalsh(H)
pos_eigs = pos_eigs[pos_eigs > 1e-10]
LR = 1.5 / pos_eigs.max() if len(pos_eigs) else 1e-3

# Run gradient descent and record the full trajectory.
x0 = np.array([unit_scales[p] for p in free_pairs])  # start: unit-norm scales
x = x0.copy()
trajectory = [x.copy()]
loss_curve = [total_loss(x)]

for _ in range(N_OPT_STEPS):
    x = x - LR * gradient(x)
    trajectory.append(x.copy())
    loss_curve.append(total_loss(x))

print(f"LR = {LR:.4e}")
print(f"Initial loss : {loss_curve[0]:.6f}")
print(f"Final loss   : {loss_curve[-1]:.6f}")


# ── Visualisation helpers ──────────────────────────────────────────────────────
def place_cam(cam_i_world: np.ndarray, i: int, j: int, s_ij: float) -> np.ndarray:
    """World position of cam j, placed from cam i's position at scale s_ij.

    Uses cam i's true rotation R_c2w[i] (orientations don't change with scale).
    """
    return R_c2w[i] @ (-rel_R[(i, j)].T @ (s_ij * rel_t[(i, j)])) + cam_i_world


# ── Rerun init ────────────────────────────────────────────────────────────────
rr.init("4cam_loop_closure_ls")
rr.spawn()
rr.log("/", rr.ViewCoordinates.BLU, static=True)


# ── Helper: log one triplet chain at full opacity ────────────────────────────
def log_triplet_chain(
    a, b, c, cam_pos, s, pop_scale=1.0, chain_phase=1.0, thickness_scale=1.0
):
    """
    Logs a triplet. Real cameras and the ghost camera pop in together.
    Lines and gaps appear during the chain_phase.
    """
    # Calculate ghost_c anchored at real cam_b
    ghost_c = place_cam(cam_pos[b], b, c, s[(b, c)])
    direct_c = cam_pos[c]
    tc = TRIPLET_COLORS[(a, b, c)]
    ipd = image_plane_dist * pop_scale

    # 1. Log ALL cameras (3 real + 1 ghost) using pop_scale
    camera_targets = [
        (f"cam_{a}", cam_pos[a], a),
        (f"cam_{b}", cam_pos[b], b),
        (f"cam_{c}", cam_pos[c], c),
        (f"ghost_c", ghost_c, c),  # The ghost is a version of camera 'c'
    ]

    for name, world_pos, k_idx in camera_targets:
        path = f"triplets/{a}{b}{c}/{name}"
        rr.log(
            path,
            rr.Transform3D(translation=world_pos, mat3x3=R_c2w[k_idx]),
            rr.TransformAxes3D(axis_length=0.05),
        )
        rr.log(
            path,
            rr.Pinhole(
                image_from_camera=K[k_idx],
                width=res_hw[k_idx][1],
                height=res_hw[k_idx][0],
                image_plane_distance=ipd,
                color=tc,
            ),
        )

    # 2. Log Chains and Gaps only when the chain_phase starts
    if chain_phase > 0:
        line_alpha = ease_pop(min(chain_phase, 1.0))
        line_radius = LINE_RADIUS * line_alpha * thickness_scale
        gap_radius = GAP_RADIUS * line_alpha * thickness_scale

        # Chain lines: a->b, b->ghost_c, and the reference a->c
        for name, p0, p1 in [
            (f"triplets/{a}{b}{c}/chain_ca", cam_pos[a], cam_pos[c]),
            (f"triplets/{a}{b}{c}/chain_ab", cam_pos[a], cam_pos[b]),
            (f"triplets/{a}{b}{c}/chain_bc", cam_pos[b], ghost_c),
        ]:
            rr.log(
                name,
                rr.LineStrips3D(
                    strips=[np.stack([p0, p1])], colors=[tc], radii=line_radius
                ),
            )

        # The Gap line (error vector)
        gap_path = f"triplets/{a}{b}{c}/gap"
        if np.linalg.norm(ghost_c - direct_c) > 1e-6:
            rr.log(
                gap_path,
                rr.LineStrips3D(
                    strips=make_dashed_strips(ghost_c, direct_c),
                    colors=[tc],
                    radii=gap_radius,
                ),
            )
        else:
            rr.log(gap_path, rr.LineStrips3D(strips=[], colors=[], radii=gap_radius))


# ── INTRO LOOP: reveal triplets one by one ───────────────────────────────────
s0 = build_s(trajectory[0])
anchor_0 = cam_centers[0]
cam_pos_init = {0: anchor_0}
for j in (1, 2, 3):
    cam_pos_init[j] = place_cam(anchor_0, 0, j, s0[(0, j)])

for frame_idx in range(n_intro):
    rr.set_time("frame", sequence=frame_idx)

    active_tidx = min(frame_idx // N_INTRO_PER_TRIPLET, len(all_triplets) - 1)

    for tidx, (a, b, c) in enumerate(all_triplets):
        t_start = tidx * N_INTRO_PER_TRIPLET
        local_f = frame_idx - t_start
        if local_f < 0:
            continue  # triplet not yet revealed

        if local_f < N_POP:
            pop_scale = ease_pop(local_f / (N_POP - 1))
            chain_phase = 0.0
        else:
            pop_scale = 1.0
            chain_phase = min((local_f - N_POP) / (N_INTRO_PER_TRIPLET - N_POP), 1.0)

        thickness = 1.0 if tidx == active_tidx else IDLE_THICKNESS_SCALE
        log_triplet_chain(
            a,
            b,
            c,
            cam_pos_init,
            s0,
            pop_scale=pop_scale,
            chain_phase=chain_phase,
            thickness_scale=thickness,
        )


# ── OPT LOOP: animate gradient descent ────────────────────────────────────────
for opt_frame_idx in range(n_total_opt):
    rr.set_time("frame", sequence=n_intro + opt_frame_idx)

    # Map opt_frame_idx to a fractional optimisation step, then interpolate.
    if opt_frame_idx < n_hold:
        s = build_s(trajectory[0])
        step = 0
        t = 0.0
    else:
        frac_step = anim_frame_to_frac_step(opt_frame_idx - n_hold)
        step_lo = int(frac_step)
        step_hi = min(step_lo + 1, N_OPT_STEPS)
        t = frac_step - step_lo
        x_interp = (1 - t) * trajectory[step_lo] + t * trajectory[step_hi]
        s = build_s(x_interp)
        step = step_lo

    # Camera placement via star spanning tree from cam_0
    anchor_0 = cam_centers[0]
    cam_pos = {0: anchor_0}
    for j in (1, 2, 3):
        cam_pos[j] = place_cam(anchor_0, 0, j, s[(0, j)])

    for a, b, c in all_triplets:
        log_triplet_chain(a, b, c, cam_pos, s, chain_phase=1.0)

    # Per-triplet losses + total (interpolated for smooth plots)
    interp = lambda lo, hi: (1 - t) * lo + t * hi
    for a, b, c in all_triplets:
        r_lo = triplet_residual(build_s(trajectory[step]), a, b, c)
        r_hi = triplet_residual(
            build_s(trajectory[min(step + 1, N_OPT_STEPS)]), a, b, c
        )
        rr.log(
            f"opt/loss_{a}{b}{c}",
            rr.Scalars(interp(float(np.dot(r_lo, r_lo)), float(np.dot(r_hi, r_hi)))),
        )
        rr.log(f"opt/loss_{a}{b}{c}", rr.SeriesLines(widths=2))
    rr.log(
        "opt/total_loss",
        rr.Scalars(interp(loss_curve[step], loss_curve[min(step + 1, N_OPT_STEPS)])),
    )
    rr.log("opt/total_loss", rr.SeriesLines(widths=2))

print(f"Done — {n_total} frames total.")
