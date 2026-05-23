#!/usr/bin/env python3
"""
Replay an aligned reference motion NPZ in MuJoCo to visually sanity-check
the result of `align_npz_to_lab.py`.

Why this exists
---------------
The deploy-side `sub8_45_tag_history` policy expects the reference NPZ to
already be in the **lab frame** that the multicam AprilTag tracker reports
(pupil_apriltags floor-tag frame -> +z = DOWN). `align_npz_to_lab.py`
applies `R_flip = diag(1, -1, -1)` to do that conversion on top of a
yaw-only alignment. If alignment ever silently produces a 180°-flipped
torso (the same class of bug that caused the 2026-05-23 deploy crash), it
is much easier to catch by eye in MuJoCo than by reading numbers.

Two viewing modes
-----------------
* `--frame sim` (default): apply the *inverse* of `R_flip` so the motion
  is rendered in a natural z-up world. The robot stands upright on a
  visible floor and grasps a box. Use this to verify limb/joint trajectories
  and gross object motion look right.
* `--frame lab`: render the data **exactly as it is in the npz**, i.e. in
  lab frame (z-down). MuJoCo's viewer is z-up, so the model will look
  upside down. This is intentional and confirms the R_flip was applied.
  Use this to confirm the deploy-time policy (which lives entirely in lab
  frame after our motion_anchor_ori_b dual-mode fix) sees a sensible
  orientation.

NPZ schema (verified for sub8_45_extended_coords_processed_v2.npz):
    fps              (1,)
    joint_pos        (T, 29)        joints in mjlab g1.xml depth-first order
    joint_vel        (T, 29)
    body_pos_w       (T, 30, 3)     body[0] is pelvis (= floating base)
    body_quat_w      (T, 30, 4)     (w, x, y, z)
    object_pos_w     (T, 3)
    object_quat_w    (T, 4)         (w, x, y, z)
    + lin/ang velocities + contact_mask (unused for visualization)

Usage
-----
    python visualize_aligned_npz_mujoco.py \
        --npz outputs/sub8_45_extended_coords_processed_v2.npz \
        --frame sim                       # default
        # add --no-loop to play once

    python visualize_aligned_npz_mujoco.py \
        --npz outputs/sub8_45_extended_coords_processed_v2.npz \
        --frame lab                       # see what the policy sees

Requires the `unitree_rl_mjlab` conda env (mjlab + mujoco>=3.0).
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from mjlab import MJLAB_SRC_PATH


G1_XML_PATH = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
).resolve()


def build_scene_model(
    g1_xml_path: Path,
    box_size_xyz: tuple[float, float, float],
    floor_z: float = 0.0,
) -> mujoco.MjModel:
    """Build the visualization scene programmatically via MjSpec.

    We load mjlab's g1.xml *as a spec* (so its assets/meshdir resolve
    against its own directory) and then *append* a floor, world-axis
    sites, and a free-floating box body to the same spec before
    compiling. Writing a wrapper XML and using `<include>` doesn't work
    here because `from_xml_string` resolves the included file's
    `meshdir="assets"` relative to the calling cwd, not the included
    file's parent dir, so STL mesh loading fails.
    """
    bx, by, bz = box_size_xyz

    spec = mujoco.MjSpec.from_file(str(g1_xml_path))

    # Floor texture/material + box material.
    tex = spec.add_texture()
    tex.name = "floor_grid"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1 = [0.1, 0.2, 0.3]
    tex.rgb2 = [0.2, 0.3, 0.4]
    tex.width = 300
    tex.height = 300
    tex.mark = mujoco.mjtMark.mjMARK_EDGE
    tex.markrgb = [0.2, 0.3, 0.4]

    mat_floor = spec.add_material()
    mat_floor.name = "floor_grid"
    mat_floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_grid"
    mat_floor.texrepeat = [6, 6]
    mat_floor.texuniform = True
    mat_floor.reflectance = 0.0

    mat_box = spec.add_material()
    mat_box.name = "box_mat"
    mat_box.rgba = [0.85, 0.55, 0.20, 1.0]

    # Add to the global worldbody.
    world = spec.worldbody

    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.pos = [0, 0, floor_z]
    floor.material = "floor_grid"
    floor.contype = 0
    floor.conaffinity = 0

    def add_axis_site(name, pos, size, rgba):
        s = world.add_site()
        s.name = name
        s.type = mujoco.mjtGeom.mjGEOM_BOX
        s.pos = pos
        s.size = size
        s.rgba = rgba

    add_axis_site("origin_x", [0.1, 0, 0], [0.1, 0.005, 0.005], [1, 0, 0, 1])
    add_axis_site("origin_y", [0, 0.1, 0], [0.005, 0.1, 0.005], [0, 1, 0, 1])
    add_axis_site("origin_z", [0, 0, 0.1], [0.005, 0.005, 0.1], [0, 0, 1, 1])

    box_body = world.add_body()
    box_body.name = "ref_box"
    box_body.pos = [0, 0, 0.5]
    box_joint = box_body.add_freejoint()
    box_joint.name = "ref_box_joint"
    box_geom = box_body.add_geom()
    box_geom.name = "ref_box_geom"
    box_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    box_geom.size = [bx / 2, by / 2, bz / 2]
    box_geom.material = "box_mat"
    box_geom.contype = 0
    box_geom.conaffinity = 0
    # Tiny axes on the box for orientation cue.
    for n, p, sz, c in [
        ("box_origin_x", [bx / 2 + 0.02, 0, 0], [0.04, 0.003, 0.003], [1, 0, 0, 1]),
        ("box_origin_y", [0, by / 2 + 0.02, 0], [0.003, 0.04, 0.003], [0, 1, 0, 1]),
        ("box_origin_z", [0, 0, bz / 2 + 0.02], [0.003, 0.003, 0.04], [0, 0, 1, 1]),
    ]:
        s = box_body.add_site()
        s.name = n
        s.type = mujoco.mjtGeom.mjGEOM_BOX
        s.pos = p
        s.size = sz
        s.rgba = c

    # Override gravity to z-up (default is already z-up).
    spec.option.gravity = [0, 0, -9.81]

    return spec.compile()


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, both inputs (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def make_unflip_transform() -> tuple[np.ndarray, np.ndarray]:
    """Return (R, q) that undoes R_flip = diag(1, -1, -1) used by
    align_npz_to_lab.py to convert sim z-up -> lab z-down. Applying this
    to a lab-frame pose recovers the sim-frame pose.

    R_flip is its own inverse, so R_unflip = diag(1, -1, -1).
    The corresponding quaternion (w, x, y, z) for diag(1,-1,-1) — which
    is a 180° rotation about the +x axis — is (0, 1, 0, 0).
    """
    R = np.diag([1.0, -1.0, -1.0])
    q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    return R, q


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path,
                    help="Aligned NPZ from align_npz_to_lab.py "
                         "(e.g. outputs/sub8_45_extended_coords_processed_v2.npz)")
    ap.add_argument("--frame", choices=["sim", "lab"], default="sim",
                    help="sim: undo R_flip and render in z-up (natural view, default). "
                         "lab: render data as-is (z-down -> robot looks upside down "
                         "in MuJoCo's z-up viewer; this is what the policy sees).")
    ap.add_argument("--box-size", type=float, nargs=3,
                    metavar=("LX", "LY", "LZ"), default=[0.45, 0.30, 0.30],
                    help="Visual box size in metres (default 0.45 0.30 0.30, "
                         "approx sub8_largebox_045).")
    ap.add_argument("--floor-z", type=str, default="auto",
                    help="Floor plane z (sim frame). 'auto' (default) places "
                         "the floor at the average ankle_roll z minus the "
                         "ankle->sole offset (-0.037), so the rendered foot "
                         "touches it. Pass a number to override (e.g. 0 to "
                         "place the floor at the lab origin-tag plane and see "
                         "the actual offset between tag plane and feet).")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier (1.0 = real-time).")
    ap.add_argument("--no-loop", action="store_true",
                    help="Play once and exit instead of looping.")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--g1-xml", type=Path, default=G1_XML_PATH,
                    help=f"Path to mjlab's g1.xml (default {G1_XML_PATH})")
    args = ap.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(args.npz)
    if not args.g1_xml.exists():
        raise FileNotFoundError(args.g1_xml)

    # ---------- Load aligned NPZ ----------
    ref = np.load(args.npz, allow_pickle=True)
    fps = float(ref["fps"].flat[0])
    joint_pos   = np.asarray(ref["joint_pos"],   dtype=np.float64)   # (T, 29)
    body_pos_w  = np.asarray(ref["body_pos_w"],  dtype=np.float64)   # (T, 30, 3)
    body_quat_w = np.asarray(ref["body_quat_w"], dtype=np.float64)   # (T, 30, 4)
    obj_pos_w   = np.asarray(ref["object_pos_w"],  dtype=np.float64) # (T, 3)
    obj_quat_w  = np.asarray(ref["object_quat_w"], dtype=np.float64) # (T, 4)
    T = joint_pos.shape[0]
    print(f"[npz] T={T} fps={fps:.1f} joints={joint_pos.shape[1]} "
          f"bodies={body_pos_w.shape[1]} object_present=True")

    # ---------- Decide floor z ----------
    # In the npz lab frame z=DOWN, ankle_roll bodies are at z ≈ +0.04 (i.e. a
    # few cm below the floor tag plane). Body indices follow mjlab's
    # depth-first order: pelvis=0, left_ankle_roll=6, right_ankle_roll=12.
    # The site `left_foot` is at pos="0.04 0 -0.037" inside ankle_roll, so the
    # actual sole is ~3.7 cm below ankle_roll along the body's local -z.
    # Approximation: foot_sole_z_lab ≈ ankle_roll_z_lab + 0.037 (z-down).
    L_FOOT_BODY, R_FOOT_BODY, ANKLE_TO_SOLE = 6, 12, 0.037

    avg_ankle_z_lab = float(0.5 * (
        np.median(body_pos_w[:, L_FOOT_BODY, 2]) +
        np.median(body_pos_w[:, R_FOOT_BODY, 2])
    ))
    sole_z_lab = avg_ankle_z_lab + ANKLE_TO_SOLE

    if args.frame == "sim":
        # sim z = -lab z, so soles end up at -sole_z_lab (negative number).
        sole_z_view = -sole_z_lab
    else:
        sole_z_view = sole_z_lab

    if args.floor_z == "auto":
        floor_z_view = sole_z_view
        print(f"[floor] auto: median ankle z (lab) = {avg_ankle_z_lab:+.4f},  "
              f"sole z ({args.frame}) = {sole_z_view:+.4f}  ->  floor placed there")
        # Also report the gap between origin-tag plane (z=0 in lab) and the
        # actual floor inferred from feet, so the user can sanity-check
        # calibration. Healthy values: 0-5 cm. Larger -> probably a bad
        # head→torso z calibration or a thick mat under the floor tag.
        print(f"[floor] origin-tag → sole offset = {sole_z_lab:+.4f} m  "
              f"(POSITIVE in z-down lab means soles are BELOW the tag plane)")
    else:
        floor_z_view = float(args.floor_z)
        print(f"[floor] manual override: floor at z={floor_z_view:+.4f} ({args.frame})")

    # ---------- Build scene + compile ----------
    model = build_scene_model(args.g1_xml, tuple(args.box_size), floor_z=floor_z_view)
    data = mujoco.MjData(model)
    print(f"[mj] nq={model.nq} nv={model.nv} nbody={model.nbody}")

    # Sanity: 29 robot joints + 7 robot freejoint + 7 box freejoint = 43.
    expected_nq = 7 + 29 + 7
    assert model.nq == expected_nq, (
        f"unexpected nq={model.nq} (expected {expected_nq}). Did the wrapper "
        f"XML match the included g1.xml?")

    # Find qpos slices. MuJoCo lays out qpos in joint order. The wrapper
    # XML defines `ref_box_joint` (freejoint, 7 dofs) BEFORE the included
    # g1.xml worldbody (which has its own freejoint + 29 hinges). Resolve
    # joint qpos addresses directly to avoid hard-coding offsets.
    box_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ref_box_joint")
    g1_freejoint_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                         "floating_base_joint")
    assert box_jid != -1 and g1_freejoint_jid != -1, "couldn't find expected joints"
    box_qpos_adr = model.jnt_qposadr[box_jid]                  # 7 dofs starting here
    pelvis_qpos_adr = model.jnt_qposadr[g1_freejoint_jid]      # 7 dofs starting here

    # The 29 robot hinges are everything except the freejoint. Build an
    # ordered list of their qpos addresses *as they appear in the XML*
    # (= the natural index order assumed by mjlab npz).
    hinge_qpos_addrs = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        hinge_qpos_addrs.append(model.jnt_qposadr[j])
    hinge_qpos_addrs = np.asarray(hinge_qpos_addrs, dtype=np.int32)
    assert hinge_qpos_addrs.shape[0] == joint_pos.shape[1], (
        f"npz has {joint_pos.shape[1]} joints but model has "
        f"{hinge_qpos_addrs.shape[0]} hinge joints; order/count mismatch.")

    # ---------- Frame transform ----------
    R, q_unflip = make_unflip_transform()
    if args.frame == "sim":
        def xform_pos(p):
            return p @ R.T          # equivalent to R @ p column-wise
        def xform_quat(q):
            # left-multiply by q_unflip so that the rotation appears as
            # if R_flip had never been applied. quaternion order (w,x,y,z).
            return quat_mul(q_unflip, q)
    else:  # 'lab'
        def xform_pos(p):
            return p
        def xform_quat(q):
            return q

    # ---------- Helper for per-frame state ----------
    def set_frame(t: int):
        # Pelvis (= body index 0 in npz, also the freejoint root)
        data.qpos[pelvis_qpos_adr + 0 : pelvis_qpos_adr + 3] = xform_pos(body_pos_w[t, 0])
        data.qpos[pelvis_qpos_adr + 3 : pelvis_qpos_adr + 7] = xform_quat(body_quat_w[t, 0])
        # 29 hinge joints
        data.qpos[hinge_qpos_addrs] = joint_pos[t]
        # Box freejoint
        data.qpos[box_qpos_adr + 0 : box_qpos_adr + 3] = xform_pos(obj_pos_w[t])
        data.qpos[box_qpos_adr + 3 : box_qpos_adr + 7] = xform_quat(obj_quat_w[t])
        # mj_kinematics is enough for visualization; mj_forward also OK
        mujoco.mj_kinematics(model, data)

    # ---------- Launch viewer + playback loop ----------
    print(f"[viewer] frame={args.frame}  speed={args.speed}x  loop={not args.no_loop}")
    print(f"[viewer] press SPACE to pause, ESC to exit, drag to orbit")
    if args.frame == "lab":
        print("[viewer] NOTE: in 'lab' mode the robot is rendered upside down.")
        print("              Floor is z=0; the npz has the robot below z=0 because")
        print("              its local 'down' is +z but MuJoCo gravity is -z.")

    dt_frame = 1.0 / max(fps * args.speed, 1e-6)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_start = time.perf_counter()
            for t in range(args.start_frame, T):
                if not viewer.is_running():
                    break
                set_frame(t)
                viewer.sync()
                # Pace playback to fps. perf_counter-based to avoid drift.
                target = t_start + (t - args.start_frame + 1) * dt_frame
                sleep = target - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
            if args.no_loop:
                break
            # Brief pause at end of clip so it's obvious we're looping
            time.sleep(0.4)


if __name__ == "__main__":
    main()
