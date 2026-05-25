"""runtime_alignment.py
Helpers for the runtime lab<->sim alignment used by the camera publisher.

The classic offline pipeline (`align_npz_to_lab.py` + aligned NPZ + lab-frame
camera pose) is replaced here by a live, on-button-press alignment:

  1. Load the *raw* NPZ (sim z-up frame, mjlab body convention).
  2. At the SPACE-press moment, snapshot the current real torso pose in the
     lab frame and compute T_sim_lab — the rigid (yaw-only by default)
     transform that maps lab-frame poses to the NPZ's sim frame.
  3. From then on, transform every camera-derived torso/box pose into sim
     frame, and compute the 6 tag-history observations in sim frame.

The torso quaternion published by `track_robot_and_box_multicam.py` follows
the head-tag's body convention (+x fwd, +y right, +z down). The NPZ uses
mjlab convention (+x fwd, +y left, +z up). They differ by a 180-degree
rotation about the body's local +x axis, encoded as R_BODY_FLIP =
diag(1, -1, -1) (right-multiplied to apply in the body's local frame).
This flip is applied to the torso (NOT the box, which the tracker already
emits in mjlab AABB body frame after the inertial-quat correction).

All math is plain numpy so the publisher process can stay light-weight and
the same code can be reused by the smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Body-frame conventions
# ---------------------------------------------------------------------------
# 180 degrees about the body's local +x axis. Right-multiplied onto a
# rotation matrix to convert head-tag z-down body conv -> mjlab z-up body
# conv. Equivalent quaternion (wxyz) = (0, 1, 0, 0). This MUST match the
# C++ Q_BODY_FLIP_ZDOWN_TO_ZUP that used to live in State_Mimic.cpp.
R_BODY_FLIP = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (wxyz layout)
# ---------------------------------------------------------------------------
def quat_wxyz_to_R(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


def R_to_quat_wxyz(R):
    q = np.empty(4, dtype=float)
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        q /= n
    return q


def rotation_to_rot6d(R):
    """Zhou et al. 2019 6D rotation: first two columns of R, column-major.

    Layout matches the IsaacLab / mjlab rot6d convention used by the
    tag-history policy: [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]].
    """
    return np.array([R[0, 0], R[1, 0], R[2, 0],
                     R[0, 1], R[1, 1], R[2, 1]], dtype=float)


# ---------------------------------------------------------------------------
# Reference-motion loader
# ---------------------------------------------------------------------------
@dataclass
class RefFrame:
    torso_pos: np.ndarray   # (3,) sim world
    torso_R:   np.ndarray   # (3, 3) sim world, mjlab body conv
    torso_quat: np.ndarray  # (4,) wxyz, sim world
    object_pos:  np.ndarray
    object_R:    np.ndarray
    object_quat: np.ndarray
    joint_pos: np.ndarray   # (dof,)
    joint_vel: np.ndarray   # (dof,)
    # Alignment-body (typically pelvis = body 0) pose at this frame. Used
    # only by the SPACE-press T_sim_lab fit, NOT by the actor obs.
    align_pos:  np.ndarray
    align_R:    np.ndarray
    align_quat: np.ndarray


class MotionLoader:
    """Owns the reference NPZ. Indexed at 50 Hz by the publisher's ref ticker.

    Mirrors the indexing in `unitree_rl_mjlab/.../State_Mimic.h::MotionLoader_`:
      * anchor_body_idx (default 15) = torso_link in the G1 30-body depth-first
        order. This is the body whose frame the actor obs are expressed in.
      * align_body_idx  (default 0)  = pelvis (root). Used only for fitting
        T_sim_lab at SPACE-press time (pelvis is approximately vertical in
        both lab FixStand and NPZ frame 0, so its yaw is a robust anchor).
    """

    def __init__(self, motion_file: str, anchor_body_idx: int = 15,
                 align_body_idx: int = 0, fps: float = 50.0):
        path = Path(motion_file)
        if not path.exists():
            raise FileNotFoundError(f"motion_file does not exist: {path}")
        data = np.load(path, allow_pickle=False)

        body_pos_w  = np.asarray(data["body_pos_w"],  dtype=float)   # [T, B, 3]
        body_quat_w = np.asarray(data["body_quat_w"], dtype=float)   # [T, B, 4] wxyz
        if anchor_body_idx >= body_pos_w.shape[1]:
            raise ValueError(
                f"anchor_body_idx={anchor_body_idx} but NPZ has only "
                f"{body_pos_w.shape[1]} bodies"
            )

        if align_body_idx >= body_pos_w.shape[1] or align_body_idx < 0:
            raise ValueError(
                f"align_body_idx={align_body_idx} but NPZ has only "
                f"{body_pos_w.shape[1]} bodies"
            )

        self.path = str(path)
        self.anchor_body_idx = int(anchor_body_idx)
        self.align_body_idx  = int(align_body_idx)
        self.num_frames = int(body_pos_w.shape[0])
        self.fps = float(fps)
        self.dt = 1.0 / self.fps

        self._torso_pos  = body_pos_w[:, anchor_body_idx, :].copy()
        self._torso_quat = body_quat_w[:, anchor_body_idx, :].copy()
        self._align_pos  = body_pos_w[:, align_body_idx,  :].copy()
        self._align_quat = body_quat_w[:, align_body_idx,  :].copy()

        self._has_object = ("object_pos_w" in data.files
                            and "object_quat_w" in data.files)
        if self._has_object:
            self._object_pos  = np.asarray(data["object_pos_w"],  dtype=float)
            self._object_quat = np.asarray(data["object_quat_w"], dtype=float)
        else:
            self._object_pos  = np.zeros((self.num_frames, 3), dtype=float)
            self._object_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]),
                                        (self.num_frames, 1))

        if "joint_pos" not in data.files or "joint_vel" not in data.files:
            raise KeyError("NPZ missing joint_pos/joint_vel — required by motion_command")
        self._joint_pos = np.asarray(data["joint_pos"], dtype=float)
        self._joint_vel = np.asarray(data["joint_vel"], dtype=float)
        self.dof = int(self._joint_pos.shape[1])

        # Pre-cache rotation matrices for the full trajectory (cheap).
        self._torso_R  = np.stack([quat_wxyz_to_R(q) for q in self._torso_quat],  axis=0)
        self._object_R = np.stack([quat_wxyz_to_R(q) for q in self._object_quat], axis=0)
        self._align_R  = np.stack([quat_wxyz_to_R(q) for q in self._align_quat],  axis=0)

    @property
    def has_object(self) -> bool:
        return self._has_object

    def at(self, frame_idx: int) -> RefFrame:
        i = max(0, min(int(frame_idx), self.num_frames - 1))
        return RefFrame(
            torso_pos  = self._torso_pos[i].copy(),
            torso_R    = self._torso_R[i].copy(),
            torso_quat = self._torso_quat[i].copy(),
            object_pos  = self._object_pos[i].copy(),
            object_R    = self._object_R[i].copy(),
            object_quat = self._object_quat[i].copy(),
            joint_pos = self._joint_pos[i].copy(),
            joint_vel = self._joint_vel[i].copy(),
            align_pos  = self._align_pos[i].copy(),
            align_R    = self._align_R[i].copy(),
            align_quat = self._align_quat[i].copy(),
        )


# ---------------------------------------------------------------------------
# T_sim_lab computation (yaw-only by default, full-rotation optional)
# ---------------------------------------------------------------------------
_FWD_AXIS_MAP = {
    "+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0]),
}


def _parse_fwd_axis(s: str) -> np.ndarray:
    s = s.strip().lower()
    if not s:
        s = "+x"
    if s[0] not in "+-":
        s = "+" + s
    if s not in _FWD_AXIS_MAP:
        raise ValueError(f"bad forward axis '{s}'")
    return _FWD_AXIS_MAP[s]


def compute_T_sim_lab(
    real_torso_pos_lab: np.ndarray,
    real_torso_R_lab_raw: np.ndarray,
    ref_torso_pos_sim: np.ndarray,
    ref_torso_R_sim: np.ndarray,
    mode: str = "yaw-only",
    torso_forward_axis: str = "+x",
):
    """Compute the 4x4 T_sim_lab that maps a lab-frame pose to the NPZ's sim frame.

    Inputs
      real_torso_pos_lab     : (3,)   torso position from camera tracker (lab world).
      real_torso_R_lab_raw   : (3,3) torso rotation from camera tracker, in lab
                                world coords with head-tag z-down body convention
                                (NOT yet body-flipped).
      ref_torso_pos_sim      : (3,)   NPZ torso position at the anchor frame.
      ref_torso_R_sim        : (3,3) NPZ torso rotation at the anchor frame
                                (mjlab z-up body convention).
      mode                   : "yaw-only" (default, gravity-preserving) or
                                "full-rotation" (matches roll/pitch too — DANGEROUS).
      torso_forward_axis     : robot forward in torso local frame (G1 = "+x").

    Returns
      T_sim_lab : (4,4) such that  pose_sim = T_sim_lab @ pose_lab
                  for any homogeneous lab-frame pose. The body-frame flip
                  (R_BODY_FLIP) is NOT included — call `transform_pose_lab_to_sim`
                  with body_flip=True for the torso to apply it.

    Notes
      The math is the inverse of `align_npz_to_lab.py::T_lab_world`. Derivation:
        align_npz_to_lab.py builds T_lab_world such that
            T_lab_world @ npz_pose_sim_zup == csv_pose_lab_zup
        where csv_pose_lab_zup = csv_pose_lab_raw with body conv flipped
        to z-up. So
            T_sim_lab := inv(T_lab_world)
            R_sim_lab = R_lab_world.T,   t_sim_lab = -R_sim_lab @ t_lab_world.

      The yaw-only branch decomposes R_lab_world = R_yaw_lab @ R_flip with
        R_flip = diag(1,-1,-1)         (sim z-up world -> lab z-down world)
        R_yaw_lab = rotation about lab z by (yaw_csv - yaw_npz),
      preserving gravity.
    """
    real_R_lab_zup = real_torso_R_lab_raw @ R_BODY_FLIP   # body conv flip
    fwd_local = _parse_fwd_axis(torso_forward_axis)

    R_flip = np.diag([1.0, -1.0, -1.0])

    if mode == "full-rotation":
        # Full 6-DoF: R_lab_world = real_R_lab_zup @ ref_R_sim.T
        R_lab_world = real_R_lab_zup @ ref_torso_R_sim.T
        delta_yaw = float("nan")
    elif mode == "yaw-only":
        fwd_npz_in_lab = R_flip @ ref_torso_R_sim @ fwd_local
        fwd_csv_in_lab = real_R_lab_zup @ fwd_local
        yaw_npz = float(np.arctan2(fwd_npz_in_lab[1], fwd_npz_in_lab[0]))
        yaw_csv = float(np.arctan2(fwd_csv_in_lab[1], fwd_csv_in_lab[0]))
        delta_yaw = yaw_csv - yaw_npz
        c, s = np.cos(delta_yaw), np.sin(delta_yaw)
        R_yaw = np.array([[c, -s, 0.0],
                          [s,  c, 0.0],
                          [0.0, 0.0, 1.0]])
        R_lab_world = R_yaw @ R_flip
    else:
        raise ValueError(f"unknown align mode '{mode}'")

    t_lab_world_vec = real_torso_pos_lab - R_lab_world @ ref_torso_pos_sim

    R_sim_lab = R_lab_world.T
    t_sim_lab = -R_sim_lab @ t_lab_world_vec

    T_sim_lab = np.eye(4)
    T_sim_lab[:3, :3] = R_sim_lab
    T_sim_lab[:3, 3]  = t_sim_lab

    T_lab_world_mat = np.eye(4)
    T_lab_world_mat[:3, :3] = R_lab_world
    T_lab_world_mat[:3, 3]  = t_lab_world_vec

    diag = {
        "mode": mode,
        "delta_yaw_rad": float(delta_yaw) if mode == "yaw-only" else None,
        "T_lab_world": T_lab_world_mat.tolist(),
        "T_sim_lab": T_sim_lab.tolist(),
    }
    return T_sim_lab, diag


# ---------------------------------------------------------------------------
# Pelvis-based T_sim_lab (yaw-only) — used by the runtime publisher's
# 3-second SPACE-press averaging routine.
#
# Why pelvis instead of torso:
#   At robot FixStand, pelvis is approximately vertical (small roll/pitch)
#   AND the NPZ frame-0 pelvis is also approximately vertical (~4 deg tilt
#   in sub8_45_extended). The torso, by contrast, is bent ~25 deg forward
#   in NPZ frame 0 but vertical in FixStand — that mismatch makes a torso-
#   based T fit at SPACE time unsafe (you'd bake the FixStand-vs-bent-fwd
#   delta into a yaw of T_sim_lab). Pelvis sidesteps the issue entirely.
#
# Why yaw-only:
#   * gravity-preserving (no roll/pitch bias gets baked in),
#   * the small pelvis-vertical residual is below sensor noise after a
#     few-second average,
#   * matches the assumption that the lab and NPZ frames differ only in
#     yaw + translation (modulo the z-down vs z-up axis convention).
#
# Why no extra calibration:
#   We don't need a full "pelvis tag -> pelvis body" rigid offset. We just
#   need the lab-world horizontal direction the robot is FACING. That's
#   recovered from one tag-local axis (--pelvis-tag-fwd-axis) projected
#   onto the lab xy plane.
# ---------------------------------------------------------------------------
def compute_T_sim_lab_pelvis_yaw(
    real_pelvis_pos_lab: np.ndarray,
    real_pelvis_R_lab:   np.ndarray,
    ref_pelvis_pos_sim:  np.ndarray,
    ref_pelvis_R_sim:    np.ndarray,
    real_fwd_axis_tag_local: str = "-z",
    ref_fwd_axis_body_local: str = "+x",
):
    """yaw-only T_sim_lab from a (typically averaged) real pelvis tag pose
    in the lab and the NPZ pelvis pose in sim.

    Inputs
      real_pelvis_pos_lab     : (3,) averaged tag position in lab world.
      real_pelvis_R_lab       : (3,3) averaged tag rotation matrix, columns =
                                tag-local axes expressed in lab world coords.
      ref_pelvis_pos_sim      : (3,) NPZ body 0 position at the alignment frame.
      ref_pelvis_R_sim        : (3,3) NPZ body 0 rotation, mjlab body conv.
      real_fwd_axis_tag_local : which pelvis-tag-local axis points toward the
                                robot's chest (i.e., body forward). For a
                                back-mounted G1 pelvis tag with
                                --pelvis-tag-up-axis=-y, body forward = -z_tag.
      ref_fwd_axis_body_local : robot forward in mjlab body coords (G1 = +x).

    Returns
      T_sim_lab (4x4 numpy)    : pose_sim = T_sim_lab @ pose_lab
                                 (apply unconditionally; no body flip needed
                                 here because pelvis tag conv is encoded in
                                 real_fwd_axis_tag_local — for the *torso*
                                 part of the obs pipeline you still have to
                                 apply R_BODY_FLIP via transform_pose_lab_to_sim
                                 with body_flip=True, since the torso is
                                 measured by the head tag whose conv is
                                 z-down body-frame).
      diag (dict)              : T_sim_lab and delta_yaw for logging/tests.
    """
    fwd_real_local = _parse_fwd_axis(real_fwd_axis_tag_local)
    fwd_ref_local  = _parse_fwd_axis(ref_fwd_axis_body_local)
    R_flip = np.diag([1.0, -1.0, -1.0])

    fwd_ref_in_lab  = R_flip @ ref_pelvis_R_sim @ fwd_ref_local
    fwd_real_in_lab = real_pelvis_R_lab @ fwd_real_local

    yaw_ref  = float(np.arctan2(fwd_ref_in_lab[1],  fwd_ref_in_lab[0]))
    yaw_real = float(np.arctan2(fwd_real_in_lab[1], fwd_real_in_lab[0]))
    delta_yaw = yaw_real - yaw_ref

    c, s = np.cos(delta_yaw), np.sin(delta_yaw)
    R_yaw = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]])
    R_lab_world = R_yaw @ R_flip
    t_lab_world = real_pelvis_pos_lab - R_lab_world @ ref_pelvis_pos_sim

    R_sim_lab = R_lab_world.T
    t_sim_lab = -R_sim_lab @ t_lab_world

    T_sim_lab = np.eye(4)
    T_sim_lab[:3, :3] = R_sim_lab
    T_sim_lab[:3, 3]  = t_sim_lab

    T_lab_world_mat = np.eye(4)
    T_lab_world_mat[:3, :3] = R_lab_world
    T_lab_world_mat[:3, 3]  = t_lab_world

    diag = {
        "mode": "pelvis-yaw-only",
        "delta_yaw_rad": float(delta_yaw),
        "delta_yaw_deg": float(np.degrees(delta_yaw)),
        "T_lab_world": T_lab_world_mat.tolist(),
        "T_sim_lab":   T_sim_lab.tolist(),
        "real_pelvis_pos_lab": np.asarray(real_pelvis_pos_lab).tolist(),
        "ref_pelvis_pos_sim":  np.asarray(ref_pelvis_pos_sim).tolist(),
    }
    return T_sim_lab, diag


# ---------------------------------------------------------------------------
# Step 1: hardcoded tag-to-link pose transforms
# ---------------------------------------------------------------------------
# These constants encode the assumed mounting of each AprilTag on the G1 body.
# They are INITIAL GUESSES; tune by overlaying the resulting link frame on the
# live camera GUI in FixStand pose. See step 1 of sub8_45_sim2real plan.
#
# AprilTag detection convention (pupil_apriltags + OpenCV):
#   tag local +X = tag image right
#   tag local +Y = tag image down
#   tag local +Z = into the tag surface (away from the camera)
#
# mjlab G1 body convention:
#   body local +X = forward
#   body local +Y = left
#   body local +Z = up
#
# A R_tag_to_body matrix has its COLUMNS = body axes expressed in tag frame.
#
# Pelvis FRONT tag (tag id 8, mounted flat on the front of the pelvis/belly).
# Verified empirically on 2026-05-25 by user inspection of the live RGB axes
# on the camera GUI in FixStand pose:
#   tag +X (R red, image right)        -> robot left  = body +Y
#   tag +Y (G green, image down)       -> robot down  = body -Z
#   tag +Z (B blue, into tag surface)  -> robot back  = body -X
#       (tag surface faces forward away from robot; +Z goes into the body)
#
# R_tag_to_body columns = body axes expressed in tag frame:
#   col 0 (body +X = robot forward) = tag (-Z) = (0,  0, -1)
#   col 1 (body +Y = robot left)    = tag (+X) = (1,  0,  0)
#   col 2 (body +Z = robot up)      = tag (-Y) = (0, -1,  0)
PELVIS_FRONT_TAG_TO_BODY = np.array([
    [ 0.0,  1.0,  0.0],
    [ 0.0,  0.0, -1.0],
    [-1.0,  0.0,  0.0],
])

# Pelvis link (root body) origin offset from tag 8 center, in BODY local frame.
# Empirically tuned 2026-05-25:
#   +5 cm along body +Z (up)   -> 5 cm above the tag
#   -7 cm along body +X (fwd)  -> 7 cm INTO the body (tag is mounted on the
#                                  front surface; link origin sits inside the
#                                  pelvis ~spine center).
PELVIS_LINK_OFFSET_IN_BODY = np.array([-0.07, 0.0, 0.05])


def tag_to_link_pose(
    T_lab_tag: np.ndarray,
    R_tag_to_body: np.ndarray,
    link_offset_in_body: np.ndarray,
) -> np.ndarray:
    """Hardcoded transform: tag pose (lab frame) -> link body pose (lab frame).

    Args:
        T_lab_tag: (4,4) SE3, tag pose in lab world frame.
        R_tag_to_body: (3,3) rotation matrix whose COLUMNS are the body local
            axes expressed in tag local frame. Encodes the tag's mounting
            orientation on the link.
        link_offset_in_body: (3,) translation from tag center to link origin,
            expressed in BODY local frame.

    Returns:
        T_lab_link: (4,4) SE3, link body pose in lab world frame.
    """
    R_tag_lab = T_lab_tag[:3, :3]
    p_tag_lab = T_lab_tag[:3, 3]
    R_body_lab = R_tag_lab @ R_tag_to_body
    p_link_lab = p_tag_lab + R_body_lab @ link_offset_in_body
    T_lab_link = np.eye(4)
    T_lab_link[:3, :3] = R_body_lab
    T_lab_link[:3, 3] = p_link_lab
    return T_lab_link


def pelvis_front_tag_to_pelvis_link(T_lab_pelvis_tag: np.ndarray) -> np.ndarray:
    """Convenience: pelvis front tag (id 8, in lab frame) -> pelvis link pose (lab frame).

    Uses module-level PELVIS_FRONT_TAG_TO_BODY and PELVIS_LINK_OFFSET_IN_BODY.
    Tune those constants empirically via the camera GUI overlay (the resulting
    pelvis link RGB triad should sit at the pelvis center with +X pointing
    forward, +Y pointing left, +Z pointing up).
    """
    return tag_to_link_pose(
        T_lab_pelvis_tag,
        PELVIS_FRONT_TAG_TO_BODY,
        PELVIS_LINK_OFFSET_IN_BODY,
    )


# ---------------------------------------------------------------------------
# Step 3: torso_link multi-tag mapping (head tag 9 + torso tags 12, 13)
# ---------------------------------------------------------------------------
# All three tags are rigidly attached to torso_link (head is NOT a separate
# kinematic body in mjlab G1 — it's a geom child of torso_link via the
# waist_pitch joint chain, so when the waist joints sit at zero, the head and
# torso tags belong to the same rigid body).
#
# Mounting verified empirically 2026-05-25 by user inspection of the live
# RGB axes on the camera GUI in FixStand pose. mjlab body convention is the
# same as for pelvis: +X = forward, +Y = left, +Z = up.
#
# Tag 9 (head, ON TOP OF THE CRANIUM, lying flat):
#   tag +X (R red,  image right)         -> robot right = body -Y
#   tag +Y (G green, image down)         -> robot back  = body -X
#   tag +Z (B blue,  into tag surface)   -> robot down  = body -Z
#   columns of R_tag_to_body = body axes in tag frame:
#     col 0 (body +X = robot fwd)  = tag -Y = (0, -1, 0)
#     col 1 (body +Y = robot left) = tag -X = (-1, 0, 0)
#     col 2 (body +Z = robot up)   = tag -Z = (0, 0, -1)
HEAD_TAG_TO_BODY = np.array([
    [ 0.0, -1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0],
])

# Tag 12 (torso BACK, mounted on the rear of the torso):
#   tag +X (R)  -> robot right = body -Y
#   tag +Y (G)  -> robot down  = body -Z
#   tag +Z (B)  -> robot front = body +X  (tag surface faces back; +Z goes inward)
#   columns:
#     col 0 (body +X = robot fwd)  = tag +Z = (0, 0, 1)
#     col 1 (body +Y = robot left) = tag -X = (-1, 0, 0)
#     col 2 (body +Z = robot up)   = tag -Y = (0, -1, 0)
TORSO_BACK_TAG_TO_BODY = np.array([
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0, -1.0],
    [ 1.0,  0.0,  0.0],
])

# Tag 13 (torso FRONT, mounted on the front (chest) of the torso):
#   tag +X (R)  -> robot left  = body +Y
#   tag +Y (G)  -> robot down  = body -Z
#   tag +Z (B)  -> robot back  = body -X  (tag surface faces forward; +Z goes inward)
#   columns:
#     col 0 (body +X = robot fwd)  = tag -Z = (0, 0, -1)
#     col 1 (body +Y = robot left) = tag +X = (1, 0, 0)
#     col 2 (body +Z = robot up)   = tag -Y = (0, -1, 0)
TORSO_FRONT_TAG_TO_BODY = np.array([
    [ 0.0,  1.0,  0.0],
    [ 0.0,  0.0, -1.0],
    [-1.0,  0.0,  0.0],
])

# Offsets from each tag's center to torso_link origin, expressed in BODY local
# frame. Crucially these are body-relative — when the torso bends forward, the
# offset rotates with it (see `tag_to_link_pose`), so we do NOT need to know
# the absolute pose to convert.
#
# Offset history / tuning notes (user visual validation, 2026-05-25):
#   * tag 9 (head crown): ~48 cm down along the tag's normal = body -Z.
#     (head_collision sits at +0.43m above torso_link origin; tag adds a few
#      more cm sitting on top of the cranium.) Validated against pelvis-FK
#     ground truth (magenta marker on GUI).
#   * tags 12/13: tried [+0.10, 0, 0]/[-0.10, 0, 0] first (inward = body +X for
#     back tag, -X for front tag). User saw the candidates land in the WRONG
#     position so we flipped signs. Then user reported the flipped version was
#     even worse and asked to go back to the original direction at 6 cm
#     instead of 10 cm. Final empirical signs: back tag goes body +X 6 cm
#     (inward = toward chest), front tag goes body -X 6 cm (inward = toward
#     spine). Confirm again visually if it still drifts.
HEAD_LINK_OFFSET_IN_BODY        = np.array([0.0, 0.0, -0.48])
TORSO_BACK_LINK_OFFSET_IN_BODY  = np.array([+0.06, 0.0, 0.0])
TORSO_FRONT_LINK_OFFSET_IN_BODY = np.array([-0.06, 0.0, 0.0])


# Map: tag id -> (R_tag_to_body, link_offset_in_body, label)
TORSO_TAG_MAP = {
    9:  (HEAD_TAG_TO_BODY,        HEAD_LINK_OFFSET_IN_BODY,        "head_tag(9)"),
    12: (TORSO_BACK_TAG_TO_BODY,  TORSO_BACK_LINK_OFFSET_IN_BODY,  "torso_back_tag(12)"),
    13: (TORSO_FRONT_TAG_TO_BODY, TORSO_FRONT_LINK_OFFSET_IN_BODY, "torso_front_tag(13)"),
}

# Per-tag fusion weights (positive integers, normalized internally).
# Tuning history (user GUI validation, 2026-05-25):
#   * pelvis tag 8 via FK chain ("pelvis-FK cross-check") — REGARDED AS GROUND
#     TRUTH AT FixStand. The pelvis tag is large, well-lit and rigidly mounted
#     on a flat surface, so its PnP is the most stable single estimate. Gets
#     the highest weight when included via `fuse_torso_link(..., include_pelvis_fk=True)`.
#   * head tag 9: second most reliable. Tag is large and faces upward to the
#     ceiling cameras at a near-orthogonal angle. Offset is large (~48 cm)
#     but the orientation is locked down well.
#   * torso back tag 12 / front tag 13: smaller tags, often partially occluded
#     by the head/shoulders/cables from the lab camera angles. Lowest weight
#     and most prone to PnP z-flip ambiguity.
# Pelvis-FK is a SPECIAL key 'pelvis_fk' (not an AprilTag id) handled by
# fuse_torso_link's include_pelvis_fk parameter.
TORSO_TAG_DEFAULT_WEIGHTS = {"pelvis_fk": 3.0, 9: 2.0, 12: 1.0, 13: 1.0}


def head_tag_to_torso_link(T_lab_tag9: np.ndarray) -> np.ndarray:
    """Single-tag estimate of torso_link pose from the head-crown tag (id 9)."""
    return tag_to_link_pose(T_lab_tag9, HEAD_TAG_TO_BODY, HEAD_LINK_OFFSET_IN_BODY)


# pelvis link -> torso link FK offset, body frame. From mjlab g1.xml:
#   waist_yaw_link    pos (0, 0, 0)                  (relative to pelvis)
#   waist_roll_link   pos (-0.0039635, 0, 0.044)     (relative to waist_yaw)
#   torso_link        pos (0, 0, 0)                  (relative to waist_roll)
# Total at zero-waist: torso_link in pelvis frame = (-0.004, 0, +0.044) m.
PELVIS_LINK_TO_TORSO_LINK_OFFSET = np.array([-0.004, 0.0, 0.044])


def pelvis_front_tag_to_torso_link(T_lab_pelvis_tag: np.ndarray) -> np.ndarray:
    """Estimate torso_link pose from pelvis tag (id 8) via mjlab FK chain at
    zero-waist pose. Equivalent to:
        pelvis_front_tag_to_pelvis_link()  then translate body +(0, 0, +0.044).
    Useful as a CROSS-CHECK against the torso-mounted tags 9/12/13. Only valid
    when the waist joints are near zero (FixStand / motion-start poses). When
    the robot bends/twists this estimate diverges from the multi-tag fusion.
    """
    offset = PELVIS_LINK_OFFSET_IN_BODY + PELVIS_LINK_TO_TORSO_LINK_OFFSET
    return tag_to_link_pose(T_lab_pelvis_tag, PELVIS_FRONT_TAG_TO_BODY, offset)


def torso_back_tag_to_torso_link(T_lab_tag12: np.ndarray) -> np.ndarray:
    """Single-tag estimate of torso_link pose from the torso back tag (id 12)."""
    return tag_to_link_pose(T_lab_tag12, TORSO_BACK_TAG_TO_BODY,
                            TORSO_BACK_LINK_OFFSET_IN_BODY)


def torso_front_tag_to_torso_link(T_lab_tag13: np.ndarray) -> np.ndarray:
    """Single-tag estimate of torso_link pose from the torso front tag (id 13)."""
    return tag_to_link_pose(T_lab_tag13, TORSO_FRONT_TAG_TO_BODY,
                            TORSO_FRONT_LINK_OFFSET_IN_BODY)


def fuse_torso_link(
    tag_poses: dict,
    weights: dict | None = None,
    pelvis_tag_pose: np.ndarray | None = None,
    include_pelvis_fk: bool = True,
) -> tuple[np.ndarray | None, dict]:
    """Multi-source fusion of torso_link pose. Candidates:
      * tags 9/12/13 via their hardcoded mounting transforms
      * pelvis tag 8 via FK chain (`pelvis_front_tag_to_torso_link`)

    Translation = weighted mean of candidates. Rotation = weighted quaternion
    mean (sign-aligned to the first candidate, normalized).

    Args:
        tag_poses: dict mapping tag_id (9/12/13) -> (4,4) SE3 in lab frame.
        weights:   optional dict mapping source-key -> weight. Keys are tag
                   ids (9, 12, 13) and the special string "pelvis_fk".
                   Defaults to TORSO_TAG_DEFAULT_WEIGHTS.
        pelvis_tag_pose: optional (4,4) SE3 for pelvis tag 8 in lab frame.
                   If provided and include_pelvis_fk=True, the FK-chain
                   torso_link estimate is added as an extra fusion source.
        include_pelvis_fk: gate for the pelvis-FK candidate. True = use the
                   pelvis tag as ground-truth-ish source (recommended at zero-
                   waist / FixStand). False = torso tags only (better when the
                   robot bends because pelvis-FK assumes waist joints at 0).

    Returns:
        (T_lab_torso_link, diag): SE3 (4,4) or None if no usable source. diag
        carries per-source candidate poses + weight + the residual angle of
        each candidate w.r.t. the fused result (degrees), useful for debugging.
    """
    if weights is None:
        weights = TORSO_TAG_DEFAULT_WEIGHTS

    candidates = {}
    for tid, T_lab_tag in tag_poses.items():
        if T_lab_tag is None or tid not in TORSO_TAG_MAP:
            continue
        R_tag2body, offset, label = TORSO_TAG_MAP[tid]
        T_lab_link = tag_to_link_pose(T_lab_tag, R_tag2body, offset)
        candidates[tid] = (T_lab_link, label)

    if include_pelvis_fk and pelvis_tag_pose is not None:
        candidates["pelvis_fk"] = (
            pelvis_front_tag_to_torso_link(pelvis_tag_pose),
            "pelvis_fk(id8)",
        )

    if not candidates:
        return None, {"n_tags": 0, "tags_used": [], "per_tag": {}}

    # Sort: numeric tag ids first (ascending), then "pelvis_fk" at the end.
    def _sort_key(k):
        return (1 if isinstance(k, str) else 0, k)
    keys = sorted(candidates.keys(), key=_sort_key)
    ws = np.array([float(weights.get(k, 1.0)) for k in keys], dtype=float)
    w_sum = float(ws.sum())
    if w_sum <= 0.0:
        ws = np.ones_like(ws)
        w_sum = float(ws.sum())
    ws = ws / w_sum

    positions = np.stack([candidates[k][0][:3, 3] for k in keys], axis=0)
    pos_avg = (ws[:, None] * positions).sum(axis=0)

    Rs = [candidates[k][0][:3, :3] for k in keys]
    quats = np.stack([R_to_quat_wxyz(R) for R in Rs], axis=0)
    ref_q = quats[0]
    signs = np.sign(quats @ ref_q)
    signs[signs == 0] = 1.0
    quats_aligned = quats * signs[:, None]
    q_mean = (ws[:, None] * quats_aligned).sum(axis=0)
    n = float(np.linalg.norm(q_mean))
    if n < 1e-12:
        return candidates[keys[0]][0], {
            "n_tags": len(keys), "tags_used": keys, "degenerate_avg": True,
            "per_tag": {k: {"label": candidates[k][1]} for k in keys},
        }
    q_mean /= n
    R_avg = quat_wxyz_to_R(q_mean)

    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = pos_avg

    per_tag = {}
    for k, w in zip(keys, ws):
        T_c, label = candidates[k]
        d_pos = float(np.linalg.norm(T_c[:3, 3] - pos_avg))
        R_resid = T_c[:3, :3] @ R_avg.T
        cos_th = max(-1.0, min(1.0, (float(np.trace(R_resid)) - 1.0) * 0.5))
        d_rot_deg = float(np.degrees(np.arccos(cos_th)))
        per_tag[k] = {
            "label": label,
            "weight": float(w),
            "candidate_pos": T_c[:3, 3].tolist(),
            "residual_pos_m": d_pos,
            "residual_rot_deg": d_rot_deg,
        }

    return T_avg, {
        "n_tags": len(keys),
        "tags_used": keys,
        "per_tag": per_tag,
        "include_pelvis_fk": bool(include_pelvis_fk),
    }


def average_pose_samples(positions, rotation_matrices):
    """Mean of a stream of pelvis pose samples taken during the SPACE-press
    averaging window. Position is element-wise mean. Rotations are averaged
    in quaternion space with sign-flipping against the first sample
    (handles the +q / -q double cover).
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions shape must be (N,3); got {positions.shape}")
    pos_avg = positions.mean(axis=0)

    quats = np.stack([R_to_quat_wxyz(R) for R in rotation_matrices], axis=0)
    if quats.shape[0] == 0:
        raise ValueError("no rotation samples to average")
    ref_q = quats[0]
    signs = np.sign(quats @ ref_q)
    signs[signs == 0] = 1.0  # tie-breaker
    quats = quats * signs[:, None]
    q_mean = quats.mean(axis=0)
    q_norm = float(np.linalg.norm(q_mean))
    if q_norm < 1e-12:
        raise ValueError("quaternion average degenerate (samples cancel out)")
    q_mean /= q_norm
    R_avg = quat_wxyz_to_R(q_mean)
    return pos_avg, R_avg


def transform_pose_lab_to_sim(
    T_sim_lab: np.ndarray,
    pos_lab: np.ndarray,
    R_lab: np.ndarray,
    body_flip: bool = False,
):
    """Apply T_sim_lab to a 6DOF pose. If body_flip is True (use for the torso),
    right-multiply R_lab by R_BODY_FLIP first (z-down body -> z-up body).
    The box does NOT need a body flip because the tracker already emits the
    box quat in mjlab AABB body frame (after T_OBB_TO_BODY).
    """
    R_lab_eff = R_lab @ R_BODY_FLIP if body_flip else R_lab
    R_sim_lab = T_sim_lab[:3, :3]
    t_sim_lab = T_sim_lab[:3, 3]
    pos_sim = R_sim_lab @ np.asarray(pos_lab, dtype=float) + t_sim_lab
    R_sim   = R_sim_lab @ R_lab_eff
    return pos_sim, R_sim


# ---------------------------------------------------------------------------
# 6-obs computation (mirrors State_Mimic.cpp)
# ---------------------------------------------------------------------------
def compute_six_obs(
    real_torso_pos: np.ndarray, real_torso_R: np.ndarray,
    real_box_pos:   np.ndarray, real_box_R:   np.ndarray,
    ref_torso_pos:  np.ndarray, ref_torso_R:  np.ndarray,
    ref_box_pos:    np.ndarray, ref_box_R:    np.ndarray,
):
    """Compute the 6 tag-history actor observations, all in the same world frame.

    The math is exactly what State_Mimic.cpp's `motion_anchor_*`,
    `object_*_torso`, `ref_object_*_torso` register blocks compute, modulo
    the body-frame flip (already absorbed into the inputs upstream).

    Returns dict with float64 numpy arrays:
      motion_anchor_pos_b   : (3,)
      motion_anchor_ori_b   : (6,)
      object_pos_torso      : (3,)
      object_ori6_torso     : (6,)
      ref_object_pos_torso  : (3,)
      ref_object_ori6_torso : (6,)
    """
    Rt = real_torso_R.T  # body-frame projection: delta_b = Rt @ delta_w

    delta_w = ref_torso_pos - real_torso_pos
    motion_anchor_pos_b = Rt @ delta_w

    R_rel_anchor = Rt @ ref_torso_R
    motion_anchor_ori_b = rotation_to_rot6d(R_rel_anchor)

    delta_w = real_box_pos - real_torso_pos
    object_pos_torso = Rt @ delta_w
    R_rel = Rt @ real_box_R
    object_ori6_torso = rotation_to_rot6d(R_rel)

    delta_w = ref_box_pos - real_torso_pos
    ref_object_pos_torso = Rt @ delta_w
    R_rel = Rt @ ref_box_R
    ref_object_ori6_torso = rotation_to_rot6d(R_rel)

    return {
        "motion_anchor_pos_b":   motion_anchor_pos_b.astype(float),
        "motion_anchor_ori_b":   motion_anchor_ori_b.astype(float),
        "object_pos_torso":      object_pos_torso.astype(float),
        "object_ori6_torso":     object_ori6_torso.astype(float),
        "ref_object_pos_torso":  ref_object_pos_torso.astype(float),
        "ref_object_ori6_torso": ref_object_ori6_torso.astype(float),
    }


# ---------------------------------------------------------------------------
# v2 ASCII wire format
# ---------------------------------------------------------------------------
# v2 packet layout (single line, '\n'-terminated):
#   "v2 <ts_ns> <phase> <frame_idx> <num_frames> <dof> "
#   "<jp_0..jp_(dof-1)> <jv_0..jv_(dof-1)> "
#   "<map_x map_y map_z> <mao_0..mao_5> "
#   "<opt_x opt_y opt_z> <oot_0..oot_5> "
#   "<rpt_x rpt_y rpt_z> <rot_0..rot_5>\n"
#
# phase: 0 = IDLE (real==ref, robot stands still)
#        1 = PLAYBACK (T_sim_lab applied, ref frame advancing at 50 Hz)
#
# Total numeric fields = 5 (header ints) + 2*dof + 27 (six obs).
# For G1 dof=29: 5 + 58 + 27 = 90 fields, ~750 bytes ASCII.

V2_PREFIX = "v2"
V2_HEADER_INTS = 5  # ts_ns, phase, frame_idx, num_frames, dof
V2_OBS_FLOATS = 27   # 3+6+3+6+3+6


def build_v2_packet(
    ts_ns: int, phase: int, frame_idx: int, num_frames: int, dof: int,
    joint_pos: np.ndarray, joint_vel: np.ndarray, obs: dict,
) -> bytes:
    """Pack a v2 ASCII line. Order of obs floats must match parse_v2_packet."""
    if joint_pos.shape != (dof,) or joint_vel.shape != (dof,):
        raise ValueError(f"joint_pos/joint_vel shape mismatch: expected ({dof},)")
    parts = [V2_PREFIX, str(int(ts_ns)), str(int(phase)),
             str(int(frame_idx)), str(int(num_frames)), str(int(dof))]
    for arr in (joint_pos, joint_vel,
                obs["motion_anchor_pos_b"], obs["motion_anchor_ori_b"],
                obs["object_pos_torso"],   obs["object_ori6_torso"],
                obs["ref_object_pos_torso"], obs["ref_object_ori6_torso"]):
        parts.extend(f"{float(v):.6f}" for v in np.asarray(arr).reshape(-1))
    return (" ".join(parts) + "\n").encode("ascii")
