"""Smoke tests for utils/runtime_alignment.

Run from repo root:
    python3 -m tests.test_runtime_alignment

Verifies:
  1. quat_wxyz_to_R / R_to_quat_wxyz round-trip stability.
  2. T_sim_lab math: when the real torso (after R_BODY_FLIP) equals the NPZ
     reference, the resulting T_sim_lab maps the real lab pose back to the
     ref sim pose exactly (zero residual).
  3. compute_six_obs sanity: when real == ref in sim frame, the 6 obs reduce
     to (zeros, identity rot6d, ref-implied box-in-torso, ref-implied
     box-in-torso) — i.e., the values used by the IDLE phase.
  4. Phase IDLE -> PLAYBACK continuity: with the latched real torso pose at
     SPACE press, the first PLAYBACK obs equal the IDLE obs (no
     discontinuity, modulo numerical noise).
  5. v2 packet build/parse round-trip via the C++-side textual format
     (parses with a tiny Python re-implementation of the C++ tokenizer).
"""

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.runtime_alignment import (   # noqa: E402
    R_BODY_FLIP,
    average_pose_samples,
    build_v2_packet,
    compute_six_obs,
    compute_T_sim_lab,
    compute_T_sim_lab_pelvis_yaw,
    quat_wxyz_to_R,
    R_to_quat_wxyz,
    rotation_to_rot6d,
    transform_pose_lab_to_sim,
)


def _approx(a, b, tol=1e-6, label=""):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    err = float(np.max(np.abs(a - b)))
    if err > tol:
        raise AssertionError(
            f"[{label}] max abs diff {err:.3e} > tol {tol:.3e}\n"
            f"  a = {a}\n  b = {b}")


def _rand_quat(seed=None):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    return q


def test_quat_round_trip():
    """quat -> R -> quat reproduces the input (up to global sign)."""
    rng = np.random.default_rng(123)
    for _ in range(50):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        R = quat_wxyz_to_R(q)
        q2 = R_to_quat_wxyz(R)
        # Quaternion sign ambiguity: pick the closer of +q2, -q2 to q.
        if np.dot(q, q2) < 0:
            q2 = -q2
        _approx(q, q2, tol=1e-6, label="quat round-trip")
    print("[ok] test_quat_round_trip")


def test_T_identity_when_real_equals_ref():
    """If real_torso (after R_BODY_FLIP) matches ref_torso at the same lab
    point, the computed T_sim_lab maps the real lab pose back to ref sim pose
    exactly. We pick lab pos = R_lab_world @ ref_pos so that t_lab_world = 0.
    """
    ref_pos = np.array([1.2, -0.3, 0.5])
    ref_q = _rand_quat(seed=42)
    ref_R = quat_wxyz_to_R(ref_q)

    # Construct a fake real_R_lab_raw whose body flip equals a yaw-only-rotated
    # ref_R: real_R_lab_zup = R_yaw @ R_flip @ ref_R, then peel off R_BODY_FLIP.
    R_flip = np.diag([1.0, -1.0, -1.0])
    delta_yaw = 0.30  # rad
    c, s = math.cos(delta_yaw), math.sin(delta_yaw)
    R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    real_R_lab_zup = R_yaw @ R_flip @ ref_R
    real_R_lab_raw = real_R_lab_zup @ R_BODY_FLIP   # invert flip on the right

    R_lab_world = R_yaw @ R_flip
    real_pos_lab = R_lab_world @ ref_pos             # so t_lab_world == 0

    T_sim_lab, diag = compute_T_sim_lab(
        real_pos_lab, real_R_lab_raw, ref_pos, ref_R,
        mode="yaw-only", torso_forward_axis="+x",
    )

    # Forward-check: applying T_sim_lab + body flip on the real lab pose
    # should give back the ref sim pose exactly.
    pos_back, R_back = transform_pose_lab_to_sim(
        T_sim_lab, real_pos_lab, real_R_lab_raw, body_flip=True,
    )
    _approx(pos_back, ref_pos, tol=1e-9, label="T pos round-trip")
    _approx(R_back, ref_R, tol=1e-9, label="T rot round-trip")

    # delta_yaw recovered
    assert diag["delta_yaw_rad"] is not None
    assert abs(diag["delta_yaw_rad"] - delta_yaw) < 1e-9, \
        f"delta_yaw mismatch: got {diag['delta_yaw_rad']} expect {delta_yaw}"
    print("[ok] test_T_identity_when_real_equals_ref")


def test_six_obs_real_equals_ref():
    """When real == ref in sim frame, the IDLE obs come out as expected:
       motion_anchor_pos_b == 0, motion_anchor_ori_b == identity rot6d,
       object_pos_torso == ref_object_pos_torso, etc.
    """
    rng = np.random.default_rng(7)
    ref_torso_pos = rng.standard_normal(3)
    ref_torso_R   = quat_wxyz_to_R(_rand_quat(seed=8))
    ref_box_pos   = ref_torso_pos + np.array([0.1, 0.2, -0.3])
    ref_box_R     = quat_wxyz_to_R(_rand_quat(seed=9))

    obs = compute_six_obs(
        ref_torso_pos, ref_torso_R, ref_box_pos, ref_box_R,
        ref_torso_pos, ref_torso_R, ref_box_pos, ref_box_R,
    )
    _approx(obs["motion_anchor_pos_b"], np.zeros(3), tol=1e-12,
            label="map_b zero on real==ref")
    _approx(obs["motion_anchor_ori_b"], rotation_to_rot6d(np.eye(3)),
            tol=1e-12, label="mao_b identity on real==ref")
    # object_pos_torso == ref_object_pos_torso, both equal to box-in-torso
    _approx(obs["object_pos_torso"], obs["ref_object_pos_torso"], tol=1e-12,
            label="opt == ref_opt on real==ref")
    _approx(obs["object_ori6_torso"], obs["ref_object_ori6_torso"], tol=1e-12,
            label="oot == ref_oot on real==ref")
    print("[ok] test_six_obs_real_equals_ref")


def test_idle_to_playback_continuity():
    """At SPACE press, the first PLAYBACK obs (frame_idx=0, real torso transformed
    to sim frame) should match the IDLE obs (real == ref) when the latched
    real torso pose really WAS at frame 0. This checks there's no
    discontinuity at the phase transition moment.
    """
    rng = np.random.default_rng(123)
    ref_torso_pos = rng.standard_normal(3)
    ref_torso_R   = quat_wxyz_to_R(_rand_quat(seed=11))
    ref_box_pos   = ref_torso_pos + np.array([0.4, -0.2, 0.6])
    ref_box_R     = quat_wxyz_to_R(_rand_quat(seed=12))

    # Pretend a real torso pose in lab (z-down body) corresponds exactly to
    # ref frame 0 (after flipping body conv + applying inv(T)). Build it by
    # picking an arbitrary R_yaw and constructing the lab-frame counterpart.
    delta_yaw = -0.7
    c, s = math.cos(delta_yaw), math.sin(delta_yaw)
    R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R_flip = np.diag([1.0, -1.0, -1.0])
    R_lab_world = R_yaw @ R_flip

    real_torso_pos_lab = R_lab_world @ ref_torso_pos
    # real_torso_R_lab_zup must equal R_lab_world @ ref_R when the start
    # pose is genuinely "at" frame 0. So real_R_lab_raw (head-tag z-down) =
    # (R_lab_world @ ref_R) @ R_BODY_FLIP^{-1} = (R_lab_world @ ref_R) @ R_BODY_FLIP.
    real_R_lab_raw = (R_lab_world @ ref_torso_R) @ R_BODY_FLIP

    # Box: tracker gives box in mjlab body conv already, so real_box_R_lab =
    # R_lab_world @ ref_box_R.
    real_box_pos_lab = R_lab_world @ ref_box_pos
    real_box_R_lab   = R_lab_world @ ref_box_R

    # IDLE obs (computed inline): real := ref
    obs_idle = compute_six_obs(
        ref_torso_pos, ref_torso_R, ref_box_pos, ref_box_R,
        ref_torso_pos, ref_torso_R, ref_box_pos, ref_box_R,
    )

    # Compute T at SPACE press, then apply it to the lab values.
    T_sim_lab, _ = compute_T_sim_lab(
        real_torso_pos_lab, real_R_lab_raw, ref_torso_pos, ref_torso_R,
        mode="yaw-only", torso_forward_axis="+x",
    )
    real_torso_sim_pos, real_torso_sim_R = transform_pose_lab_to_sim(
        T_sim_lab, real_torso_pos_lab, real_R_lab_raw, body_flip=True,
    )
    real_box_sim_pos, real_box_sim_R = transform_pose_lab_to_sim(
        T_sim_lab, real_box_pos_lab, real_box_R_lab, body_flip=False,
    )

    obs_play = compute_six_obs(
        real_torso_sim_pos, real_torso_sim_R, real_box_sim_pos, real_box_sim_R,
        ref_torso_pos, ref_torso_R, ref_box_pos, ref_box_R,
    )

    for k in obs_idle:
        _approx(obs_idle[k], obs_play[k], tol=1e-9,
                label=f"IDLE/PLAYBACK continuity '{k}'")
    print("[ok] test_idle_to_playback_continuity")


def test_v2_packet_build_and_parse():
    """Round-trip a v2 packet through the publisher's build_v2_packet and a
    Python re-implementation of the C++ parser (camera_pose_subscriber.h::
    parse_v2)."""
    dof = 29
    obs = {
        "motion_anchor_pos_b": np.array([0.1, -0.2, 0.3]),
        "motion_anchor_ori_b": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        "object_pos_torso":    np.array([-0.4, 0.5, 0.6]),
        "object_ori6_torso":   np.array([0.7, 0.1, 0.0, -0.1, 0.7, 0.0]),
        "ref_object_pos_torso":  np.array([0.2, 0.0, 0.1]),
        "ref_object_ori6_torso": np.array([0.5, 0.5, 0.0, -0.5, 0.5, 0.0]),
    }
    jp = np.linspace(-1.0, 1.0, dof)
    jv = np.linspace(0.0, 0.5, dof)
    pkt = build_v2_packet(
        ts_ns=123_456_789, phase=1, frame_idx=42, num_frames=434,
        dof=dof, joint_pos=jp, joint_vel=jv, obs=obs,
    )
    txt = pkt.decode("ascii")
    assert txt.startswith("v2 "), "missing v2 prefix"
    assert txt.endswith("\n"), "missing newline terminator"
    toks = txt.strip().split()
    # 1 ("v2") + 5 header ints + 2*dof + 27 = 6 + 58 + 27 = 91
    assert len(toks) == 1 + 5 + 2 * dof + 27, \
        f"unexpected token count {len(toks)}"

    # Re-parse manually
    cur = iter(toks)
    assert next(cur) == "v2"
    assert int(next(cur)) == 123_456_789
    assert int(next(cur)) == 1
    assert int(next(cur)) == 42
    assert int(next(cur)) == 434
    assert int(next(cur)) == dof
    jp_parsed = np.array([float(next(cur)) for _ in range(dof)])
    jv_parsed = np.array([float(next(cur)) for _ in range(dof)])
    _approx(jp_parsed, jp, tol=1e-5, label="jp v2 parse")
    _approx(jv_parsed, jv, tol=1e-5, label="jv v2 parse")
    for k, n in [("motion_anchor_pos_b", 3), ("motion_anchor_ori_b", 6),
                 ("object_pos_torso", 3), ("object_ori6_torso", 6),
                 ("ref_object_pos_torso", 3), ("ref_object_ori6_torso", 6)]:
        parsed = np.array([float(next(cur)) for _ in range(n)])
        _approx(parsed, obs[k], tol=1e-5, label=f"v2 parse '{k}'")
    print("[ok] test_v2_packet_build_and_parse")


def test_pelvis_yaw_T_identity():
    """Pelvis-yaw T_sim_lab: when the lab pelvis tag is yawed by some known
    delta vs the NPZ pelvis (in mjlab body conv), the recovered delta_yaw
    matches and T round-trips the lab pelvis pose to the sim pelvis pose."""
    ref_pelvis_pos_sim = np.array([0.4, -0.1, 0.78])
    ref_pelvis_R_sim   = quat_wxyz_to_R(_rand_quat(seed=21))

    # Build a fake real pelvis tag rotation: in the tag's local frame,
    # body-forward = -z_tag (default --pelvis-tag-fwd-axis). We place the
    # real pelvis in lab world such that its yaw differs from the ref by a
    # known amount, with no roll/pitch (matches "pelvis is vertical"
    # assumption at FixStand).
    R_flip = np.diag([1.0, -1.0, -1.0])
    delta_yaw_truth = 0.40
    c, s = math.cos(delta_yaw_truth), math.sin(delta_yaw_truth)
    R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R_lab_world = R_yaw @ R_flip

    # Set up a tag-local frame whose -z axis points along (R_lab_world @
    # ref_R_sim @ +x). Equivalent: R_real_lab @ (-z) == fwd_dir_in_lab.
    fwd_in_lab = R_lab_world @ ref_pelvis_R_sim @ np.array([1.0, 0.0, 0.0])
    fwd_in_lab[2] = 0.0   # zero out vertical so the tag is "horizontal-fwd"
    fwd_in_lab = fwd_in_lab / (np.linalg.norm(fwd_in_lab) + 1e-12)
    # Build a tag local frame: we want -z_tag (in lab) = fwd_in_lab,
    # +y_tag (in lab) = (0, 0, -1) (lab z-down -> body up = lab -z).
    z_tag_in_lab = -fwd_in_lab
    y_tag_in_lab = np.array([0.0, 0.0, -1.0])
    x_tag_in_lab = np.cross(y_tag_in_lab, z_tag_in_lab)
    x_tag_in_lab = x_tag_in_lab / (np.linalg.norm(x_tag_in_lab) + 1e-12)
    real_pelvis_R_lab = np.column_stack([x_tag_in_lab, y_tag_in_lab,
                                          z_tag_in_lab])

    real_pelvis_pos_lab = R_lab_world @ ref_pelvis_pos_sim   # so t_lab=0

    T_sim_lab, diag = compute_T_sim_lab_pelvis_yaw(
        real_pelvis_pos_lab, real_pelvis_R_lab,
        ref_pelvis_pos_sim, ref_pelvis_R_sim,
        real_fwd_axis_tag_local="-z", ref_fwd_axis_body_local="+x",
    )
    assert abs(diag["delta_yaw_rad"] - delta_yaw_truth) < 1e-6, \
        f"pelvis delta_yaw mismatch: got {diag['delta_yaw_rad']}, expect {delta_yaw_truth}"

    # T should map real_pelvis_pos_lab back to ref_pelvis_pos_sim.
    pos_back = (T_sim_lab[:3, :3] @ real_pelvis_pos_lab) + T_sim_lab[:3, 3]
    _approx(pos_back, ref_pelvis_pos_sim, tol=1e-9, label="pelvis pos round-trip")
    print("[ok] test_pelvis_yaw_T_identity")


def test_average_pose_samples():
    """average_pose_samples: with no noise, the mean reproduces the input;
    with random noise, the mean is close to the noiseless input (sanity)."""
    base_R = quat_wxyz_to_R(_rand_quat(seed=99))
    base_pos = np.array([0.5, -0.2, 0.8])

    pos_samples = [base_pos.copy() for _ in range(10)]
    R_samples = [base_R.copy() for _ in range(10)]
    pos_avg, R_avg = average_pose_samples(pos_samples, R_samples)
    _approx(pos_avg, base_pos, tol=1e-12, label="avg pos noiseless")
    _approx(R_avg,   base_R,   tol=1e-9,  label="avg R noiseless")

    # Small perturbations
    rng = np.random.default_rng(11)
    pos_samples_n = [base_pos + rng.normal(0, 1e-3, 3) for _ in range(50)]
    # Random small rotation perturbations in quat space
    R_samples_n = []
    for _ in range(50):
        d = rng.normal(0, 5e-3, 4)
        q = R_to_quat_wxyz(base_R) + d
        q = q / np.linalg.norm(q)
        R_samples_n.append(quat_wxyz_to_R(q))
    pos_avg_n, R_avg_n = average_pose_samples(pos_samples_n, R_samples_n)
    _approx(pos_avg_n, base_pos, tol=5e-4, label="avg pos noisy")
    # Rotation: ensure the averaged R is close to base in Frobenius distance
    err = float(np.linalg.norm(R_avg_n - base_R))
    assert err < 5e-2, f"avg R noisy too far: {err}"
    print("[ok] test_average_pose_samples")


if __name__ == "__main__":
    test_quat_round_trip()
    test_T_identity_when_real_equals_ref()
    test_six_obs_real_equals_ref()
    test_idle_to_playback_continuity()
    test_v2_packet_build_and_parse()
    test_pelvis_yaw_T_identity()
    test_average_pose_samples()
    print("\nAll smoke tests passed.")
