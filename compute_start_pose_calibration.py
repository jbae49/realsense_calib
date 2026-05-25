"""compute_start_pose_calibration.py

Step 1 of the simplified deploy calibration flow.

Given a `track_robot_and_box_multicam.py` CSV captured while the robot + box
are sitting in the policy's *start pose*, average the last N (or any window
of N) frames where the relevant tags are simultaneously visible, and emit a
single JSON that holds the mean lab-frame poses of:

  * pelvis  (tag 7/8, averaged across whichever was used)
  * torso   (fused head_path / pelvis_path — same as tracker's `torso_*`)
  * object  (box, in mjlab AABB body frame after T_OBB_TO_BODY correction)

Plus per-component std (for sanity / sensor-noise inspection).

Subsequent steps:
  * `compute_T_sim_lab_pelvis.py`  — uses pelvis_pose + NPZ frame-0 pelvis
  * `compute_R_obj_tag.py`         — uses object_pose + NPZ frame-0 object
                                     + T_sim_lab

Why pelvis (not torso) for T_sim_lab:
  At FixStand the pelvis is approximately vertical, but the torso is bent
  ~25 deg forward in the NPZ frame 0 vs vertical in FixStand. Anchoring the
  world transform on pelvis avoids baking that mismatch into T_sim_lab.
  See utils/runtime_alignment.compute_T_sim_lab_pelvis_yaw docstring.

Usage:
  python compute_start_pose_calibration.py \
    --tracker-csv outputs/sub8_45_taghist_20260524_025853_20260524_030505.csv \
    --num-frames 100 \
    --window tail \
    --out-json config/start_pose_calibration.json

The CSV is expected to be one produced by track_robot_and_box_multicam.py
with the standard column set (pelvis_*, torso_*, obj_*, *_visible flags).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

from utils.runtime_alignment import (
    average_pose_samples,
    quat_wxyz_to_R,
    R_to_quat_wxyz,
)


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------
REQUIRED_COLS = [
    # visibility flags used to gate "good" frames
    "pelvis_visible", "head_visible", "box_visible",
    # pelvis lab pose (raw tag pose; pelvis_pos_x is body-shifted to root in
    # tracker, so keep both: 'pelvis_*' is the tag, 'root_*' is +0.05m up).
    "pelvis_pos_x", "pelvis_pos_y", "pelvis_pos_z",
    "pelvis_quat_w", "pelvis_quat_x", "pelvis_quat_y", "pelvis_quat_z",
    # torso lab pose (fused). The tracker also emits torso_head_* and
    # torso_pelvis_* so the user can sanity-check which path was used.
    "torso_pos_x", "torso_pos_y", "torso_pos_z",
    "torso_quat_w", "torso_quat_x", "torso_quat_y", "torso_quat_z",
    # object lab pose (box AABB body frame, post T_OBB_TO_BODY)
    "obj_pos_x", "obj_pos_y", "obj_pos_z",
    "obj_quat_w", "obj_quat_x", "obj_quat_y", "obj_quat_z",
]


def _to_bool(v: str) -> bool:
    s = (v or "").strip().lower()
    return s in ("1", "true", "yes", "y", "t")


def load_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    header = set(rows[0].keys())
    missing = [c for c in REQUIRED_COLS if c not in header]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Expected from track_robot_and_box_multicam.py output."
        )
    return rows


def select_window(
    rows: List[dict],
    num_frames: int,
    window: str,
    skip_first: int,
) -> List[dict]:
    """Filter to rows where pelvis, head, and box are all visible, then pick
    `num_frames` of them based on `window` ('head' = first N, 'tail' = last N,
    'middle' = central N). Skips the very first `skip_first` raw rows before
    visibility filtering (useful when the tracker has a couple of warmup
    frames with bad pose).
    """
    if skip_first > 0:
        rows = rows[skip_first:]
    good = [
        r for r in rows
        if _to_bool(r["pelvis_visible"])
        and _to_bool(r["head_visible"])
        and _to_bool(r["box_visible"])
    ]
    if len(good) < num_frames:
        raise ValueError(
            f"Only {len(good)} rows have all of pelvis/head/box visible; "
            f"requested {num_frames}. Re-capture with better tag visibility."
        )

    if window == "head":
        return good[:num_frames]
    if window == "tail":
        return good[-num_frames:]
    if window == "middle":
        start = (len(good) - num_frames) // 2
        return good[start:start + num_frames]
    raise ValueError(f"bad --window '{window}'")


def extract_pose(
    rows: List[dict],
    pos_keys: Tuple[str, str, str],
    quat_keys: Tuple[str, str, str, str],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Returns (positions Nx3 ndarray, list of R(3x3))."""
    pos = np.array(
        [[float(r[k]) for k in pos_keys] for r in rows],
        dtype=float,
    )
    quats = np.array(
        [[float(r[k]) for k in quat_keys] for r in rows],
        dtype=float,
    )
    # Normalize quats defensively.
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    quats = quats / norms
    Rs = [quat_wxyz_to_R(q) for q in quats]
    return pos, Rs


def pos_std(pos: np.ndarray) -> List[float]:
    return [float(s) for s in pos.std(axis=0)]


def quat_std_deg(Rs: List[np.ndarray], R_mean: np.ndarray) -> float:
    """Average geodesic distance (in degrees) from R_mean. A scalar; useful
    for one-number 'how stable was the rotation across samples' sanity.
    """
    if not Rs:
        return 0.0
    angs = []
    for R in Rs:
        dR = R_mean.T @ R
        c = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
        angs.append(float(np.degrees(np.arccos(c))))
    return float(np.mean(angs))


def summarize_pose(name: str, pos: np.ndarray, Rs: List[np.ndarray]) -> dict:
    pos_mean, R_mean = average_pose_samples(pos, Rs)
    q_mean = R_to_quat_wxyz(R_mean)
    rot_std = quat_std_deg(Rs, R_mean)
    return {
        f"{name}_pos_mean_xyz_m": [float(x) for x in pos_mean],
        f"{name}_quat_mean_wxyz": [float(x) for x in q_mean],
        f"{name}_R_mean_3x3":     R_mean.tolist(),
        f"{name}_pos_std_xyz_m":  pos_std(pos),
        f"{name}_rot_std_deg":    rot_std,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracker-csv", type=Path, required=True,
                   help="CSV from track_robot_and_box_multicam.py")
    p.add_argument("--num-frames", type=int, default=100,
                   help="How many *valid* frames to average over (default 100).")
    p.add_argument("--window", choices=["head", "tail", "middle"],
                   default="tail",
                   help="Which N-frame window of the valid rows to use. "
                        "'tail' (default) is robust: last N frames after the "
                        "operator has confirmed the start arrangement.")
    p.add_argument("--skip-first", type=int, default=0,
                   help="Drop first K raw rows before visibility filtering. "
                        "Useful if the tracker has warmup frames.")
    p.add_argument("--out-json", type=Path, required=True,
                   help="Where to save the calibration JSON.")
    args = p.parse_args()

    rows = load_csv(args.tracker_csv)
    sel = select_window(rows, args.num_frames, args.window, args.skip_first)

    pelvis_pos, pelvis_Rs = extract_pose(
        sel,
        ("pelvis_pos_x", "pelvis_pos_y", "pelvis_pos_z"),
        ("pelvis_quat_w", "pelvis_quat_x", "pelvis_quat_y", "pelvis_quat_z"),
    )
    torso_pos, torso_Rs = extract_pose(
        sel,
        ("torso_pos_x", "torso_pos_y", "torso_pos_z"),
        ("torso_quat_w", "torso_quat_x", "torso_quat_y", "torso_quat_z"),
    )
    obj_pos, obj_Rs = extract_pose(
        sel,
        ("obj_pos_x", "obj_pos_y", "obj_pos_z"),
        ("obj_quat_w", "obj_quat_x", "obj_quat_y", "obj_quat_z"),
    )

    out = {
        "meta": {
            "tracker_csv": str(args.tracker_csv.resolve()),
            "num_frames_requested": int(args.num_frames),
            "num_valid_rows": int(len(sel)),
            "total_csv_rows": int(len(rows)),
            "window": args.window,
            "skip_first": int(args.skip_first),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "frame_idx_first": int(sel[0]["frame_idx"]),
            "frame_idx_last": int(sel[-1]["frame_idx"]),
        },
        "frame": "lab (pupil_apriltags floor-anchor; +Z = down)",
        "quat_convention": "wxyz, unit",
    }
    out.update(summarize_pose("pelvis", pelvis_pos, pelvis_Rs))
    out.update(summarize_pose("torso",  torso_pos,  torso_Rs))
    out.update(summarize_pose("object", obj_pos,    obj_Rs))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))

    # Console summary
    print("===========================================================")
    print(f"start-pose calibration saved -> {args.out_json}")
    print(f"  source csv         : {args.tracker_csv}")
    print(f"  rows used / total  : {len(sel)} / {len(rows)} "
          f"(window={args.window})")
    print(f"  frame_idx span     : {out['meta']['frame_idx_first']} ..."
          f" {out['meta']['frame_idx_last']}")
    for name in ("pelvis", "torso", "object"):
        pos = out[f"{name}_pos_mean_xyz_m"]
        std = out[f"{name}_pos_std_xyz_m"]
        rstd = out[f"{name}_rot_std_deg"]
        print(f"  {name:<7s} pos = [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m"
              f"   pos_std = [{std[0]*1000:.1f}, {std[1]*1000:.1f}, {std[2]*1000:.1f}] mm"
              f"   rot_std = {rstd:.2f} deg")
    print("===========================================================")


if __name__ == "__main__":
    main()
