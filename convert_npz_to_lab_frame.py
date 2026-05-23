"""Convert a sub8 / mimic motion npz from its original world frame
(npz convention: +Z up, gravity = -Z) into the LAB / origin-tag frame
used by `track_robot_and_box_multicam.py` (+Z down, gravity = +Z).

Anchor: at frame 0 of the npz the robot's torso_link must equal the average
of the first --num-frames torso poses recorded in --csv. This pins down
the rigid transform (yaw + xyz translation; the z-flip is fixed by gravity
direction).

The original npz file is NEVER modified. A new file is written to --out-npz.

Why "yaw + translation only" (4 DoF)?
  - Both lab and world have z = gravity axis. So pitch / roll of the
    transform is fixed to align the gravity direction (180° flip).
  - The remaining freedom is azimuthal heading (yaw) + position offset.
  - Trying to fit pitch/roll would soak up tracking noise / coordinate
    convention errors and degrade the trajectory.

Transform definition:
    R_lab_world = Rz(yaw) @ Rx(180)         # 3x3
    t_lab_world = csv_torso_pos - R_lab_world @ npz_torso_pos_frame0
    Then for every WORLD-frame quantity in npz:
        position p:           p_lab     = R_lab_world @ p_world + t_lab_world
        rotation R:           R_lab     = R_lab_world @ R_world
        linear vel v:         v_lab     = R_lab_world @ v_world
        angular vel omega:    omega_lab = R_lab_world @ omega_world
"""
import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np


# ---------------------------- math helpers ----------------------------

def quat_wxyz_to_R(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def R_to_quat_wxyz(R):
    q = np.empty(4, dtype=float)
    t = np.trace(R)
    if t > 0:
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
    n = np.linalg.norm(q)
    if n > 1e-12:
        q /= n
    if q[0] < 0:                       # canonical hemisphere
        q = -q
    return q


def average_R(quats_wxyz):
    """SVD-projected mean rotation from a list of wxyz quaternions."""
    Rs = np.stack([quat_wxyz_to_R(q) for q in quats_wxyz], axis=0)
    M = Rs.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    return R


def Rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def angle_between_R(R1, R2):
    R = R1.T @ R2
    cos = (np.trace(R) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


# ---------------------------- CSV loading ----------------------------

def load_csv_anchor(csv_path: str, num_frames: int):
    """Read CSV and return mean torso + object poses over the first N valid frames."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    torso_valid = []
    obj_valid = []
    for r in rows:
        try:
            tp = np.array([float(r["torso_pos_x"]), float(r["torso_pos_y"]), float(r["torso_pos_z"])])
            tq = np.array([float(r["torso_quat_w"]), float(r["torso_quat_x"]),
                           float(r["torso_quat_y"]), float(r["torso_quat_z"])])
            op = np.array([float(r["obj_pos_x"]), float(r["obj_pos_y"]), float(r["obj_pos_z"])])
            oq = np.array([float(r["obj_quat_w"]), float(r["obj_quat_x"]),
                           float(r["obj_quat_y"]), float(r["obj_quat_z"])])
            torso_valid.append((tp, tq))
            obj_valid.append((op, oq))
        except (ValueError, KeyError):
            continue

    if len(torso_valid) == 0:
        raise ValueError(f"No valid torso+obj rows in CSV: {csv_path}")

    n = min(num_frames, len(torso_valid))
    return {
        "torso_pos_mean": np.mean([p for p, q in torso_valid[:n]], axis=0),
        "torso_R_mean":   average_R([q for p, q in torso_valid[:n]]),
        "torso_pos_std":  np.std([p for p, q in torso_valid[:n]], axis=0),
        "obj_pos_mean":   np.mean([p for p, q in obj_valid[:n]], axis=0),
        "obj_R_mean":     average_R([q for p, q in obj_valid[:n]]),
        "obj_pos_std":    np.std([p for p, q in obj_valid[:n]], axis=0),
        "n_used": n,
        "n_valid": len(torso_valid),
        "n_total": len(rows),
    }


# ---------------------------- main ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert mimic motion npz from world (+Z up) to lab (+Z down) "
                    "frame, anchored on the average torso pose of the first N CSV frames."
    )
    parser.add_argument("--src-npz", type=str, required=True,
                        help="Original npz (NEVER modified).")
    parser.add_argument("--csv", type=str, required=True,
                        help="Multicam tracker CSV (must have torso_* columns).")
    parser.add_argument("--torso-body-idx", type=int, default=16,
                        help="Body index of torso_link in body_pos_w / body_quat_w. "
                             "Default 16 (G1).")
    parser.add_argument("--num-frames", type=int, default=30,
                        help="Average first N CSV frames as the lab-frame anchor.")
    parser.add_argument("--yaw-source", type=str, default="torso_to_obj",
                        choices=["torso_to_obj", "torso_rot", "obj_rot"],
                        help="How to determine yaw_lab. "
                             "'torso_to_obj' (default, recommended): use the horizontal "
                             "vector from torso to object — convention-independent. "
                             "'torso_rot' / 'obj_rot': use rotation of torso (or object) "
                             "directly — sensitive to head-tag-frame vs torso_link-frame "
                             "convention mismatch.")
    parser.add_argument("--out-npz", type=str, default="",
                        help="Output npz path. If empty, defaults to "
                             "<src_stem>_lab_frame_<YYYYMMDD_HHMMSS>.npz next to src.")
    parser.add_argument("--report-json", type=str, default="",
                        help="Optional path to write the transform + diagnostics as JSON.")
    args = parser.parse_args()

    src_path = Path(args.src_npz)
    assert src_path.exists(), f"Missing src npz: {src_path}"

    if args.out_npz:
        out_path = Path(args.out_npz)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = src_path.with_name(f"{src_path.stem}_lab_frame_{ts}.npz")

    print(f"src  : {src_path}")
    print(f"csv  : {args.csv}")
    print(f"out  : {out_path}  (NEW file; src is read-only)")

    # ---------- Load ----------
    ref = np.load(src_path)
    csv_anchor = load_csv_anchor(args.csv, args.num_frames)
    T_steps = ref["body_pos_w"].shape[0]
    n_bodies = ref["body_pos_w"].shape[1]
    if not (0 <= args.torso_body_idx < n_bodies):
        raise ValueError(f"--torso-body-idx={args.torso_body_idx} out of range [0,{n_bodies-1}]")

    # ---------- Frame-0 anchor in world ----------
    npz_torso_pos0 = ref["body_pos_w"][0, args.torso_body_idx].astype(float)
    npz_torso_R0 = quat_wxyz_to_R(ref["body_quat_w"][0, args.torso_body_idx].astype(float))
    npz_obj_pos0 = ref["object_pos_w"][0].astype(float)
    npz_obj_R0 = quat_wxyz_to_R(ref["object_quat_w"][0].astype(float))

    # ---------- Construct R_lab_world (4 DoF: yaw + flip-Rx180) ----------
    # 1) Rx(180) flips Y and Z of world. After this step gravity is correctly
    #    aligned to lab's +Z = down.
    R_x180 = np.diag([1.0, -1.0, -1.0])

    # 2) Solve yaw_lab. Three possible signals:
    #    (a) torso_to_obj  (default, recommended): use the *vector* from torso to
    #        object at frame 0 — purely positional, so unaffected by tag-frame vs
    #        body-frame convention differences.
    #    (b) torso_rot:     align torso rotation. Sensitive to convention mismatch.
    #    (c) obj_rot:       align object rotation. Same sensitivity.
    if args.yaw_source == "torso_to_obj":
        v_world = npz_obj_pos0 - npz_torso_pos0
        v_world_flipped = R_x180 @ v_world           # apply gravity flip first
        v_lab = csv_anchor["obj_pos_mean"] - csv_anchor["torso_pos_mean"]
        # We want Rz(yaw) @ v_world_flipped to point in the direction of v_lab in xy.
        ang_world = float(np.arctan2(v_world_flipped[1], v_world_flipped[0]))
        ang_lab   = float(np.arctan2(v_lab[1], v_lab[0]))
        yaw_lab = (ang_lab - ang_world + np.pi) % (2 * np.pi) - np.pi
    elif args.yaw_source == "torso_rot":
        R_after_flip = R_x180 @ npz_torso_R0
        R_residual = csv_anchor["torso_R_mean"] @ R_after_flip.T
        yaw_lab = yaw_of(R_residual)
    elif args.yaw_source == "obj_rot":
        R_after_flip = R_x180 @ npz_obj_R0
        R_residual = csv_anchor["obj_R_mean"] @ R_after_flip.T
        yaw_lab = yaw_of(R_residual)
    else:
        raise ValueError(f"unknown --yaw-source: {args.yaw_source}")

    R_lab_world = Rz(yaw_lab) @ R_x180

    # 3) Translation: anchor on csv torso position. (Independent of yaw choice.)
    t_lab_world = csv_anchor["torso_pos_mean"] - R_lab_world @ npz_torso_pos0

    print("")
    print(f"=== Transform R_lab_world (3x3)   yaw_source={args.yaw_source} ===")
    for r in R_lab_world:
        print("  [{:+.4f}, {:+.4f}, {:+.4f}]".format(*r))
    print(f"=== t_lab_world (m) ===  [{t_lab_world[0]:+.4f},{t_lab_world[1]:+.4f},{t_lab_world[2]:+.4f}]")
    print(f"yaw_lab   = {np.degrees(yaw_lab):+.3f} deg")
    print(f"npz frame0 torso_pos_world: [{npz_torso_pos0[0]:+.3f},{npz_torso_pos0[1]:+.3f},{npz_torso_pos0[2]:+.3f}]")
    print(f"npz frame0 obj_pos_world:   [{npz_obj_pos0[0]:+.3f},{npz_obj_pos0[1]:+.3f},{npz_obj_pos0[2]:+.3f}]")
    print(f"csv mean   torso_pos_lab:   [{csv_anchor['torso_pos_mean'][0]:+.3f},"
          f"{csv_anchor['torso_pos_mean'][1]:+.3f},{csv_anchor['torso_pos_mean'][2]:+.3f}]")
    print(f"csv mean   obj_pos_lab:     [{csv_anchor['obj_pos_mean'][0]:+.3f},"
          f"{csv_anchor['obj_pos_mean'][1]:+.3f},{csv_anchor['obj_pos_mean'][2]:+.3f}]")
    print(f"csv num frames used: {csv_anchor['n_used']}/{csv_anchor['n_valid']}/{csv_anchor['n_total']} "
          "(used / valid / total)")

    # ---------- Apply transform to every *_w field ----------
    def transform_pos(pos):                 # last dim = 3
        flat = pos.reshape(-1, 3) @ R_lab_world.T + t_lab_world
        return flat.reshape(pos.shape).astype(np.float32)

    def transform_vec(v):                   # rotates only (lin/ang vel)
        flat = v.reshape(-1, 3) @ R_lab_world.T
        return flat.reshape(v.shape).astype(np.float32)

    def transform_quat(q):                  # last dim = 4 (wxyz)
        flat = q.reshape(-1, 4)
        out = np.empty_like(flat)
        for i in range(flat.shape[0]):
            R = quat_wxyz_to_R(flat[i])
            R_new = R_lab_world @ R
            out[i] = R_to_quat_wxyz(R_new)
        return out.reshape(q.shape).astype(np.float32)

    out_data = {}
    for k in ref.files:
        a = ref[k]
        if k.endswith("_pos_w"):
            out_data[k] = transform_pos(a.astype(float))
        elif k.endswith("_quat_w"):
            out_data[k] = transform_quat(a.astype(float))
        elif k.endswith("_lin_vel_w") or k.endswith("_ang_vel_w"):
            out_data[k] = transform_vec(a.astype(float))
        else:
            # frame-invariant: joint_pos / joint_vel / contact_mask / fps
            out_data[k] = a

    # ---------- Self-check on frame 0 ----------
    new_torso_pos0 = out_data["body_pos_w"][0, args.torso_body_idx]
    new_torso_R0 = quat_wxyz_to_R(out_data["body_quat_w"][0, args.torso_body_idx])
    new_obj_pos0 = out_data["object_pos_w"][0]
    new_obj_R0 = quat_wxyz_to_R(out_data["object_quat_w"][0])

    pos_err_torso = np.linalg.norm(new_torso_pos0 - csv_anchor["torso_pos_mean"])
    pos_err_obj = np.linalg.norm(new_obj_pos0 - csv_anchor["obj_pos_mean"])

    # Yaw of the torso→box vector (convention-independent)
    v_lab_npz = (new_obj_pos0 - new_torso_pos0)
    v_lab_csv = (csv_anchor["obj_pos_mean"] - csv_anchor["torso_pos_mean"])
    ang_npz = float(np.arctan2(v_lab_npz[1], v_lab_npz[0]))
    ang_csv = float(np.arctan2(v_lab_csv[1], v_lab_csv[0]))
    yaw_v_err = (ang_npz - ang_csv + np.pi) % (2 * np.pi) - np.pi

    yaw_torso_err = (yaw_of(new_torso_R0) - yaw_of(csv_anchor["torso_R_mean"]) + np.pi) % (2 * np.pi) - np.pi
    full_R_err_torso = angle_between_R(new_torso_R0, csv_anchor["torso_R_mean"])
    full_R_err_obj = angle_between_R(new_obj_R0, csv_anchor["obj_R_mean"])

    print("")
    print("=== Frame-0 alignment self-check ===")
    print(f"  torso position residual:        {pos_err_torso*1000:>7.2f} mm   (always 0 by construction)")
    print(f"  obj   position residual:        {pos_err_obj*1000:>7.2f} mm   (small = good xy yaw)")
    print(f"  torso→obj vector yaw residual:  {np.degrees(yaw_v_err):>+7.2f} deg  "
          f"(target 0 when --yaw-source=torso_to_obj)")
    print(f"  torso rotation yaw residual:    {np.degrees(yaw_torso_err):>+7.2f} deg  "
          f"(non-0 unless head_tag-frame == torso_link-frame)")
    print(f"  torso |R| total residual:       {np.degrees(full_R_err_torso):>+7.2f} deg")
    print(f"  obj   |R| total residual:       {np.degrees(full_R_err_obj):>+7.2f} deg  "
          f"(non-0 if box body-frame convention differs)")
    print(f"  new transformed obj_pos_lab:    [{new_obj_pos0[0]:+.3f},"
          f"{new_obj_pos0[1]:+.3f},{new_obj_pos0[2]:+.3f}]")

    # ---------- Save ----------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {out_path}")
    np.savez(out_path, **out_data)
    print("")
    print(f"[ok] wrote {out_path}  (T={T_steps} steps, {n_bodies} bodies)")
    print("     keys preserved:", ", ".join(sorted(ref.files)))

    # ---------- Optional report ----------
    report = {
        "src_npz": str(src_path),
        "csv": args.csv,
        "out_npz": str(out_path),
        "torso_body_idx": int(args.torso_body_idx),
        "num_frames_used": int(csv_anchor["n_used"]),
        "yaw_source": args.yaw_source,
        "yaw_lab_deg": float(np.degrees(yaw_lab)),
        "R_lab_world": R_lab_world.tolist(),
        "t_lab_world_m": [float(x) for x in t_lab_world],
        "frame0_residuals": {
            "torso_pos_mm": float(pos_err_torso * 1000.0),
            "obj_pos_mm": float(pos_err_obj * 1000.0),
            "torso_to_obj_yaw_deg": float(np.degrees(yaw_v_err)),
            "torso_rot_yaw_deg": float(np.degrees(yaw_torso_err)),
            "torso_rot_R_total_deg": float(np.degrees(full_R_err_torso)),
            "obj_rot_R_total_deg": float(np.degrees(full_R_err_obj)),
        },
        "csv_anchor": {
            "torso_pos_mean_m": [float(x) for x in csv_anchor["torso_pos_mean"]],
            "torso_pos_std_mm": [float(x * 1000.0) for x in csv_anchor["torso_pos_std"]],
            "obj_pos_mean_m":   [float(x) for x in csv_anchor["obj_pos_mean"]],
            "obj_pos_std_mm":   [float(x * 1000.0) for x in csv_anchor["obj_pos_std"]],
        },
        "method": ("yaw + translation alignment with fixed Rx(180) gravity flip; "
                   f"yaw source = {args.yaw_source}"),
    }
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"     report -> {args.report_json}")


if __name__ == "__main__":
    main()
