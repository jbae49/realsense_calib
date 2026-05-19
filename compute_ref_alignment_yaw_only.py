import argparse
import csv
import json
from pathlib import Path

import numpy as np


def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_wxyz_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def rmat_from_yaw(yaw: float):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def pose_to_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def robust_mean(vals, k=3.5):
    vals = np.asarray(vals, dtype=float)
    med = np.median(vals, axis=0)
    mad = np.median(np.abs(vals - med), axis=0)
    mad = np.where(mad < 1e-9, 1e-9, mad)
    keep = np.all(np.abs(vals - med) <= (k * mad), axis=1)
    if np.sum(keep) < max(3, int(0.3 * len(vals))):
        keep = np.ones(len(vals), dtype=bool)
    return np.mean(vals[keep], axis=0), keep


def find_cols(header, candidates):
    for cand in candidates:
        if all(c in header for c in cand):
            return cand
    return None


def load_obs_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    header = set(rows[0].keys())

    torso_pos_cols = find_cols(
        header,
        [
            ("torso_x", "torso_y", "torso_z"),
            ("torso_pos_x", "torso_pos_y", "torso_pos_z"),
            ("root_x", "root_y", "root_z"),
            ("root_pos_x", "root_pos_y", "root_pos_z"),
        ],
    )
    torso_quat_cols = find_cols(
        header,
        [
            ("torso_qw", "torso_qx", "torso_qy", "torso_qz"),
            ("torso_quat_w", "torso_quat_x", "torso_quat_y", "torso_quat_z"),
            ("root_qw", "root_qx", "root_qy", "root_qz"),
            ("root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z"),
        ],
    )
    obj_pos_cols = find_cols(
        header,
        [
            ("obj_x", "obj_y", "obj_z"),
            ("obj_pos_x", "obj_pos_y", "obj_pos_z"),
            ("object_x", "object_y", "object_z"),
            ("object_pos_x", "object_pos_y", "object_pos_z"),
        ],
    )

    if torso_pos_cols is None or torso_quat_cols is None:
        raise ValueError(
            "CSV must include torso/root pos+quat columns. "
            "Example: root_pos_x,y,z and root_quat_w,x,y,z"
        )

    torso_pos = []
    torso_quat = []
    obj_pos = []
    has_obj = obj_pos_cols is not None

    for r in rows:
        torso_pos.append([float(r[c]) for c in torso_pos_cols])
        torso_quat.append([float(r[c]) for c in torso_quat_cols])
        if has_obj:
            obj_pos.append([float(r[c]) for c in obj_pos_cols])

    out = {
        "torso_pos": np.asarray(torso_pos, dtype=float),
        "torso_quat": np.asarray(torso_quat, dtype=float),
        "has_obj": has_obj,
    }
    if has_obj:
        out["obj_pos"] = np.asarray(obj_pos, dtype=float)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Estimate fixed T_ref_lab from initial N frames (yaw-only robust alignment)."
    )
    parser.add_argument("--obs-csv", type=str, required=True, help="Observed lab-frame torso/object CSV")
    parser.add_argument(
        "--ref-npz",
        type=str,
        default="humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz",
        help="Reference motion npz (world frame)",
    )
    parser.add_argument("--ref-start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=60, help="Use first N frames for initialization")
    parser.add_argument("--yaw-gate-deg", type=float, default=20.0, help="Yaw outlier gate in deg")
    parser.add_argument("--out-json", type=str, default="config/T_ref_lab_init.json")
    args = parser.parse_args()

    obs = load_obs_csv(args.obs_csv)
    ref = np.load(args.ref_npz)

    ref_torso_pos = ref["body_pos_w"][:, 0, :].astype(float)
    ref_torso_quat = ref["body_quat_w"][:, 0, :].astype(float)
    ref_obj_pos = ref["object_pos_w"].astype(float)

    n_obs = len(obs["torso_pos"])
    n_ref = len(ref_torso_pos) - args.ref_start_frame
    n = min(args.num_frames, n_obs, n_ref)
    if n < 5:
        raise ValueError(f"Not enough frames for robust init: {n}")

    sl = slice(args.ref_start_frame, args.ref_start_frame + n)
    obs_torso_pos = obs["torso_pos"][:n]
    obs_torso_yaw = np.array([quat_wxyz_to_yaw(q) for q in obs["torso_quat"][:n]], dtype=float)
    ref_torso_pos_n = ref_torso_pos[sl]
    ref_torso_yaw = np.array([quat_wxyz_to_yaw(q) for q in ref_torso_quat[sl]], dtype=float)

    # 1) Robust yaw offset from multiple frames.
    yaw_diff = np.array([wrap_pi(r - o) for r, o in zip(ref_torso_yaw, obs_torso_yaw)], dtype=float)
    yaw_gate = np.deg2rad(args.yaw_gate_deg)
    yaw_med = np.median(yaw_diff)
    yaw_keep = np.abs(np.array([wrap_pi(y - yaw_med) for y in yaw_diff])) <= yaw_gate
    if np.sum(yaw_keep) < max(5, int(0.4 * len(yaw_diff))):
        yaw_keep = np.ones_like(yaw_keep, dtype=bool)
    yaw = np.arctan2(np.mean(np.sin(yaw_diff[yaw_keep])), np.mean(np.cos(yaw_diff[yaw_keep])))
    R = rmat_from_yaw(yaw)

    # 2) Robust translation from torso positions.
    torso_delta = ref_torso_pos_n - (obs_torso_pos @ R.T)
    t, t_keep = robust_mean(torso_delta, k=3.5)

    T_ref_lab = pose_to_T(R, t)

    # 3) Diagnostics.
    torso_pred = (obs_torso_pos @ R.T) + t
    torso_err = torso_pred - ref_torso_pos_n
    torso_rmse = np.sqrt(np.mean(np.sum(torso_err * torso_err, axis=1)))

    diag = {
        "n_used": int(n),
        "yaw_deg": float(np.degrees(yaw)),
        "translation_xyz_m": [float(x) for x in t],
        "torso_rmse_m": float(torso_rmse),
        "yaw_inlier_count": int(np.sum(yaw_keep)),
        "trans_inlier_count": int(np.sum(t_keep)),
    }

    if obs["has_obj"]:
        obs_obj = obs["obj_pos"][:n]
        ref_obj = ref_obj_pos[sl]
        obj_pred = (obs_obj @ R.T) + t
        obj_err = obj_pred - ref_obj
        obj_rmse = np.sqrt(np.mean(np.sum(obj_err * obj_err, axis=1)))
        diag["obj_rmse_m"] = float(obj_rmse)

    out = {
        "T_ref_lab": T_ref_lab.tolist(),
        "R_ref_lab": R.tolist(),
        "t_ref_lab": [float(x) for x in t],
        "diagnostics": diag,
        "meta": {
            "obs_csv": args.obs_csv,
            "ref_npz": args.ref_npz,
            "ref_start_frame": int(args.ref_start_frame),
            "num_frames": int(n),
            "method": "yaw-only robust init (median gate + circular mean + MAD translation filter)",
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("======================================")
    print("T_ref_lab initialization complete")
    print("======================================")
    print(f"out: {out_path}")
    print(f"n_used={diag['n_used']}")
    print(f"yaw_deg={diag['yaw_deg']:+.3f}")
    print(
        "t_xyz_m=[{:+.4f},{:+.4f},{:+.4f}]".format(
            diag["translation_xyz_m"][0],
            diag["translation_xyz_m"][1],
            diag["translation_xyz_m"][2],
        )
    )
    print(f"torso_rmse_m={diag['torso_rmse_m']:.4f}")
    if "obj_rmse_m" in diag:
        print(f"obj_rmse_m={diag['obj_rmse_m']:.4f}")
    print(
        f"inliers yaw/trans = {diag['yaw_inlier_count']}/{diag['n_used']} , "
        f"{diag['trans_inlier_count']}/{diag['n_used']}"
    )


if __name__ == "__main__":
    main()
