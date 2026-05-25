#!/usr/bin/env python3
"""Compare measured torso_link + pelvis_link trajectories (lab -> sim via
T_sim_lab) against NPZ reference trajectories during a motion replay.

Validates Step 3 multi-tag fusion + the overall lab-to-sim mapping by
overlaying:
  * measured torso_link    (from tracker CSV, transformed lab -> sim)
  * measured pelvis_link   (idem)
  * NPZ ref body[15] torso (already sim, mjlab body conv)
  * NPZ ref body[0]  pelvis

Auto-detects motion-start (first frame >5cm displacement from initial mean
on either pelvis OR torso) and aligns it to NPZ motion-start.

Saves plots:
  outputs/step3_torso_3d.png      - 3D trajectory overlay (torso & pelvis)
  outputs/step3_torso_xyz.png     - time-axis x/y/z (torso & pelvis)
  outputs/step3_torso_rpy.png     - orientation RPY over time
  outputs/step3_torso_orient_diff.png  - per-frame geodesic angle diff
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def load_tsim_lab(path):
    d = json.loads(Path(path).read_text())
    return np.array(d["R_sim_lab_3x3"]), np.array(d["t_sim_lab_xyz"])


def quat_wxyz_to_R(q):
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def R_to_rpy_deg(R):
    sy = -R[2, 0]
    cy = (R[0, 0] ** 2 + R[1, 0] ** 2) ** 0.5
    if cy > 1e-6:
        roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(sy, cy))
        yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(sy, cy))
        yaw   = 0.0
    return np.array([roll, pitch, yaw])


def geodesic_deg(R_a, R_b):
    R = R_a @ R_b.T
    cos_th = max(-1.0, min(1.0, (np.trace(R) - 1.0) * 0.5))
    return np.degrees(np.arccos(cos_th))


def unwrap_rpy(rpy):
    out = rpy.copy()
    for k in range(3):
        out[:, k] = np.degrees(np.unwrap(np.radians(out[:, k])))
    return out


def read_link_poses(csv_path, prefix):
    """Read pose columns prefixed with `prefix`_pos_{xyz} / `prefix`_quat_{wxyz}."""
    ts, ps, qs = [], [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                t = float(row["t_sec"])
                p = [float(row[f"{prefix}_pos_{c}"]) for c in "xyz"]
                q = [float(row[f"{prefix}_quat_{c}"]) for c in ("w", "x", "y", "z")]
            except (KeyError, ValueError, TypeError):
                continue
            if p[0] == 0.0 and p[1] == 0.0 and p[2] == 0.0:
                continue
            if sum(abs(x) for x in q) < 0.5:
                continue
            ts.append(t); ps.append(p); qs.append(q)
    return np.array(ts), np.array(ps), np.array(qs)


def detect_motion_start(pos, thresh=0.05, window=10):
    if len(pos) < window + 1:
        return 0
    p0 = pos[:window].mean(axis=0)
    for i in range(window, len(pos)):
        if np.linalg.norm(pos[i] - p0) > thresh:
            return i
    return 0


def transform_pose_lab_to_sim(R_sim_lab, t_sim_lab, pos_lab, R_lab):
    pos_sim = (R_sim_lab @ pos_lab.T).T + t_sim_lab
    R_sim = np.einsum("ij,njk->nik", R_sim_lab, R_lab)
    return pos_sim, R_sim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracker-csv", required=True)
    p.add_argument("--ref-npz", required=True)
    p.add_argument("--tsim-lab-json", default="config/T_sim_lab.json")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--motion-thresh", type=float, default=0.05,
                   help="m, displacement threshold for motion-start detect")
    p.add_argument("--torso-body-idx", type=int, default=15,
                   help="NPZ body index for torso_link (mjlab G1 default 15)")
    p.add_argument("--pelvis-body-idx", type=int, default=0,
                   help="NPZ body index for pelvis (mjlab G1 default 0)")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    R_sim_lab, t_sim_lab = load_tsim_lab(args.tsim_lab_json)

    # ===== Measured (lab) -> sim =====
    t_torso, torso_pos_lab, torso_quat_lab = read_link_poses(args.tracker_csv, "torso_link")
    t_pelv,  pelv_pos_lab,  pelv_quat_lab  = read_link_poses(args.tracker_csv, "pelvis_link")
    R_lab_torso = np.array([quat_wxyz_to_R(q) for q in torso_quat_lab])
    R_lab_pelv  = np.array([quat_wxyz_to_R(q) for q in pelv_quat_lab])
    torso_pos_sim, R_sim_torso = transform_pose_lab_to_sim(R_sim_lab, t_sim_lab,
                                                            torso_pos_lab, R_lab_torso)
    pelv_pos_sim, R_sim_pelv   = transform_pose_lab_to_sim(R_sim_lab, t_sim_lab,
                                                            pelv_pos_lab,  R_lab_pelv)
    rpy_torso = unwrap_rpy(np.array([R_to_rpy_deg(R) for R in R_sim_torso]))
    rpy_pelv  = unwrap_rpy(np.array([R_to_rpy_deg(R) for R in R_sim_pelv]))
    print(f"[step3] measured torso : {len(t_torso)} frames, "
          f"{(t_torso[-1] - t_torso[0]) if len(t_torso) else 0:.2f}s")
    print(f"[step3] measured pelvis: {len(t_pelv)} frames, "
          f"{(t_pelv[-1] - t_pelv[0]) if len(t_pelv) else 0:.2f}s")
    print(f"  torso_link  pos_sim z range: [{torso_pos_sim[:,2].min():+.3f}, {torso_pos_sim[:,2].max():+.3f}]")
    print(f"  pelvis_link pos_sim z range: [{pelv_pos_sim[:,2].min():+.3f}, {pelv_pos_sim[:,2].max():+.3f}]")

    # ===== NPZ reference =====
    npz = np.load(args.ref_npz)
    body_pos  = npz["body_pos_w"]    # (T, B, 3) sim
    body_quat = npz["body_quat_w"]   # (T, B, 4) wxyz
    fps = float(npz["fps"][0]) if hasattr(npz["fps"], "__len__") else float(npz["fps"])
    t_ref = np.arange(body_pos.shape[0]) / fps
    torso_pos_ref  = body_pos[:, args.torso_body_idx, :]
    pelv_pos_ref   = body_pos[:, args.pelvis_body_idx, :]
    torso_quat_ref = body_quat[:, args.torso_body_idx, :]
    pelv_quat_ref  = body_quat[:, args.pelvis_body_idx, :]
    R_ref_torso = np.array([quat_wxyz_to_R(q) for q in torso_quat_ref])
    R_ref_pelv  = np.array([quat_wxyz_to_R(q) for q in pelv_quat_ref])
    rpy_torso_ref = unwrap_rpy(np.array([R_to_rpy_deg(R) for R in R_ref_torso]))
    rpy_pelv_ref  = unwrap_rpy(np.array([R_to_rpy_deg(R) for R in R_ref_pelv]))
    print(f"[step3] reference: {len(t_ref)} frames, {t_ref[-1]:.2f}s @ {fps} fps")

    # ===== Time-align: use torso position for motion-start detect on each stream =====
    i_torso = detect_motion_start(torso_pos_sim, thresh=args.motion_thresh)
    i_pelv  = detect_motion_start(pelv_pos_sim,  thresh=args.motion_thresh)
    i_ref   = detect_motion_start(torso_pos_ref, thresh=args.motion_thresh)
    t_torso_a = t_torso - t_torso[i_torso]
    t_pelv_a  = t_pelv  - t_pelv[i_pelv]
    t_ref_a   = t_ref   - t_ref[i_ref]
    print(f"[step3] motion-start: torso frame {i_torso} (t={t_torso[i_torso]:.2f}s) "
          f"pelvis frame {i_pelv} (t={t_pelv[i_pelv]:.2f}s) "
          f"ref frame {i_ref} (t={t_ref[i_ref]:.2f}s)")

    # ===== 3D plot =====
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(torso_pos_ref[:, 0], torso_pos_ref[:, 1], torso_pos_ref[:, 2],
            "-", color="black", lw=2, label="NPZ torso")
    ax.plot(torso_pos_sim[:, 0], torso_pos_sim[:, 1], torso_pos_sim[:, 2],
            "-", color="C0", lw=1.5, alpha=0.9, label="measured torso (sim)")
    ax.plot(pelv_pos_ref[:, 0], pelv_pos_ref[:, 1], pelv_pos_ref[:, 2],
            "--", color="gray", lw=2, label="NPZ pelvis")
    ax.plot(pelv_pos_sim[:, 0], pelv_pos_sim[:, 1], pelv_pos_sim[:, 2],
            "--", color="C1", lw=1.5, alpha=0.9, label="measured pelvis (sim)")
    ax.scatter(*torso_pos_ref[0],  color="green", s=70, label="ref start")
    ax.scatter(*torso_pos_ref[-1], color="red",   s=70, label="ref end")
    ax.scatter(*torso_pos_sim[0],  color="lime",  marker="^", s=70, label="meas start")
    ax.scatter(*torso_pos_sim[-1], color="orange",marker="^", s=70, label="meas end")
    ax.set_xlabel("sim x"); ax.set_ylabel("sim y"); ax.set_zlabel("sim z (up)")
    ax.set_title("Torso & Pelvis trajectory (sim frame)")
    ax.legend(loc="upper left", fontsize=8)
    pts = np.vstack([torso_pos_ref, torso_pos_sim, pelv_pos_ref, pelv_pos_sim])
    rng = pts.max(axis=0) - pts.min(axis=0)
    c = (pts.max(axis=0) + pts.min(axis=0)) / 2
    r = rng.max() / 2 * 1.1
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    plt.tight_layout()
    out_3d = out_dir / "step3_torso_3d.png"
    plt.savefig(out_3d, dpi=120); plt.close()
    print(f"[step3] saved {out_3d}")

    # ===== xyz time plot =====
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    labels = ["sim x", "sim y", "sim z"]
    for k in range(3):
        ax = axes[k, 0]
        ax.plot(t_ref_a,   torso_pos_ref[:, k], "-", color="black", lw=2,
                label="NPZ" if k == 0 else None)
        ax.plot(t_torso_a, torso_pos_sim[:, k], "-", color="C0", lw=1.5,
                label="meas" if k == 0 else None)
        ax.set_ylabel(f"torso {labels[k]}"); ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right"); ax.set_title("torso_link")
        ax = axes[k, 1]
        ax.plot(t_ref_a,  pelv_pos_ref[:, k], "-", color="black", lw=2)
        ax.plot(t_pelv_a, pelv_pos_sim[:, k], "-", color="C1", lw=1.5)
        ax.set_ylabel(f"pelvis {labels[k]}"); ax.grid(True, alpha=0.3)
        if k == 0:
            ax.set_title("pelvis (root)")
    axes[-1, 0].set_xlabel("time (s, aligned to motion-start)")
    axes[-1, 1].set_xlabel("time (s, aligned to motion-start)")
    plt.tight_layout()
    out_xyz = out_dir / "step3_torso_xyz.png"
    plt.savefig(out_xyz, dpi=120); plt.close()
    print(f"[step3] saved {out_xyz}")

    # ===== RPY plot =====
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    rpy_lab = ["roll (deg)", "pitch (deg)", "yaw (deg)"]
    for k in range(3):
        ax = axes[k, 0]
        ax.plot(t_ref_a,   rpy_torso_ref[:, k], "-", color="black", lw=2,
                label="NPZ" if k == 0 else None)
        ax.plot(t_torso_a, rpy_torso[:, k], "-", color="C0", lw=1.5,
                label="meas" if k == 0 else None)
        ax.set_ylabel(f"torso {rpy_lab[k]}"); ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right"); ax.set_title("torso_link")
        ax = axes[k, 1]
        ax.plot(t_ref_a,  rpy_pelv_ref[:, k], "-", color="black", lw=2)
        ax.plot(t_pelv_a, rpy_pelv[:, k], "-", color="C1", lw=1.5)
        ax.set_ylabel(f"pelvis {rpy_lab[k]}"); ax.grid(True, alpha=0.3)
        if k == 0:
            ax.set_title("pelvis")
    axes[-1, 0].set_xlabel("time (s)"); axes[-1, 1].set_xlabel("time (s)")
    plt.tight_layout()
    out_rpy = out_dir / "step3_torso_rpy.png"
    plt.savefig(out_rpy, dpi=120); plt.close()
    print(f"[step3] saved {out_rpy}")

    # ===== Geodesic orientation diff (resample ref onto meas times) =====
    def _resample_R(R_src, t_src, t_dst):
        out = np.empty((len(t_dst), 3, 3))
        for i, t in enumerate(t_dst):
            if t < t_src[0]:
                out[i] = R_src[0]
            elif t > t_src[-1]:
                out[i] = R_src[-1]
            else:
                j = int(np.argmin(np.abs(t_src - t)))
                out[i] = R_src[j]
        return out

    R_torso_ref_at_meas = _resample_R(R_ref_torso, t_ref_a, t_torso_a)
    R_pelv_ref_at_meas  = _resample_R(R_ref_pelv,  t_ref_a, t_pelv_a)
    geo_torso = np.array([geodesic_deg(R_sim_torso[i], R_torso_ref_at_meas[i])
                          for i in range(len(R_sim_torso))])
    geo_pelv  = np.array([geodesic_deg(R_sim_pelv[i],  R_pelv_ref_at_meas[i])
                          for i in range(len(R_sim_pelv))])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_torso_a, geo_torso, "-", color="C0", lw=1.5, label="torso")
    ax.plot(t_pelv_a,  geo_pelv,  "-", color="C1", lw=1.5, label="pelvis")
    ax.set_xlabel("time (s, aligned to motion-start)")
    ax.set_ylabel("orientation diff (deg, geodesic)")
    ax.set_title("Torso / Pelvis orientation residual: measured vs NPZ ref")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right")
    ax.axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    out_geo = out_dir / "step3_torso_orient_diff.png"
    plt.savefig(out_geo, dpi=120); plt.close()
    print(f"[step3] saved {out_geo}")

    # ===== Numeric summary =====
    print()
    print("=== POSITION SUMMARY (sim frame) ===")
    print(f"  torso: ref start={torso_pos_ref[0]}   end={torso_pos_ref[-1]}")
    print(f"         meas start={torso_pos_sim[0]}   end={torso_pos_sim[-1]}")
    print(f"         z range:  ref [{torso_pos_ref[:,2].min():+.3f}, {torso_pos_ref[:,2].max():+.3f}]"
          f"   meas [{torso_pos_sim[:,2].min():+.3f}, {torso_pos_sim[:,2].max():+.3f}]")
    print(f"  pelvis: ref start={pelv_pos_ref[0]}   end={pelv_pos_ref[-1]}")
    print(f"          meas start={pelv_pos_sim[0]}   end={pelv_pos_sim[-1]}")

    print()
    print("=== ORIENTATION RESIDUAL (geodesic angle, deg) ===")
    print(f"  torso : start={geo_torso[0]:6.2f}  end={geo_torso[-1]:6.2f}  "
          f"mean={geo_torso.mean():6.2f}  max={geo_torso.max():6.2f}")
    print(f"  pelvis: start={geo_pelv[0]:6.2f}  end={geo_pelv[-1]:6.2f}  "
          f"mean={geo_pelv.mean():6.2f}  max={geo_pelv.max():6.2f}")


if __name__ == "__main__":
    main()
