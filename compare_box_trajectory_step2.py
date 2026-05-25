#!/usr/bin/env python3
"""Compare measured box trajectory (lab -> sim via T_sim_lab) vs NPZ reference.

Validates the T_sim_lab transformation by overlaying:
  * measured box trajectory from tracker CSV (transformed lab -> sim)
  * NPZ reference object_pos_w trajectory (already in sim)

If T_sim_lab is correct AND the user roughly followed the sub8_45 motion,
the two trajectories should be similar in shape (rise -> translate -> drop).

Auto-detects motion start time in the measured trajectory (first frame whose
position deviates > 5cm from initial mean), and aligns it to the NPZ motion's
own motion-start (frame 100 ~ 2s after a long static segment).

Saves three plots:
  outputs/step2_box_3d.png    - 3D trajectory overlay
  outputs/step2_box_xyz.png   - time-axis x/y/z 1D plots
  outputs/step2_box_topdown.png - top-down (sim xy) overlay
"""
import argparse, csv, json, os
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def load_tsim_lab(path):
    d = json.loads(Path(path).read_text())
    R = np.array(d["R_sim_lab_3x3"])
    t = np.array(d["t_sim_lab_xyz"])
    return R, t


def read_box_pose_lab(csv_path):
    ts, ps, qs = [], [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                t = float(row["t_sec"])
                p = [float(row[f"obj_pos_{c}"]) for c in "xyz"]
                q = [float(row[f"obj_quat_{c}"]) for c in ("w", "x", "y", "z")]
            except (KeyError, ValueError, TypeError):
                continue
            if p[0] == 0.0 and p[1] == 0.0 and p[2] == 0.0:
                continue
            if sum(abs(x) for x in q) < 0.5:
                continue
            ts.append(t); ps.append(p); qs.append(q)
    return np.array(ts), np.array(ps), np.array(qs)


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


def unwrap_rpy(rpy_series):
    rpy = rpy_series.copy()
    for k in range(3):
        rpy[:, k] = np.degrees(np.unwrap(np.radians(rpy[:, k])))
    return rpy


def detect_motion_start(pos, thresh=0.05, window=10):
    """Return index of first frame where |pos - mean(pos[:window])| > thresh."""
    if len(pos) < window + 1:
        return 0
    p0 = pos[:window].mean(axis=0)
    for i in range(window, len(pos)):
        if np.linalg.norm(pos[i] - p0) > thresh:
            return i
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracker-csv", required=True)
    p.add_argument("--ref-npz", required=True)
    p.add_argument("--tsim-lab-json", default="config/T_sim_lab.json")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--motion-thresh", type=float, default=0.05,
                   help="m, displacement threshold for motion-start detect")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    # === measured (lab) -> sim ===
    R_sim_lab, t_sim_lab = load_tsim_lab(args.tsim_lab_json)
    t_meas, pos_lab, quat_lab = read_box_pose_lab(args.tracker_csv)
    pos_sim_meas = (R_sim_lab @ pos_lab.T).T + t_sim_lab
    R_lab_meas = np.array([quat_wxyz_to_R(q) for q in quat_lab])
    R_sim_meas = np.einsum("ij,njk->nik", R_sim_lab, R_lab_meas)
    rpy_sim_meas = unwrap_rpy(np.array([R_to_rpy_deg(R) for R in R_sim_meas]))
    print(f"[step2] measured: {len(t_meas)} frames, {t_meas[-1] - t_meas[0]:.2f}s")
    print(f"  pos_lab x range: [{pos_lab[:,0].min():+.3f}, {pos_lab[:,0].max():+.3f}]")
    print(f"  pos_lab y range: [{pos_lab[:,1].min():+.3f}, {pos_lab[:,1].max():+.3f}]")
    print(f"  pos_lab z range: [{pos_lab[:,2].min():+.3f}, {pos_lab[:,2].max():+.3f}]")
    print(f"  pos_sim z (height) range: [{pos_sim_meas[:,2].min():+.3f}, {pos_sim_meas[:,2].max():+.3f}]")

    # === reference (sim) ===
    npz = np.load(args.ref_npz)
    pos_sim_ref = npz["object_pos_w"]
    quat_sim_ref = npz["object_quat_w"]  # (T, 4) wxyz
    R_sim_ref = np.array([quat_wxyz_to_R(q) for q in quat_sim_ref])
    rpy_sim_ref = unwrap_rpy(np.array([R_to_rpy_deg(R) for R in R_sim_ref]))
    fps = float(npz["fps"][0]) if hasattr(npz["fps"], "__len__") else float(npz["fps"])
    t_ref = np.arange(len(pos_sim_ref)) / fps
    print(f"[step2] reference: {len(t_ref)} frames, {t_ref[-1]:.2f}s @ {fps} fps")

    # === time-align: motion start ===
    i_meas = detect_motion_start(pos_sim_meas, thresh=args.motion_thresh)
    i_ref  = detect_motion_start(pos_sim_ref,  thresh=args.motion_thresh)
    t_meas_aligned = t_meas - t_meas[i_meas]
    t_ref_aligned  = t_ref  - t_ref[i_ref]
    print(f"[step2] motion-start detected:")
    print(f"  measured: frame {i_meas}, t={t_meas[i_meas]:.2f}s")
    print(f"  ref:      frame {i_ref}, t={t_ref[i_ref]:.2f}s")

    # === 3D plot ===
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos_sim_ref[:, 0], pos_sim_ref[:, 1], pos_sim_ref[:, 2],
            "-", color="black", lw=2, label=f"NPZ ref ({len(pos_sim_ref)} fr)")
    ax.scatter(*pos_sim_ref[0], color="green", s=80, label="ref start")
    ax.scatter(*pos_sim_ref[-1], color="red", s=80, label="ref end")
    ax.plot(pos_sim_meas[:, 0], pos_sim_meas[:, 1], pos_sim_meas[:, 2],
            "-", color="C0", lw=1.5, alpha=0.8, label=f"measured (sim) ({len(pos_sim_meas)} fr)")
    ax.scatter(*pos_sim_meas[0], color="lime", marker="^", s=80, label="meas start")
    ax.scatter(*pos_sim_meas[-1], color="orange", marker="^", s=80, label="meas end")
    ax.set_xlabel("sim x"); ax.set_ylabel("sim y"); ax.set_zlabel("sim z (up)")
    ax.set_title("Box trajectory in sim frame")
    ax.legend(loc="upper left", fontsize=8)
    # equal aspect
    pts = np.vstack([pos_sim_ref, pos_sim_meas])
    rng = pts.max(axis=0) - pts.min(axis=0)
    c = (pts.max(axis=0) + pts.min(axis=0)) / 2
    r = rng.max() / 2 * 1.1
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    plt.tight_layout()
    out_3d = out_dir / "step2_box_3d.png"
    plt.savefig(out_3d, dpi=120); plt.close()
    print(f"[step2] saved {out_3d}")

    # === time-axis 1D plots ===
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["sim x", "sim y", "sim z (height)"]
    for k, ax in enumerate(axes):
        ax.plot(t_ref_aligned, pos_sim_ref[:, k], "-", color="black", lw=2,
                label="NPZ ref" if k == 0 else None)
        ax.plot(t_meas_aligned, pos_sim_meas[:, k], "-", color="C0", lw=1.5,
                label="measured (sim)" if k == 0 else None)
        ax.set_ylabel(labels[k])
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("time (s, aligned to motion-start)")
    axes[0].set_title("Box trajectory: measured (lab→sim) vs NPZ ref")
    plt.tight_layout()
    out_xyz = out_dir / "step2_box_xyz.png"
    plt.savefig(out_xyz, dpi=120); plt.close()
    print(f"[step2] saved {out_xyz}")

    # === top-down (sim xy) plot ===
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(pos_sim_ref[:, 0], pos_sim_ref[:, 1], "-", color="black", lw=2, label="NPZ ref")
    ax.scatter(pos_sim_ref[0, 0], pos_sim_ref[0, 1], color="green", s=80, label="ref start")
    ax.scatter(pos_sim_ref[-1, 0], pos_sim_ref[-1, 1], color="red", s=80, label="ref end")
    ax.plot(pos_sim_meas[:, 0], pos_sim_meas[:, 1], "-", color="C0", lw=1.5, alpha=0.8,
            label="measured (sim)")
    ax.scatter(pos_sim_meas[0, 0], pos_sim_meas[0, 1], color="lime", marker="^", s=80,
               label="meas start")
    ax.scatter(pos_sim_meas[-1, 0], pos_sim_meas[-1, 1], color="orange", marker="^", s=80,
               label="meas end")
    ax.set_aspect("equal")
    ax.set_xlabel("sim x"); ax.set_ylabel("sim y")
    ax.set_title("Top-down (sim xy): box trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out_top = out_dir / "step2_box_topdown.png"
    plt.savefig(out_top, dpi=120); plt.close()
    print(f"[step2] saved {out_top}")

    # === orientation plot (sim frame RPY over time) ===
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    rpy_labels = ["roll (deg)", "pitch (deg)", "yaw (deg)"]
    for k, ax in enumerate(axes):
        ax.plot(t_ref_aligned, rpy_sim_ref[:, k], "-", color="black", lw=2,
                label="NPZ ref" if k == 0 else None)
        ax.plot(t_meas_aligned, rpy_sim_meas[:, k], "-", color="C0", lw=1.5,
                label="measured (sim)" if k == 0 else None)
        ax.set_ylabel(rpy_labels[k])
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("time (s, aligned to motion-start)")
    axes[0].set_title("Box ORIENTATION (sim frame, mjlab body conv): measured vs NPZ ref")
    plt.tight_layout()
    out_rpy = out_dir / "step2_box_rpy.png"
    plt.savefig(out_rpy, dpi=120); plt.close()
    print(f"[step2] saved {out_rpy}")

    # === geodesic angle difference (only at overlapping motion-start-aligned times) ===
    # Resample ref onto measured times (using motion-start aligned timeline).
    R_sim_ref_at_meas = np.empty_like(R_sim_meas)
    for i, t in enumerate(t_meas_aligned):
        if t < t_ref_aligned[0]:
            R_sim_ref_at_meas[i] = R_sim_ref[0]
        elif t > t_ref_aligned[-1]:
            R_sim_ref_at_meas[i] = R_sim_ref[-1]
        else:
            j = int(np.argmin(np.abs(t_ref_aligned - t)))
            R_sim_ref_at_meas[i] = R_sim_ref[j]
    geo = np.array([geodesic_deg(R_sim_meas[i], R_sim_ref_at_meas[i])
                    for i in range(len(R_sim_meas))])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_meas_aligned, geo, "-", color="C3", lw=1.5)
    ax.set_xlabel("time (s, aligned to motion-start)")
    ax.set_ylabel("orientation diff (deg, geodesic)")
    ax.set_title("Box orientation residual: measured (sim) vs NPZ ref (sim)")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", lw=0.5)
    plt.tight_layout()
    out_geo = out_dir / "step2_box_orient_diff.png"
    plt.savefig(out_geo, dpi=120); plt.close()
    print(f"[step2] saved {out_geo}")

    # === numeric summary ===
    print()
    print("=== POSITION SUMMARY ===")
    print(f"  ref   start (sim) = {pos_sim_ref[0]}    end = {pos_sim_ref[-1]}")
    print(f"  meas  start (sim) = {pos_sim_meas[0]}    end = {pos_sim_meas[-1]}")
    print(f"  ref   travel    = {pos_sim_ref[-1] - pos_sim_ref[0]}  |.|={np.linalg.norm(pos_sim_ref[-1] - pos_sim_ref[0])*100:.1f}cm")
    print(f"  meas  travel    = {pos_sim_meas[-1] - pos_sim_meas[0]}  |.|={np.linalg.norm(pos_sim_meas[-1] - pos_sim_meas[0])*100:.1f}cm")
    print(f"  ref   max height = {pos_sim_ref[:, 2].max():.3f}m   (peak at frame {pos_sim_ref[:, 2].argmax()})")
    print(f"  meas  max height = {pos_sim_meas[:, 2].max():.3f}m   (peak at frame {pos_sim_meas[:, 2].argmax()})")

    print()
    print("=== ORIENTATION SUMMARY (sim frame RPY, deg) ===")
    print(f"  ref   start RPY = {rpy_sim_ref[0]}    end = {rpy_sim_ref[-1]}")
    print(f"  meas  start RPY = {rpy_sim_meas[0]}    end = {rpy_sim_meas[-1]}")
    print(f"  ref   max |yaw change|   over motion = {np.ptp(rpy_sim_ref[:, 2]):+.2f} deg")
    print(f"  meas  max |yaw change|   over motion = {np.ptp(rpy_sim_meas[:, 2]):+.2f} deg")
    print(f"  ref   max |pitch change| over motion = {np.ptp(rpy_sim_ref[:, 1]):+.2f} deg")
    print(f"  meas  max |pitch change| over motion = {np.ptp(rpy_sim_meas[:, 1]):+.2f} deg")
    g_start = geodesic_deg(R_sim_meas[0], R_sim_ref[0])
    g_end   = geodesic_deg(R_sim_meas[-1], R_sim_ref[-1])
    g_mean  = float(np.mean(geo))
    g_max   = float(np.max(geo))
    print(f"  geodesic angle diff: start={g_start:.2f} deg  end={g_end:.2f} deg  mean={g_mean:.2f}  max={g_max:.2f}")


if __name__ == "__main__":
    main()
