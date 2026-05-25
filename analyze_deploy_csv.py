"""analyze_deploy_csv.py

Post-mortem analysis for a deploy run logged by track_robot_and_box_multicam.py
(the CSV that --csv-out produces). Answers three questions:

  1. Did the tracker keep seeing torso+box throughout the deploy?
     → visibility %, longest-gap durations, n_cams histogram
  2. Did the box stay where calibration thought it was, or did it drift?
     → box position vs calibration mean, max deviation, percentiles
  3. What did the per-step "obs" look like (box-in-torso) vs NPZ ref?
     → already in CSV as obj_in_torso_pos_{x,y,z} + obj_in_torso_rot6d_*.
       Compare quantiles to ref derived from start_pose_calibration.json.

Usage:
  python3 analyze_deploy_csv.py outputs/deploy_run_20260524_231638.csv
  python3 analyze_deploy_csv.py outputs/deploy_run_20260524_231638.csv \
      --calib config/start_pose_calibration.json --start 30 --end 60
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import math
from pathlib import Path

import numpy as np


def _parse_csv_to_columns(path: Path) -> dict:
    """Read CSV into dict of name->np.array. Numeric columns become float (NaN on
    blanks); non-numeric columns stay as object arrays."""
    with path.open("r", newline="") as f:
        rdr = _csv.reader(f)
        header = next(rdr)
        cols = {h: [] for h in header}
        for row in rdr:
            for h, v in zip(header, row):
                cols[h].append(v)
    out = {}
    for h, vals in cols.items():
        try:
            out[h] = np.array([float(v) if v != "" else np.nan for v in vals],
                              dtype=float)
        except ValueError:
            out[h] = np.array(vals, dtype=object)
    return out


def fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100.0 * n / total:5.1f}% ({n}/{total})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="deploy_run_*.csv from tracker")
    p.add_argument("--calib", type=Path, default=None,
                   help="start_pose_calibration.json — to report box drift "
                        "vs the calibrated mean.")
    p.add_argument("--start", type=float, default=None,
                   help="window start [sec] (default = whole file)")
    p.add_argument("--end", type=float, default=None,
                   help="window end [sec] (default = whole file)")
    p.add_argument("--bin", type=float, default=5.0,
                   help="time bin size for per-window summary table [sec]")
    args = p.parse_args()

    cols_all = _parse_csv_to_columns(args.csv)
    n_total = len(cols_all["t_sec"])
    if n_total == 0:
        print("CSV is empty.")
        return

    t_all = cols_all["t_sec"].astype(float)
    duration = float(np.nanmax(t_all) - np.nanmin(t_all))
    avg_rate = n_total / duration if duration > 0 else float("nan")

    mask_all = np.ones(n_total, dtype=bool)
    if args.start is not None:
        mask_all &= (t_all >= args.start)
    if args.end is not None:
        mask_all &= (t_all <= args.end)
    cols = {k: v[mask_all] for k, v in cols_all.items()}
    n = mask_all.sum()
    if n == 0:
        print("Empty window.")
        return

    t = cols["t_sec"].astype(float)
    win_dur = float(t.max() - t.min()) if n > 1 else 0.0

    # torso_path is non-numeric ("head"/"pelvis"/"fused"/"") — present iff visible.
    tp = cols["torso_path"]
    torso_vis = np.array([str(v) not in ("", "nan", "None") for v in tp], dtype=bool)
    box_vis = np.nan_to_num(cols["box_visible"], nan=0).astype(int) > 0
    head_vis = np.nan_to_num(cols["head_visible"], nan=0).astype(int)
    pelv_vis = np.nan_to_num(cols["pelvis_visible"], nan=0).astype(int)
    n_tot_tag = np.nan_to_num(cols["n_total_tags"], nan=0).astype(int)
    both = torso_vis & box_vis

    print("=" * 78)
    print(f" deploy CSV: {args.csv.name}")
    print("=" * 78)
    print(f"  whole file       : {n_total} frames, {duration:.1f} s, "
          f"~{avg_rate:.1f} Hz")
    if args.start is not None or args.end is not None:
        print(f"  window           : {win_dur:.1f} s "
              f"({args.start or t.min():.1f}..{args.end or t.max():.1f}) "
              f"-> {n} frames")
    print()

    print("--- VISIBILITY ---")
    print(f"  torso visible    : {fmt_pct(torso_vis.sum(), n)}")
    print(f"  box   visible    : {fmt_pct(box_vis.sum(), n)}")
    print(f"  both visible     : {fmt_pct(both.sum(), n)}    <-- usable for policy obs")
    print(f"  head tag seen    : {fmt_pct(head_vis.sum(), n)}")
    print(f"  pelvis tag seen  : {fmt_pct(pelv_vis.sum(), n)}")
    print(f"  n_total_tags mean: {n_tot_tag.mean():.1f}  "
          f"(min {n_tot_tag.min()}, max {n_tot_tag.max()})")
    print()

    # ---- gap stats: longest stretch of "both invisible" ----
    invisible = ~both
    if invisible.any():
        # compute run-length encoding of invisible streaks
        diff = np.diff(np.concatenate([[0], invisible.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        gap_durs = []
        for s, e in zip(starts, ends):
            if e <= len(t):
                gap_durs.append(float(t[min(e, len(t)-1)] - t[s]))
        gap_durs = sorted(gap_durs, reverse=True)
        print("--- WORST OBS GAPS (both torso+box invisible at once) ---")
        for i, g in enumerate(gap_durs[:5]):
            print(f"  #{i+1}: {g*1000:.0f} ms")
        if not gap_durs:
            print("  (none)")
        print()

    # ---- box pose stability ----
    box_x = cols["obj_pos_x"].astype(float)
    box_y = cols["obj_pos_y"].astype(float)
    box_z = cols["obj_pos_z"].astype(float)
    box_mask = box_vis & np.isfinite(box_x) & np.isfinite(box_y) & np.isfinite(box_z)
    if box_mask.any():
        bx = box_x[box_mask]; by = box_y[box_mask]; bz = box_z[box_mask]
        print("--- BOX LAB-FRAME POSITION (across visible frames) ---")
        print(f"  mean : [{bx.mean():+.4f}, {by.mean():+.4f}, {bz.mean():+.4f}] m")
        print(f"  std  : [{bx.std()*1000:6.1f}, {by.std()*1000:6.1f}, {bz.std()*1000:6.1f}] mm")
        print(f"  range: [{(bx.max()-bx.min())*1000:6.1f}, "
              f"{(by.max()-by.min())*1000:6.1f}, "
              f"{(bz.max()-bz.min())*1000:6.1f}] mm  "
              f"(if huge -> box was moved or detection jumped)")
        if args.calib is not None and args.calib.exists():
            calib = json.loads(args.calib.read_text())
            cm = np.array(calib["object_pos_mean_xyz_m"], float)
            d = np.stack([bx - cm[0], by - cm[1], bz - cm[2]], axis=1)
            err_mm = np.linalg.norm(d, axis=1) * 1000.0
            print(f"  vs calib mean    : "
                  f"median={np.median(err_mm):.1f} mm, "
                  f"p95={np.percentile(err_mm, 95):.1f} mm, "
                  f"max={err_mm.max():.1f} mm")
        print()

    # ---- torso pose stability (only when visible) ----
    tx = cols["torso_pos_x"].astype(float)
    ty = cols["torso_pos_y"].astype(float)
    tz = cols["torso_pos_z"].astype(float)
    tor_mask = torso_vis & np.isfinite(tx) & np.isfinite(ty) & np.isfinite(tz)
    if tor_mask.any():
        a, b, c = tx[tor_mask], ty[tor_mask], tz[tor_mask]
        print("--- TORSO LAB-FRAME POSITION ---")
        print(f"  mean : [{a.mean():+.4f}, {b.mean():+.4f}, {c.mean():+.4f}] m")
        print(f"  std  : [{a.std()*1000:6.1f}, {b.std()*1000:6.1f}, {c.std()*1000:6.1f}] mm")
        print(f"  range: [{(a.max()-a.min())*1000:6.1f}, "
              f"{(b.max()-b.min())*1000:6.1f}, "
              f"{(c.max()-c.min())*1000:6.1f}] mm  "
              f"(big values = robot actually moved during deploy)")
        if args.calib is not None and args.calib.exists():
            calib = json.loads(args.calib.read_text())
            cm = np.array(calib["torso_pos_mean_xyz_m"], float)
            d = np.stack([a - cm[0], b - cm[1], c - cm[2]], axis=1)
            err_mm = np.linalg.norm(d, axis=1) * 1000.0
            print(f"  vs calib mean    : "
                  f"median={np.median(err_mm):.1f} mm, "
                  f"p95={np.percentile(err_mm, 95):.1f} mm, "
                  f"max={err_mm.max():.1f} mm")
        print()

    # ---- box-in-torso (= roughly object_pos_torso obs the policy reads) ----
    oit_cols = ["obj_in_torso_pos_x", "obj_in_torso_pos_y", "obj_in_torso_pos_z"]
    if all(c in cols for c in oit_cols):
        oit_mask = both
        if oit_mask.any():
            ax = cols["obj_in_torso_pos_x"].astype(float)[oit_mask]
            ay = cols["obj_in_torso_pos_y"].astype(float)[oit_mask]
            az = cols["obj_in_torso_pos_z"].astype(float)[oit_mask]
            print("--- BOX-IN-TORSO (= deploy obs object_pos_torso, lab-derived) ---")
            print(f"  mean : [{ax.mean():+.4f}, {ay.mean():+.4f}, {az.mean():+.4f}] m")
            print(f"  std  : [{ax.std()*1000:6.1f}, {ay.std()*1000:6.1f}, {az.std()*1000:6.1f}] mm")
            print("  (this is in tracker head-tag torso frame, NOT yet in mjlab body conv. "
                  "Still useful as a noise indicator.)")
            print()

    # ---- per-window summary table ----
    print(f"--- PER-{args.bin:.0f}-SEC WINDOW SUMMARY ---")
    print("   t_start..t_end | both_vis% | torso% | box% | n_tags_mean | "
          "box_pos_std(mm) | torso_pos_std(mm)")
    print("  " + "-" * 100)
    edges = np.arange(t.min(), t.max() + args.bin, args.bin)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (t >= lo) & (t < hi)
        if m.sum() == 0:
            continue
        tv = torso_vis[m].sum() / m.sum()
        bv = box_vis[m].sum() / m.sum()
        bm = both[m].sum() / m.sum()
        nt = n_tot_tag[m].mean()
        bms = box_mask & m
        tms = tor_mask & m
        bstd = (np.std(box_x[bms]) ** 2 + np.std(box_y[bms]) ** 2 +
                np.std(box_z[bms]) ** 2) ** 0.5 * 1000.0 if bms.sum() > 1 else float("nan")
        tstd = (np.std(tx[tms]) ** 2 + np.std(ty[tms]) ** 2 +
                np.std(tz[tms]) ** 2) ** 0.5 * 1000.0 if tms.sum() > 1 else float("nan")
        flag = ""
        if bm < 0.8:
            flag += " [obs-loss]"
        if bms.sum() > 1 and bstd > 50:
            flag += " [box-jitter]"
        if tms.sum() > 1 and tstd > 100:
            flag += " [torso-moving]"
        print(f"   {lo:6.1f}..{hi:6.1f}s |   {bm*100:5.1f}% | "
              f"{tv*100:5.1f}% | {bv*100:5.1f}% |    {nt:5.1f}   |  "
              f"{bstd:8.1f}   |   {tstd:8.1f}{flag}")
    print()

    print("=" * 78)
    print(" INTERPRETATION CHEATSHEET")
    print("=" * 78)
    print("  both_vis >= 95%, no flags          ->  tracker was solid; problem is policy/setup")
    print("  both_vis < 80%, [obs-loss] flag    ->  policy got 0/I fallback obs often")
    print("  box_pos_std > 50mm in a window     ->  box detection jumped (wrong tag id?)")
    print("  torso_pos_std > 100mm in a window  ->  robot WAS moving (good if policy active)")
    print("  Look for the time window when policy was active (R1+Y press),")
    print("  re-run with --start <s> --end <s> for a focused report.")


if __name__ == "__main__":
    main()
