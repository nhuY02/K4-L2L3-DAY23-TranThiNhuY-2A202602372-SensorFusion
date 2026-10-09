"""Bonus script: CVAT export + BEV visualization + calibration analysis.

Run from repo root:
    python student/bonus/run_bonus.py

Produces:
    student/bonus/tracks_cvat.json          (Bonus 1 - CVAT export)
    student/bonus/bev_frame_*.png           (Bonus 2 - BEV visualization)
    student/bonus/calibration_analysis.png  (Bonus 3 - calibration)
"""
from __future__ import annotations

import json
import sys
import copy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- path setup ----
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "platform"))
sys.path.insert(0, str(REPO_ROOT / "student"))

from fusion_lab.tracking.filter import Filter
from fusion_lab.tracking.manager import TrackManager, Track
from fusion_lab.tracking.sensors import Sensor
from fusion_lab.workspace_loader import load_workspace
from fusion_lab.scripts.run_lab import _load_paths_config, _resolve_weights, _lidar_observations, _front_observations
from fusion_lab.evaluation import valid_ground_truth, tracking_counts, detection_counts, aggregate_records, record_json
from fusion_lab.export_cvat import export_tracks_json
from fusion_lab import tracking_params

import torch

BONUS_DIR = REPO_ROOT / "student" / "bonus"
BONUS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG = REPO_ROOT / "student" / "config" / "paths.yaml"


# ============================================================
# SHARED HELPERS
# ============================================================

def _load_data(cfg, frame_end=30):
    """Load Waymo frames up to frame_end."""
    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
    tfrecord = Path(cfg["waymo_dir"]) / cfg["segment"]
    reader = WaymoDataFileReader(str(tfrecord))
    frames = []
    for cnt, frame in enumerate(reader):
        if cnt > frame_end:
            break
        frames.append((cnt, frame, dataset_pb2, label_pb2))
    return frames


def _run_tracking_frames(frames, cfg, fusion_mode="lidar", seed=0, extrinsic_offset=None):
    """Run tracking on frames list, return (track_log, records).
    
    extrinsic_offset: if set, shift camera translation by this (x, y, z) numpy array.
    """
    ws = load_workspace()
    kalman = ws["kalman"]
    assoc = ws["association"]
    cam = ws["camera_fusion"]
    bev_ws = ws["bev_mapping"]
    det_pipe = ws["detection_pipeline"]
    det_metrics = ws["detection_metrics"]

    weights = _resolve_weights(cfg)
    det_cfg = det_pipe.load_fpn_resnet_config(str(weights) if weights else None)
    model = det_pipe.create_fpn_model(det_cfg, str(weights) if weights else None)

    from fusion_lab.lidar_pcl import pcl_from_range_image
    from simple_waymo_open_dataset_reader import utils as waymo_utils

    rng = np.random.default_rng(seed)
    KF = Filter(kalman)
    manager = TrackManager(ws["track_management"])
    lidar_sensor = None
    camera_sensor = None
    records = []
    track_log = []  # per-frame: list of {id, x, y, z, state}

    for cnt, frame, dataset_pb2, label_pb2 in frames:
        if lidar_sensor is None:
            lidar_sensor = Sensor(
                "lidar",
                waymo_utils.get(frame.context.laser_calibrations, dataset_pb2.LaserName.TOP),
                cam,
            )
        if fusion_mode == "fused" and camera_sensor is None:
            cam_calib = waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
            camera_sensor = Sensor("camera", cam_calib, cam)
            # Apply optional calibration offset
            if extrinsic_offset is not None:
                T = np.array(camera_sensor.veh_to_sens, dtype=float)
                # Shift translation column (last column of 4x4 transform)
                T[:3, 3] += np.array(extrinsic_offset, dtype=float)
                camera_sensor.veh_to_sens = np.asmatrix(T)
                # Recompute sens_to_veh
                camera_sensor.sens_to_veh = np.asmatrix(np.linalg.inv(T))

        from fusion_lab.lidar_pcl import pcl_from_range_image
        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        tensor = torch.from_numpy(bev_ws.bev_maps_from_pcl(points, det_cfg)).unsqueeze(0).float()
        detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)
        labels = valid_ground_truth(frame.laser_labels, det_cfg, label_pb2.Label.Type.TYPE_VEHICLE)

        observations = _lidar_observations(cnt, detections, lidar_sensor, det_cfg)
        for track in manager.track_list:
            KF.predict(track)
            track.set_t(cnt * tracking_params.dt)
        assoc.associate_and_update(manager, observations, KF, lidar_sensor)

        if fusion_mode == "fused" and camera_sensor is not None:
            obs_cam = _front_observations(frame, cnt, camera_sensor, rng,
                                          dataset_pb2.CameraName.FRONT, label_pb2.Label.Type.TYPE_VEHICLE)
            if obs_cam is not None:
                assoc.associate_and_update(manager, obs_cam, KF, camera_sensor)

        frame_tracks = []
        for t in manager.track_list:
            x = np.asarray(t.x, dtype=float).reshape(-1)
            frame_tracks.append({
                "id": id(t),
                "x": float(x[0]), "y": float(x[1]), "z": float(x[2]),
                "vx": float(x[3]), "vy": float(x[4]),
                "state": t.state, "score": float(t.score),
            })
        track_log.append({"frame": cnt, "tracks": frame_tracks})

        record = {
            "mode": fusion_mode, "frame": cnt,
            **detection_counts(labels, detections, det_metrics),
            "valid_gt": len(labels),
            **tracking_counts(manager.track_list, labels),
        }
        records.append(record)

    return track_log, records


# ============================================================
# BONUS 1: CVAT Export
# ============================================================

def bonus1_cvat_export(track_log):
    """Export track_log to CVAT JSON in student/bonus/."""
    out = BONUS_DIR / "tracks_cvat.json"
    export_tracks_json(track_log, out)
    print(f"[Bonus 1] CVAT JSON exported → {out}")
    return out


# ============================================================
# BONUS 2: BEV Visualization
# ============================================================

COLORS = plt.cm.tab10.colors


def bonus2_bev_visualization(track_log_lidar, track_log_fused, num_frames=5):
    """Draw side-by-side BEV plots comparing lidar-only vs fused tracks."""
    # Collect all track IDs to assign consistent colors
    all_ids_lidar = sorted({t["id"] for f in track_log_lidar for t in f["tracks"]})
    all_ids_fused = sorted({t["id"] for f in track_log_fused for t in f["tracks"]})
    id_color_lidar = {tid: COLORS[i % len(COLORS)] for i, tid in enumerate(all_ids_lidar)}
    id_color_fused = {tid: COLORS[i % len(COLORS)] for i, tid in enumerate(all_ids_fused)}

    saved = []
    # Pick representative frames where there are confirmed tracks
    frames_with_tracks = [
        f for f in track_log_lidar
        if any(t["state"] == "confirmed" for t in f["tracks"])
    ]
    selected = frames_with_tracks[::max(1, len(frames_with_tracks) // num_frames)][:num_frames]

    for entry in selected:
        frame_idx = entry["frame"]

        # Find matching fused entry
        fused_entry = next((f for f in track_log_fused if f["frame"] == frame_idx), None)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"BEV Track Visualization — Frame {frame_idx}", fontsize=14, fontweight="bold")

        for ax, log_entry, id_color, mode_name in [
            (axes[0], entry, id_color_lidar, "LiDAR-only"),
            (axes[1], fused_entry, id_color_fused, "LiDAR + Camera (Fused)"),
        ]:
            ax.set_facecolor("#1a1a2e")
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.set_aspect("equal")
            ax.set_title(mode_name, color="white", fontsize=11)
            ax.tick_params(colors="white")
            ax.set_xlabel("X (m)", color="white")
            ax.set_ylabel("Y (m)", color="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

            # Ego vehicle
            ax.plot(0, 0, "w^", markersize=12, label="Ego", zorder=5)

            if log_entry is not None:
                for t in log_entry["tracks"]:
                    c = id_color.get(t["id"], "gray")
                    marker = "o" if t["state"] == "confirmed" else "s"
                    alpha = 1.0 if t["state"] == "confirmed" else 0.4
                    ax.plot(t["x"], t["y"], marker=marker, color=c,
                            markersize=10, alpha=alpha, zorder=4)
                    ax.annotate(
                        f"ID{t['id'] % 1000:03d}\n({t['state'][0].upper()})",
                        (t["x"], t["y"]), color="white", fontsize=6,
                        ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points",
                    )
                    # Draw velocity arrow
                    ax.arrow(t["x"], t["y"], t["vx"] * 0.5, t["vy"] * 0.5,
                             head_width=0.8, head_length=0.5, fc=c, ec=c, alpha=0.7, zorder=3)

            legend_patches = [
                mpatches.Patch(color="white", label="Ego vehicle (▲)"),
                mpatches.Patch(color="lime", label="● Confirmed track"),
                mpatches.Patch(color="gray", label="■ Tentative track"),
            ]
            ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
                      facecolor="#333", labelcolor="white", framealpha=0.8)

        plt.tight_layout()
        fname = BONUS_DIR / f"bev_frame_{frame_idx:03d}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight", facecolor="#0f0f23")
        plt.close()
        saved.append(fname)
        print(f"[Bonus 2] BEV saved → {fname}")

    return saved


# ============================================================
# BONUS 3: Calibration Analysis
# ============================================================

def bonus3_calibration_analysis(frames, cfg):
    """Run fused tracking with increasing camera extrinsic offset; measure RMSE."""
    offsets = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]  # metres shift in X
    results = []

    for offset in offsets:
        off_vec = np.array([offset, 0.0, 0.0])
        track_log, records = _run_tracking_frames(
            frames, cfg, fusion_mode="fused",
            seed=0, extrinsic_offset=off_vec if offset > 0 else None
        )
        # Compute aggregate
        ws = load_workspace()
        det_pipe = ws["detection_pipeline"]
        weights = _resolve_weights(cfg)
        det_cfg = det_pipe.load_fpn_resnet_config(str(weights) if weights else None)

        matches_total = sum(r["matches"] for r in records)
        sum_sq = sum(r["sum_sq_err"] for r in records)
        ghosts = sum(r["ghosts"] for r in records)
        rmse = float(np.sqrt(sum_sq / matches_total)) if matches_total > 0 else None
        results.append({
            "offset_m": offset,
            "rmse": rmse,
            "matches": matches_total,
            "ghosts": ghosts,
            "gating_rejects": None,  # approximated from drop in matches
        })
        print(f"[Bonus 3] offset={offset:+.1f}m → RMSE={rmse:.4f}m  matches={matches_total}  ghosts={ghosts}")

    # Plot
    off_vals = [r["offset_m"] for r in results]
    rmse_vals = [r["rmse"] for r in results]
    ghosts_vals = [r["ghosts"] for r in results]
    matches_vals = [r["matches"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Calibration Error Analysis: Camera Extrinsic X-offset", fontsize=13, fontweight="bold")

    ax1, ax2, ax3 = axes

    ax1.plot(off_vals, rmse_vals, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax1.axhline(0.45, color="green", linestyle="--", alpha=0.7, label="Full-score threshold (0.45m)")
    ax1.axhline(0.75, color="orange", linestyle="--", alpha=0.7, label="Partial-score threshold (0.75m)")
    ax1.set_xlabel("Camera X-offset (m)")
    ax1.set_ylabel("RMSE (m)")
    ax1.set_title("RMSE vs Calibration Error")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_facecolor("#f8f9fa")

    ax2.plot(off_vals, ghosts_vals, "s-", color="#e67e22", linewidth=2, markersize=8)
    ax2.set_xlabel("Camera X-offset (m)")
    ax2.set_ylabel("Ghost track frames")
    ax2.set_title("Ghost Tracks vs Calibration Error")
    ax2.grid(alpha=0.3)
    ax2.set_facecolor("#f8f9fa")

    ax3.plot(off_vals, matches_vals, "^-", color="#27ae60", linewidth=2, markersize=8)
    ax3.set_xlabel("Camera X-offset (m)")
    ax3.set_ylabel("Total matches")
    ax3.set_title("GT Matches vs Calibration Error")
    ax3.grid(alpha=0.3)
    ax3.set_facecolor("#f8f9fa")

    plt.tight_layout()
    fname = BONUS_DIR / "calibration_analysis.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Bonus 3] Calibration plot saved → {fname}")

    return results, fname


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Running all bonus tasks...")
    print("=" * 60)

    cfg = _load_paths_config(CONFIG)
    # Use first 30 frames for speed (bonus doesn't need full segment)
    frame_end = 49

    print("\n[Loading Waymo frames...]")
    frames = _load_data(cfg, frame_end=frame_end)
    print(f"Loaded {len(frames)} frames.")

    # --- Bonus 1 & 2: need both lidar-only and fused track logs ---
    print("\n--- Bonus 1 & 2: tracking lidar-only ---")
    track_log_lidar, records_lidar = _run_tracking_frames(frames, cfg, fusion_mode="lidar", seed=0)

    print("\n--- Bonus 1 & 2: tracking fused ---")
    track_log_fused, records_fused = _run_tracking_frames(frames, cfg, fusion_mode="fused", seed=0)

    # Bonus 1: CVAT export
    print("\n--- Bonus 1: CVAT Export ---")
    cvat_file = bonus1_cvat_export(track_log_fused)

    # Bonus 2: BEV visualization
    print("\n--- Bonus 2: BEV Visualization ---")
    bev_files = bonus2_bev_visualization(track_log_lidar, track_log_fused, num_frames=5)

    # Bonus 3: Calibration analysis
    print("\n--- Bonus 3: Calibration Analysis ---")
    cal_results, cal_plot = bonus3_calibration_analysis(frames, cfg)

    # Save calibration table to JSON
    cal_json = BONUS_DIR / "calibration_results.json"
    with open(cal_json, "w") as f:
        json.dump(cal_results, f, indent=2)

    print("\n" + "=" * 60)
    print("BONUS COMPLETE. Files generated:")
    print(f"  {cvat_file}")
    for p in bev_files:
        print(f"  {p}")
    print(f"  {cal_plot}")
    print(f"  {cal_json}")
    print("=" * 60)

    return cal_results


if __name__ == "__main__":
    main()
