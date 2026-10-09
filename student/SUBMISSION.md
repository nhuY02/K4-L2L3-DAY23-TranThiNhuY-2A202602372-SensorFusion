# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Trần Thị Như Ý
- MSSV: 2A202602372
- Email: y.2a202602372@vinuni.edu.vn
- Link repo (fork): https://github.com/NhuY-2A202602372/K4-L2L3-DAY23-TranThiNhuY-2A202602372-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): f9146840f58f67c2059607504e38b3c5445e32b8

## Tóm tắt kết quả

- `fusion_mode`: `compare`; `frames`: [0, 198]; `segment`: `training_segment-10094743350625019937_3420_000_3440_000_with_camera_labels.tfrecord`; `seed`: 0
- `detection.precision`: 0.9586 (95.9%), `detection.recall`: 0.4021 (40.2%), `tp`: 1019, `fp`: 44, `fn`: 1515
- `tracking.lidar`: `rmse`=0.3462 m, `matches`=1000, `sum_sq_err`=119.848, `ghost_track_frames`=0, `missed_gt_frames`=1534, `mean_confirmed_tracks`=5.025
- `tracking.fused`: `rmse`=0.5180 m, `matches`=987, `sum_sq_err`=264.850, `ghost_track_frames`=2, `missed_gt_frames`=1547, `mean_confirmed_tracks`=4.970
- **Giải thích khác biệt hai mode:** LiDAR-only đạt RMSE thấp hơn (0.346 m vs 0.518 m) với 1000 matches và 0 ghost. Fused mode có RMSE cao hơn vì camera simulation dùng tâm hộp GT 2D có nhiễu seed — khi projection không chính xác, EKF update camera kéo state ra xa vị trí thật. Fused có 2 ghost và ít matches hơn (987 vs 1000) do một số tracks bị camera update làm lệch khỏi cổng ghép với GT. Đây là giới hạn của camera simulation từ nhãn (không phải camera detector thật), không phản ánh hiệu năng fusion thực tế.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

**1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?**

LiDAR đo trực tiếp vị trí 3D `z = (x, y, z)` trong vehicle frame, nên `H` là ma trận tuyến tính 3×6 (lấy 3 hàng đầu state). `R` là ma trận 3×3 diagonal với `sigma_lidar = 0.1 m`. Camera đo pixel 2D `z = (u, v)` sau phép chiếu pinhole: `u = c_i - f_i*(y_s/x_s)`, `v = c_j - f_j*(z_s/x_s)` — phi tuyến theo state 3D, nên cần Jacobian `H` 2×6 (platform tính tự động). `R` là 2×2 diagonal với `sigma_cam = 5.0` pixels. LiDAR có 3 bậc tự do (đủ chiều vị trí), camera chỉ constrain 2 bậc (lateral + vertical sau projection).

**2. Vì sao cần gating Mahalanobis trước khi gán?**

Mahalanobis distance `d² = γᵀ S⁻¹ γ` chuẩn hóa theo innovation covariance `S = HPHᵀ + R`, nên tính được khoảng cách thống kê. Cổng χ² (với `gating_threshold = 0.995`) loại bỏ các cặp có xác suất xuất hiện < 0.5% — ngăn gán nhầm detection xa, giảm ghost track và bảo vệ track khỏi bị kéo bởi outlier. Không có gating, greedy matching có thể gán detection của xe A vào track của xe B nếu chúng gần nhau trong ảnh/BEV.

**3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.**

Pipeline là **track-then-fuse**: chỉ có một danh sách track duy nhất; mỗi frame predict một lần (`E: ekf_predict`), rồi update tuần tự: (1) AssocL — gán lidar + EKF update lidar + quản lý track (score/init/delete), (2) AssocC — gán camera + EKF update camera (không đổi score). Trên `grade_run.log`, mỗi record `(mode, frame)` có đúng một bộ số detection (`det_tp`, `det_fp`, `det_fn`) giống nhau giữa lidar và fused — chứng tỏ detector chạy một lần, không phải fuse sensor rồi detect.

**4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**

Nếu extrinsic bị lệch (ví dụ `t` sai), hàm `camera_measurement_prediction` chiếu state vào pixel sai → `h(x)` lệch so với đo thật `z`. Innovation `γ = z - h(x)` sẽ có **bias hệ thống** (không zero-mean) thay vì nhiễu trắng. `S = HPHᵀ + R` không thay đổi nhưng `d² = γᵀ S⁻¹ γ` tăng, có thể vượt ngưỡng χ² → cổng chặn measurement (gating rejects), camera không update. Nếu lệch nhỏ (không bị chặn), EKF nhận bias → state bị kéo sai → RMSE tăng trên confirmed tracks.

**5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.**

Ngay cả khi `meas_list = []` (frame rỗng), `associate_and_update` vẫn phải gọi `manager.manage_tracks(unassigned_tracks, [], sensor)` để xử lý vòng đời. Nếu không biết `sensor` là lidar hay camera, `manage_tracks` không biết có nên trừ score (miss trong FOV) hay không. **Chỉ lidar** quyết định vòng đời track: hit `+1/window` (tối đa 1), miss trong FOV `-1/window`, score > `confirmed_threshold` → confirmed, score < `delete_threshold` (confirmed) hoặc ≤ 0 (tentative) → xóa. Camera chỉ tinh chỉnh state EKF (không cộng/trừ score, không tạo, không xóa track) vì đo camera 2D không đủ độ tin cậy để ra quyết định tồn tại — thiếu depth và dễ bị occluded.

**6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.**

Trong `update_track_score` (chỉ gọi sau lidar):
- **Khởi tạo:** `init_track_state_from_meas` tạo track với `state = "initialized"`, `score = 1/window`.
- **Xác nhận:** `score > confirmed_threshold (0.8)` → `state = "confirmed"`. Cần ít nhất ~5-6 lần hit liên tiếp trong window=6.
- **Giữ confirmed sau miss:** khi track đã `confirmed` và bị miss (`associated=False`), score bị trừ `1/window` nhưng `state` **không thay đổi** (giữ "confirmed") — code kiểm tra `if track["state"] != "confirmed"` trước khi hạ.
- **Xóa:** `should_delete_track` trả True khi: (1) `P[0,0] > max_P (9.0)` hoặc `P[1,1] > max_P` (phương sai vị trí XY quá lớn), (2) confirmed và `score < delete_threshold (0.6)`, hoặc (3) chưa confirmed và `score <= 0`.

## Bonus (không bắt buộc)

- Không

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity (Claude Sonnet) — AI coding assistant tích hợp trong IDE
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ implement các hàm trong Part E (`kalman.py`), Part F (`association.py`), Part G (`camera_fusion.py`), Part H (`track_management.py`); debug lỗi môi trường; kiểm tra paths.yaml trỏ đúng segment; điền báo cáo SUBMISSION.md
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Chạy `pytest student/tests/ -v` — 122/122 tests PASSED; chạy `fusion-run-lab --fusion compare --seed 0` và xác minh `metrics.json` có `fusion_mode = "compare"`, có cả `tracking.lidar` và `tracking.fused`; đối chiếu công thức EKF với `docs/HUONG_DAN_KY_THUAT.md`

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không sửa
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
