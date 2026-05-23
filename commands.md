# RealSense AprilTag 트래킹 — 자주 쓰는 커맨드 모음

모든 명령은 다음 환경에서 실행:

```bash
conda activate unitree_rl_mjlab
cd /home/roy/realsense_calib
```

좌표 컨벤션 한 번에 짚기:
- `--origin-id`를 floor tag(보통 id=1)로 두면 그 태그가 world origin.
- `pupil_apriltags`는 OpenCV solvePnP 컨벤션을 따라 tag local Z축이 "태그 뒷면(=지면 속)" 방향. 그래서 face-up floor tag 기준 world frame은 **+Z = 아래 / −Z = 위**.
- `track_robot_and_box*.py`의 `--head-z-offset`, `--pelvis-to-root-z`, `--root-to-torso-z` 디폴트는 모두 이 컨벤션 기준.

카메라 시리얼 / 캘리브 파일 매핑:

| 카메라 | serial | calib npz |
|---|---|---|
| cam1 (D435i) | `935322072654` | `camera1_935322072654_calibration.npz` |
| cam2 (D435)  | `115222071236` | `camera2_115222071236_calibration.npz` |
| cam3 (D435)  | `112322072671` | `camera3_112322072671_calibration.npz` |

extrinsic 파일 (cam1↔cam2, cam3↔cam2 두 개만 잡으면 cam1↔cam3는 chain으로 처리됨):

| 파일 | 의미 |
|---|---|
| `camera1_to_camera2_extrinsic.npz` | cam1→cam2 (`T_c2_c1` 키) |
| `camera3_to_camera2_extrinsic.npz` | cam3→cam2 (`T_c2_c1` 키, cam3 측이지만 키 이름은 통일) |

---

## 1. 단일 카메라 트래커 — `track_robot_and_box.py`

floor tag(id=1) 기준 world에서 head/pelvis/box를 한 카메라만 보고 추적. 기본은 박스 태그 개별 ID는 숨기고 fused box pose만 표시. `--show-box-tags`로 박스 패널 켤 수 있음.

### cam1
```bash
python track_robot_and_box.py \
  --cam-serial 935322072654 \
  --cam-calib camera1_935322072654_calibration.npz \
  --origin-id 1 \
  --csv-out shadow_log_cam1.csv \
  --print-every 30
```

### cam2
```bash
python track_robot_and_box.py \
  --cam-serial 115222071236 \
  --cam-calib camera2_115222071236_calibration.npz \
  --origin-id 1 \
  --csv-out shadow_log_cam2.csv \
  --print-every 30
```

### cam3
```bash
python track_robot_and_box.py \
  --cam-serial 112322072671 \
  --cam-calib camera3_112322072671_calibration.npz \
  --origin-id 1 \
  --csv-out shadow_log_cam3.csv \
  --print-every 30
```

### 자주 쓰는 옵션
- `--show-box-tags` : 박스 태그 개별 좌표 + 우상단 box 패널 표시 (기본 OFF).
- `--no-show-robot-tags` : 로봇 태그 좌표/공식 패널 OFF (기본 ON).
- `--torso-source head` / `pelvis` / `fused` (기본 fused).
- `--no-origin-hold` : floor tag 잠깐 가려질 때 직전 origin pose 유지 OFF.

---

## 2. 멀티 카메라 트래커 — `track_robot_and_box_multicam.py`

여러 카메라가 동시에 보고 있을 때 **decision_margin 가중 융합**으로 더 안정적이고 occlusion에 강한 추적. 카메라 수가 많을수록 효과 큼.

### 알고리즘 — 카메라 extrinsic을 안 쓰고, 바닥의 보조 anchor(tag 10)를 통해 origin을 복원

매 프레임 카메라별로 `T_camN_origin`(=tag 1의 cam 좌표) 을 다음 우선순위로 결정:

| 우선순위 | 조건 | 정의 |
|---|---|---|
| 1. **DIRECT** | 그 카메라가 tag 1(=origin)을 직접 봄 | `T_camN_origin = T_camN_tag1` |
| 2. **VIA_ANCHOR** | tag 1은 안 보이지만 tag 10(보조 anchor)이 보임 | `T_camN_origin = T_camN_anchor @ inv(T_origin_anchor)` |
| 3. **HOLD** | 둘 다 안 보이지만 직전 프레임 origin이 holdable | 직전 `T_camN_origin` |
| 4. **NONE** | 위 다 실패 | 그 카메라는 이번 프레임 fusion 불참 |

`T_origin_anchor` (= tag 1 frame에서 본 tag 10의 pose) 는:
- `config/floor_anchor_transforms.json` 에 미리 측정된 값을 startup 시 카메라별로 로드.
- 런타임에도 **카메라가 같은 프레임에서 tag 1과 tag 10을 동시에 보면 그 카메라의 자체 추정치를 last-seen으로 갱신**. 이로써 카메라별 detection 바이어스를 그 카메라 안에서 흡수.

origin이 잡힌 카메라는 자기가 본 모든 다른 태그(head, pelvis, box)를 origin frame으로 바로 변환:
```
T_origin_tag = inv(T_camN_origin) @ T_camN_tag
weight       = min(origin_path_weight, tag_margin)
```
여러 카메라에서 같은 태그가 잡히면 → translation은 가중 평균, rotation은 SVD-projected 가중 평균.

이후 robot/box 계산은 단일카메라 버전과 동일:
- **head → torso_head**:
  1. `T_tag_torso.npz` 있으면 `torso_head = head_tag @ T_tag_torso` (이게 가장 정확. 기울기/회전 자동 반영)
  2. 없을 때 **`--head-up-mode tag-axis`** (DEFAULT) — head 태그의 LOCAL `--head-tag-down-axis` (G1은 `+z`, pupil_apriltags 컨벤션상 tag +z = into tag = into head = body-down) 방향으로 `--head-to-torso-body` (25 cm) 이동. 머리가 기울어도 척추 축을 따라감.
  3. `--head-up-mode world-z` (legacy) — world `+z` 방향 25 cm 고정. 머리가 똑바로 있을 때만 맞고 기울이면 torso 위치가 실제 위치에서 9-12 cm 빗나감.
- **pelvis → torso_pelvis** (head 안 보일 때 fallback):
  1. **`--pelvis-up-mode tag-axis`** (DEFAULT) — pelvis 태그의 LOCAL `--pelvis-tag-up-axis` (G1은 `-y`) 방향으로 `--pelvis-to-root-body` (5 cm) → `--root-to-torso-body` (20 cm) 이동. 로봇이 굽혔을 때도 척추 축을 따라가므로 정확함.
  2. `--pelvis-up-mode world-z` (legacy) — world `+/-z` 방향 고정 이동.
- `torso(fused) = (torso_head + torso_pelvis)/2` (둘 다 있을 때, rotation은 head 기준)
- `box = mean(per_tag_T @ inv(BOX_T_TAG[id]))`

GUI: 각 카메라 윈도우 좌상단에 `ORIGIN: DIRECT / via ANCHOR id10 / HOLD ({n}f) / NOT SEEN` 표시.
"FUSED" 패널 두 번째 줄에 `cams: cam1=D cam2=A10 cam3=H4   src(D/A/H)=1/1/1   fused_tags=8` 처럼 한눈에 상태 파악.

> **CSV overwrite 방지**: `--csv-out shadow.csv`를 주면 **자동으로 `_YYYYMMDD_HHMMSS`가 확장자 앞에 삽입**됨 (`shadow_20260523_150500.csv`). 정확히 같은 이름으로 쓰고 싶으면 `--csv-no-timestamp` (단, 동일 파일이 이미 있으면 에러로 보호).

### 2-cam (cam1 + cam2)

```bash
python track_robot_and_box_multicam.py \
  --cam1-serial 935322072654 \
  --cam2-serial 115222071236 \
  --cam1-calib camera1_935322072654_calibration.npz \
  --cam2-calib camera2_115222071236_calibration.npz \
  --origin-id 1 \
  --anchor-ids 10 \
  --margin-min 30 \
  --csv-out shadow_log_2cam.csv
```

### 3-cam (cam1 + cam2 + cam3) — 추천

```bash
python track_robot_and_box_multicam.py \
  --cam1-serial 935322072654 \
  --cam2-serial 115222071236 \
  --cam3-serial 112322072671 \
  --cam1-calib camera1_935322072654_calibration.npz \
  --cam2-calib camera2_115222071236_calibration.npz \
  --cam3-calib camera3_112322072671_calibration.npz \
  --origin-id 1 \
  --anchor-ids 10 \
  --margin-min 20 \
  --detector-quad-decimate 2.0 \
  --no-show-cam-windows \
  --csv-out outputs/shadow_log.csv \
  --print-every 30
```

> `--detect-parallel`은 기본 ON이라 명시 안 해도 됨. 끄려면 `--no-detect-parallel` 추가.
> `--detector-nthreads` 기본값은 9 (3 cam × 3 thread, i5-13500HX 같은 14-core CPU sweet spot). CPU에 따라 6/8/12 등으로 조정.

> `--margin-min 30` 이유: pelvis(7,8) 태그는 로봇이 굽힐 때 비스듬해서 mean margin이 35–39로 떨어짐. 기본값 40에서는 pelvis가 거의 잡히지 않음. ORIGIN/ANCHOR(1,10)는 mean 60+이라 30에서도 안전. 종료 시 자동 출력되는 per-tag margin 표를 확인해 본인 환경에 맞게 조정 가능.

### Loop fps 튜닝

기본 세팅(`--detector-quad-decimate 2.0`, `--detect-parallel`, `--no-show-cam-windows`)에서 3-cam 960×540 기준 **~20-25 fps** 기대. 단계별 옵션:

| 옵션 | 효과 | trade-off |
|---|---|---|
| `--detect-parallel` (기본 ON) | 3 카메라 detect 동시 실행 (GIL 풀린 C 호출 → true parallel). 대략 **2× 가속**. | 카메라별 Detector 인스턴스화로 메모리 약간 ↑. CPU core 4개 미만이면 효과 줄어듦. |
| `--no-detect-parallel` | 직렬 실행 (벤치마크 / 저코어 CPU 용) | fps ~50% ↓ |
| `--detector-nthreads 4` (기본) | parallel 모드에선 카메라당 `floor(nthreads/n_cams)` 로 자동 분배 (3 cam → 1 thread/cam). 총 thread는 안 늘어남. | 너무 작은 값 (1~2)에서는 detect 자체가 느려짐. CPU core 8 이상이면 6~8까지 올려도 OK. |
| `--detector-quad-decimate 2.0` (기본) | detection 영상 면적 1/4. 큰 가속. | 작은 태그/먼 거리 detection ↓. pelvis가 25%대로 떨어지면 1.5로 완화. |
| `--detector-quad-decimate 1.0` | full-res, 가장 정확 | 매우 느림 (3 cam 직렬 시 ~4.5 fps) |
| `--no-show-cam-windows` (권장) | per-cam imshow skip, FUSED panel만 | 카메라별 raw 화면 안 보임 |
| `--width 640 --height 480` | 입력 해상도 ↓로 추가 가속 | 먼 거리 작은 태그 검출률 ↓ |

매 `--print-every` 프레임마다 `[timing avg/30f] grab=… detect=… fuse=… gui=… csv=… TOTAL=… (X fps)` 한 줄이 자동으로 찍힘 — 어느 단계가 병목인지 확인 가능. parallel 모드에서 `detect` 시간이 `serial / n_cams` 근처면 잘 작동 중.

> **벤치마크 비교 절차**: 같은 환경에서 `--no-detect-parallel`로 한 번, parallel(default)로 한 번 돌려 timing 출력의 `detect=…` 값 비교 (parallel이 ~2~3배 빠른 게 정상).

### 자주 쓰는 옵션 (멀티카메라)

- `--anchor-ids "10,11"` : 보조 anchor 추가 (`config/floor_anchor_transforms.json` 에 등록되어 있어야 함).
- `--anchor-config <path>` : anchor JSON 경로 변경.
- `--margin-min 25~40` : 너무 작으면 노이즈 증가, 너무 크면 fusion 후보가 줄어듦. **권장 30** (pelvis tag가 비스듬할 때 mean이 35-39로 떨어지므로). 기본 40.
- `--no-origin-hold` : hold 끄기.
- `--origin-hold-max-frames 30` : hold 유지 가능 최대 프레임 (기본 30 = 60fps에서 0.5초).
- `--torso-source head | pelvis | fused`
- `--show-box-tags` : 박스 태그별 candidate 다 표시.
- `--no-show-axes` : 카메라 이미지에 RGB 축 그리기 끄기.

### Tag-history 정책용: 카메라 → 정책 UDP 송신

`sub8_45_tag_history` 정책은 카메라가 추정한 torso/box pose 를 매 step 마다 정책에 공급해야 함. 트래커에 `--udp-publish` 만 추가하면 매 프레임 ASCII UDP 패킷을 보냄. **PC-only 모드 (권장)** 에선 default `--udp-host 127.0.0.1` 그대로 쓰면 됨:

```bash
# PC-only (g1_ctrl 도 같은 PC에서 실행)
python track_robot_and_box_multicam.py \
  --cam1-serial 935322072654 --cam2-serial 115222071236 --cam3-serial 112322072671 \
  --cam1-calib camera1_935322072654_calibration.npz \
  --cam2-calib camera2_115222071236_calibration.npz \
  --cam3-calib camera3_112322072671_calibration.npz \
  --origin-id 1 --anchor-ids 10 --margin-min 20 \
  --detector-quad-decimate 1.5 \
  --no-show-cam-windows \
  --udp-publish \
  --csv-out outputs/sub8_45_taghist.csv --print-every 30
```

```bash
# Split-machine (g1_ctrl 을 Jetson 에서 실행하는 옵션)
python track_robot_and_box_multicam.py ... \
  --udp-publish --udp-host 192.168.123.164 --udp-port 9999
```

옵션:
- `--udp-publish` : 송신 활성화 (기본 OFF, tag-history 정책 외에는 불필요).
- `--udp-host` : 송신 대상. **기본 `127.0.0.1`** (PC-only). Jetson 모드면 `192.168.123.164`.
- `--udp-port` : 기본 9999. deploy 측 `CAMERA_POSE_PORT` env 와 일치해야 함.

종료 시 콘솔에 `[multicam] UDP packets sent: <N>`. 수신 측 `g1_ctrl` 의 `[cam]` 로그 `recv_count` 와 비슷하면 손실 거의 없음 (loopback 이면 1:1 에 가까움).

**Wire format**: ASCII 한 줄 = `<ts_ns> <torso_v> <tx><ty><tz> <tqw><tqx><tqy><tqz> <box_v> <bx><by><bz> <bqw><bqx><bqy><bqz>\n` (17 fields, ~150 bytes). 좌표는 lab frame 그대로.

> **Lab frame ↔ npz 정합**: deploy 측 `motion_file` 은 반드시 `align_npz_to_lab.py` 로 만든 v2 (정렬된) NPZ 사용. 시각 검증은 `python visualize_aligned_npz_mujoco.py --npz outputs/..._processed_v2.npz` 로 가능.

### 카메라 위치/세팅 가이드

- 카메라 사이 extrinsic은 **이 멀티카메라 트래커에 더 이상 필요 없음**. 모든 카메라가 floor tag (id=1) 또는 보조 anchor(id=10)만 잘 보면 된다.
- floor tag 두 개(예: id=1, id=10)를 바닥에 충분히 떨어뜨려 배치하고, 사람/로봇이 둘 다를 동시에 가리는 일이 없도록 하면 origin 복구가 거의 항상 성공.
- 처음 셋업 시 한 번 모든 카메라가 두 태그를 동시에 잘 보는 자세에서 트래커를 1~2분 돌려두면, 카메라별 자체 `T_origin_anchor` 추정이 갱신되면서 안정화.

---

## 3. 시각화/디버그 헬퍼

### 단일 카메라 origin-frame viewer — `detect_apriltag_with_origin_coords.py`

floor tag(id=1) 기준 좌표 보여주는 뷰어. cam frame 좌표 같이 띄워서 디버깅하기 좋음.

```bash
# cam2 예
python detect_apriltag_with_origin_coords.py \
  --serial 115222071236 \
  --calib camera2_115222071236_calibration.npz \
  --origin-id 1 \
  --width 960 --height 540 --fps 60 \
  --show-camera-coords \
  --show-distance-check
```

cam1: `--serial 935322072654 --calib camera1_935322072654_calibration.npz`
cam3: `--serial 112322072671 --calib camera3_112322072671_calibration.npz`

옵션:
- `--show-camera-coords` : 각 태그의 cam-frame 좌표도 같이 표시.
- `--show-distance-check` : `|Δt_cam|`과 `|t_origin|`이 일치하는지(= 좌표 변환이 수학적으로 정확한지) 검증. 빨강이면 사이즈/검출 노이즈, 초록이면 수학 OK.
- `--debug-print-every 60` : 1초에 한 번 콘솔에 `T_cam_origin`, `R_origin_cam[2,2]`, 모든 태그 위치 덤프.
- `--resizable-window` : 창 크기 조절 허용 (스케일하면 약간 흐려짐).

### 다중 카메라 origin-frame fusion viewer — `detect_apriltag_two_cams_origin_fusion.py`

multicam 트래커와 같은 fusion 알고리즘이지만 robot/box 계산 없이 **모든 태그**의 fused origin-frame pose만 보여주는 뷰어 (캘리브/배치 점검용).

```bash
python detect_apriltag_two_cams_origin_fusion.py \
  --cam1-serial 935322072654 \
  --cam2-serial 115222071236 \
  --cam3-serial 112322072671 \
  --cam1-calib camera1_935322072654_calibration.npz \
  --cam2-calib camera2_115222071236_calibration.npz \
  --cam3-calib camera3_112322072671_calibration.npz \
  --extrinsic camera1_to_camera2_extrinsic.npz \
  --extrinsic-cam3-to-c2 camera3_to_camera2_extrinsic.npz \
  --margin-min 40 \
  --width 960 --height 540 --fps 60 \
  --show-orientation --orientation-format both \
  --show-axes
```

---

## 4. 캘리브레이션 (참고)

카메라 위치 옮긴 후 항상 다시 해야 하는 것들:

### intrinsic (체크보드)

```bash
python capture_checkerboard.py --cam-serial 935322072654 --output cam1_chk_imgs --no-gui
python calibrate_camera.py --image-dir cam1_chk_imgs --output camera1_935322072654_calibration.npz
```

cam2/cam3도 동일 패턴 (`--cam-serial` / 출력 파일명만 변경).

### extrinsic (cam pair)

```bash
# cam1 ↔ cam2
python calibrate_extrinsic_two_cams.py \
  --cam1-serial 935322072654 --cam2-serial 115222071236 \
  --cam1-calib camera1_935322072654_calibration.npz \
  --cam2-calib camera2_115222071236_calibration.npz \
  --output camera1_to_camera2_extrinsic.npz

# cam3 ↔ cam2
python calibrate_extrinsic_two_cams.py \
  --cam1-serial 112322072671 --cam2-serial 115222071236 \
  --cam1-calib camera3_112322072671_calibration.npz \
  --cam2-calib camera2_115222071236_calibration.npz \
  --output camera3_to_camera2_extrinsic.npz
```

`--cam1-*` 위치에 cam3을 넣고 `--cam2-*`를 cam2로 지정해서 **결과 키 이름이 `T_c2_c1`** 으로 통일되게 한 점 주의.

### 바닥 보조 anchor (tag 10) 캘리브 — multicam tracker가 fallback에 사용

multicam tracker(`track_robot_and_box_multicam.py`)는 카메라가 tag 1을 못 보면 tag 10 (또는 다른 보조 anchor) 으로 origin을 복원합니다. 이때 `T_origin_anchor` (tag 1 frame에서 본 tag 10의 pose) 가 `config/floor_anchor_transforms.json`에 있어야 함.

태그 1과 10을 둘 다 동시에 잘 보는 카메라 한 대로 한 번 측정:

```bash
python calibrate_floor_anchor_transform.py \
  --serial 115222071236 \
  --calib camera2_115222071236_calibration.npz \
  --origin-id 1 \
  --anchor-id 10 \
  --num-samples 200 \
  --out-config config/floor_anchor_transforms.json
```

`--num-samples` 만큼 두 태그 모두 검출된 프레임을 모아서 평균 (회전은 SVD-projected 평균). 결과는 JSON에 누적 (다른 anchor id 추가 가능). multicam tracker는 startup 시 이 JSON을 로드하고 런타임에 카메라별로 재추정해서 last-seen으로 갱신함.

> **주의**: tag 1, 10 위치가 바뀌면 다시 측정해야 함. (둘 다 바닥에 고정되어 있어야 의미 있음.)

### 박스 / 머리 캘리브 (한 번 박아두면 재사용)

```bash
# 박스 위 태그들 등록 (id=0이 보이는 상태에서 실행 권장)
python register_box_tag_map.py \
  --cam-serial 115222071236 \
  --cam-calib camera2_115222071236_calibration.npz

# 머리 태그 → torso_link 변환 캘리브
python calibrate_head_tag.py \
  --cam-serial 112322072671 \
  --cam-calib camera3_112322072671_calibration.npz
```

---

## 5. shadow 로깅 → 정렬 워크플로우

```bash
# 1) multicam shadow log 수집 (로봇이 sub8_45 모션을 replay하는 동안)
#    --csv-out 에 _YYYYMMDD_HHMMSS 가 자동 삽입되어 overwrite 방지됨
#    e.g. outputs/shadow_3cam_20260523_150500.csv
python track_robot_and_box_multicam.py \
  --cam1-serial 935322072654 --cam2-serial 115222071236 --cam3-serial 112322072671 \
  --cam1-calib camera1_935322072654_calibration.npz \
  --cam2-calib camera2_115222071236_calibration.npz \
  --cam3-calib camera3_112322072671_calibration.npz \
  --origin-id 1 --anchor-ids 10 \
  --margin-min 30 \
  --csv-out outputs/shadow_3cam.csv

# 2-A) 첫 N 프레임 평균 vs sub8_45 npz 첫 프레임 reference 비교, yaw 정합 추정
#      (단일 4-DoF 변환만 추정, npz 자체는 안 건드림. deploy 코드가 매 프레임 적용)
python compute_ref_alignment_yaw_only.py \
  --obs-csv outputs/shadow_3cam_<timestamp>.csv \
  --ref-npz humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz \
  --ref-start-frame 0 --num-frames 60 --yaw-gate-deg 20

# 2-B) ★ 권장 ★ npz 자체를 lab 좌표계로 통째로 변환 (rigid transform 1회 적용 후 새 npz로 저장)
#      shadow CSV의 첫 N 프레임 torso pose를 npz frame 0의 torso pose에 1:1 정합.
#      원본 npz는 절대 안 건드림. 결과 .alignment.json 사이드카에 변환 행렬 + 진단치 저장.
python align_npz_to_lab.py \
  --obs-csv outputs/shadow_3cam_<timestamp>.csv \
  --ref-npz humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz \
  --out-npz outputs/sub8_45_coords_processed_v1.npz \
  --num-frames 30
# 옵션:
#   --ref-frame 0          npz의 어느 프레임을 csv 초기 자세에 맞출지 (default 0)
#   --anchor-body-idx 16   torso_link의 body axis index (G1 default 16)
#   --force                기존 출력 파일 덮어쓰기 허용
```

`align_npz_to_lab.py` 핵심:
- 변환은 **단일 rigid 4×4** (`T_lab_world = T_lab_torso(csv) @ inv(T_world_torso(npz frame F))`).
- z-부호 뒤집기 (npz +Z up → lab +Z down) + yaw 회전 + xyz 평행이동 모두 한 번에 표현.
- 적용 대상: `body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w / object_pos_w / object_quat_w / object_lin_vel_w / object_ang_vel_w`. (joint_pos, joint_vel, contact_mask, fps 등은 그대로)
- **상대 구성 보존**: npz의 torso↔box, torso↔손/발 등 모든 상대 pose는 변환 후에도 numerically 동일. 그래서 정책 obs (`motion_anchor_*`, `object_*_torso`)도 변환 전후 동일.
- `frame 0 torso 잔차 = 0` (정의상). `frame 0 obj 잔차`는 진단치이며, 큰 값은 `head_tag → torso_link` 캘리브 부재 또는 박스 물리 배치 차이를 의미함.

> **`head_tag → torso_link` 캘리브가 없는 경우**: CSV의 "torso" 회전은 head_tag 규약 (x=오른쪽, y=아래, z=앞) 그대로라서 mujoco torso_link 규약과 frame 차이가 있음. 그래도 정책의 **상대** obs는 보존되어 deploy에 영향 없음. 절대 정합도 맞추고 싶으면 `python calibrate_head_tag.py` 먼저 돌리고 새 CSV로 재변환.

shadow 로그는 다음을 보장하면 안전:
- `cam{1,2,3}_origin_source ∈ {"direct","anchor:10"}` 인 프레임이 충분히 많음 ("hold","none" 비율 낮음)
- `n_total_tags ≥ 4` (head + pelvis + box 일부)
- `n_anchor_cams + n_held_cams` 비중이 높으면 → 카메라가 원점/anchor 둘 다 잘 못 보는 시간이 길다는 뜻이라 배치 점검.

### CSV 추가 컬럼 (actor obs와 직접 비교용)

`track_robot_and_box_multicam.py`는 위의 raw pose 외에 **actor 정책이 deploy 시 보는 형태로 가공된 컬럼**도 함께 기록합니다 (lab/origin frame 기준, `T_ref_lab` 적용 전):

| 컬럼 | 의미 |
|---|---|
| `torso_rot6d_0..5` | torso 회전을 6D representation으로 (Zhou 2019, R[:,0]∥R[:,1]) |
| `obj_rot6d_0..5` | object 회전 6D |
| `obj_in_torso_pos_x/y/z` | **`object_pos_torso`** — box 위치를 torso frame으로 변환 |
| `obj_in_torso_rot6d_0..5` | **`object_ori6_torso`** — box 회전을 torso frame에서 6D |
| `torso_yaw_rad`, `obj_yaw_rad` | 빠른 yaw 정렬 디버그 |

활용:
- npz의 `object_pos_torso` / `object_ori6_torso` reference와 직접 비교하여 카메라 추정값 RMSE 확인.
- `compute_ref_alignment_yaw_only.py`는 raw torso/obj pose 컬럼을 그대로 사용 (변경 없음). 6D 컬럼은 분석/플롯/추후 obs 빌더용.

---

## 6. 빠른 트러블슈팅

| 증상 | 원인/조치 |
|---|---|
| `cam fr Δz`가 `origin Δz`와 매우 다른데 거리(`|Δ|`)는 같음 | 정상 (회전 효과). 거리 일치하면 수학 OK. |
| `cam fr Δz`와 `origin Δz`가 양쪽 다 이상 | 거의 항상 `config/tag_sizes.json`의 해당 태그 사이즈 mismatch. 자로 다시 측정. |
| `qt.qpa.xcb: could not connect to display` | SSH/headless 환경. `--no-gui`(있는 스크립트만) 또는 ssh `-X` / X11 forwarding. |
| GUI에 head/pelvis 태그는 OK인데 박스가 안 보임 | `box_tag_map.npz`에 그 id가 등록 안 된 경우 (id=1 같은 floor tag는 의도적으로 빠짐). |
| origin tag가 보이는데도 fused 패널에 모든 태그가 fallback로 잡힘 | 그 카메라의 `origin tag margin < margin_min`. `--margin-min` 낮춰보거나 조명 / 태그 사이즈 점검. |
