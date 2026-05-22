# apriltag2obs — AprilTag 측정값을 mimic obs로 연결하는 단계별 튜토리얼

이 문서는 **`sub8_45_tag/model_38500_all_6_tag.onnx`** (AprilTag 기반 obs를 사용하는 mimic policy)를
실기에 안전하게 올리기 위한 가이드다.

핵심 전략은 **2단계 접근**이다:

1) Phase A — **Shadow Mode**  
   현재 동작 검증된 모션-only 폴리시(`sub8_45`)를 그대로 돌리면서, AprilTag로 만든 obs는
   **폴리시에 주입하지 않고** 콘솔/CSV로만 출력한다. 좌표계, 스케일, 지연이 맞는지 비교만 한다.

2) Phase B — **Tag Policy 활성화**  
   shadow에서 ref/real 일치가 충분히 검증되면, 같은 obs를 C++ deploy에 정식 obs term으로 등록하고
   `sub8_45_tag/params/deploy.yaml`을 만들어 태그 폴리시를 실제로 구동한다.

---

## 0) 전제

- 두 ONNX는 **같은 `npz`** 사용 (`sub8_largebox_045_original_extended.npz`)
- 현재 npz 키 (확인 완료):
  ```
  fps, joint_pos, joint_vel,
  body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w,
  object_pos_w, object_quat_w, object_lin_vel_w, object_ang_vel_w,
  contact_mask
  ```
- `sub8_45` (모션-only) ONNX 입력 obs 차원: 학습 측 yaml과 같은 `deploy.yaml`로 동작 중
- `sub8_45_tag` ONNX 입력: `obs[1, 178]`, `time_step[1, 1]`  
  → `sub8_45` 대비 **추가 obs 차원**이 들어 있음 (object/torso anchor 계열로 추정)

---

## 1) 로봇 위 AprilTag 배치 (확정 사실)

| Tag ID | 위치 | 의미 | 고정 오프셋 |
|---|---|---|---|
| `9` | 머리 위 | 머리 태그 | tag → torso_link: 머리 기준 **z 방향 -25 cm** |
| `8`, `7` | pelvis 살짝 아래 | pelvis 태그 | tag → root(pelvis): **z 방향 +10 cm** |
| `0~5` | 박스 위 | 박스 태그 (이미 사용중) | `box_tag_map.npz` 기준 |
| `1` | 바닥 | 실험실 origin (옵션) | floor anchor |

추가 사실 (사용자 제공):
- pelvis(root) → torso_link는 **z 방향 +20 cm** 정도
- 따라서 torso position은 두 경로로 추정 가능:
  - **경로 A (pelvis tag)**: `T_world_root = T_world_pelvisTag * (z+0.10)`,  
    `T_world_torso ≈ T_world_root * (z+0.20, ori는 root quat 기준)`
  - **경로 B (head tag)**: `T_world_torso = T_world_headTag * (z-0.25)`

> **torso orientation 자체는 새로 만들지 말고**, 기존 `motion_anchor_ori_b` 코드를 그대로 사용한다  
> (IMU root quat + waist 3축 모터각). 즉 외부 비전은 **위치 위주**로만 우선 활용.

---

## 2) 좌표계 정리 (실수 방지용)

이름 그대로 코드/로그에 박아 쓰자.

- World/Camera2 frame (실험실 카메라 기준 world):
  - `T_world_torso_real`  ← AprilTag 기반 torso 추정 (경로 A 또는 B 또는 두 개 평균)
  - `T_world_object_real` ← box 태그 기반 (`track_robot_and_box.py`, `estimate_box_pose_two_cams_top_tags.py`)
- Reference frame (npz):
  - `T_world_torso_ref`   ← `body_pos_w[t, torso_idx]`, `body_quat_w[t, torso_idx]`  
    (현재 합의된 `torso_idx = 16`, frame 0 값은 `apriltag_setup.md` §B-4 참고)
  - `T_world_object_ref`  ← `object_pos_w[t]`, `object_quat_w[t]`
- 실험실/ref 정합:
  - `T_realWorld_refWorld`  ← yaw-only 정합 (이미 있는 `compute_ref_alignment_yaw_only.py`)

Body frame (`*_b`) 변환 (정책이 기대하는 표현):
- `T_torso_object = inv(T_world_torso) @ T_world_object`
- `motion_anchor_pos_b = R_world_torso^T (p_world_torsoRef - p_world_torsoReal)`  
  (학습측 정의에 맞춰서 부호/순서는 obs term 코드에서 최종 확정)
- `motion_anchor_ori_b` (6D): 이미 `State_Mimic.cpp`에 구현된 식 그대로 사용

---

## 3) Phase A — Shadow Mode 단계별

### 3-1. 환경 준비

```bash
cd /home/roy/realsense_calib
conda activate unitree_rl_mjlab

# 카메라 인식 점검
rs-enumerate-devices
```

### 3-2. AprilTag 기반 torso/object pose 실시간 측정

GUI에서 보여야 하는 태그:
- 머리 태그(9), pelvis 태그(8/7), 박스 태그(0~5)

이미 동작 검증된 두 가지 옵션:

(a) **단일 카메라(예: cam2)** 빠른 점검
```bash
python detect_apriltag_with_origin_coords.py \
  --serial 115222071236 \
  --calib camera2_115222071236_calibration.npz \
  --width 960 --height 540 --fps 60 \
  --resizable-window
```

(b) **3카메라 가중 융합** (추천: 우리가 방금 확장한 스크립트)
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
  --show-axes --axis-length 0.08 \
  --width 960 --height 540 --fps 60
```

GUI에서 다음 항목들이 보여야 한다:
- 머리 태그(9), pelvis 태그(8/7), 박스 태그(0~5)
- 각 태그에 RGB 축
- origin id 기준 상대좌표

### 3-3. shadow 로깅 스크립트 (권장: 별도 스크립트 분리)

`track_robot_and_box.py` 또는 `estimate_box_pose_two_cams_top_tags.py`를 이용해 다음을 CSV로 저장:

필요한 컬럼 (정합 스크립트와 호환):
```
frame_idx, t_sec,
root_pos_x, root_pos_y, root_pos_z,
root_quat_w, root_quat_x, root_quat_y, root_quat_z,
torso_pos_x, torso_pos_y, torso_pos_z,
torso_quat_w, torso_quat_x, torso_quat_y, torso_quat_z,
obj_pos_x, obj_pos_y, obj_pos_z,
obj_quat_w, obj_quat_x, obj_quat_y, obj_quat_z,
torso_path  # "head_tag" or "pelvis_tag" or "fused"
```

여기서:
- `root_pos = pelvisTag_world + R_pelvisTag_world @ [0,0,0.10]` (오프셋 부호는 부착 방향에 맞춰 검증)
- `torso_pos`:
  - `head_tag`: `headTag_world + R_headTag_world @ [0,0,-0.25]`
  - `pelvis_tag`: `root_pos + R_root @ [0,0,0.20]`
  - `fused`: 두 값의 가중평균 (margin/가시성 기반)
- `torso_quat`: 외부 비전이 아니라 **로봇 IMU+모터**로 만든 값을  
  shadow에서는 같이 기록만 (실제 mimic obs 계산 코드와 동일하게 만들기 위해)

> 권장: `track_robot_and_box.py`에 `--csv-out` 옵션과 `tag-id` 매핑(7=head, 8/9=pelvis) 추가하여
> 위 컬럼을 그대로 떨어뜨리도록 확장. (파일명 예: `outputs/init_obs_torso_obj.csv`)

### 3-4. ref/real 정합 (yaw-only)

이미 만들어둔 스크립트를 그대로 사용:

```bash
python compute_ref_alignment_yaw_only.py \
  --obs-csv outputs/init_obs_torso_obj.csv \
  --ref-npz unitree_rl_mjlab/deploy/ref_npz/sub8_largebox_045_original_extended.npz \
  --num-frames 60 \
  --out-json config/T_ref_lab_sub8_45.json
```

산출물 `config/T_ref_lab_sub8_45.json`:
- `T_ref_lab`, `R_ref_lab`, `t_ref_lab`
- `diagnostics.torso_rmse_m`, `obj_rmse_m`

목표:
- torso RMSE: 수 cm 이내
- obj RMSE: 수 cm 이내 (정지 박스 기준)
- yaw_deg 안정 (짧은 측정 동안 흔들림 ≤ 1~2°)

### 3-5. 모션-only 폴리시 (`sub8_45`)와 동시 실행

핵심: **태그 obs는 절대 폴리시에 넣지 않는다.**

터미널 1 (실기 deploy, 모션-only 폴리시):
```bash
cd /home/roy/realsense_calib/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl  # 기존 절차대로 sub8_45 (no tag) 실행
```

터미널 2 (외부 측정 + shadow CSV):
```bash
cd /home/roy/realsense_calib
python detect_apriltag_two_cams_origin_fusion.py \
  ... (위 옵션) \
  --csv-out outputs/shadow_obs_sub8_45.csv  # ← 이 옵션은 추가 필요시 patch
```

> 주의: 두 터미널의 **시각(time)** 정렬 위해 시작 시각·deploy step 인덱스를 같이 기록.

### 3-6. shadow 비교 (offline)

```bash
python - <<'PY'
import numpy as np, pandas as pd, json
ref = np.load("unitree_rl_mjlab/deploy/ref_npz/sub8_largebox_045_original_extended.npz")
obs = pd.read_csv("outputs/shadow_obs_sub8_45.csv")
T = json.load(open("config/T_ref_lab_sub8_45.json"))

R = np.array(T["R_ref_lab"]); t = np.array(T["t_ref_lab"])
def to_ref(p):
    p = np.asarray(p, float)
    return p @ R.T + t

# 시작 60 프레임 비교: torso_pos, obj_pos
N = 60
ref_torso = ref["body_pos_w"][:N, 16, :]
ref_obj   = ref["object_pos_w"][:N]
real_torso = to_ref(obs[["torso_pos_x","torso_pos_y","torso_pos_z"]].values[:N])
real_obj   = to_ref(obs[["obj_pos_x","obj_pos_y","obj_pos_z"]].values[:N])

print("torso RMSE [m]:", np.sqrt(np.mean(np.sum((real_torso-ref_torso)**2, 1))))
print("obj   RMSE [m]:", np.sqrt(np.mean(np.sum((real_obj-ref_obj)**2, 1))))
PY
```

게이트 (Phase A 통과 기준 예시):
- torso RMSE ≤ 5cm, obj RMSE ≤ 5cm
- 정지 시 obj jitter p95 ≤ 1cm
- tag 가시성 dropout ≤ 5% (대상 구간)

---

## 4) Phase B — Tag Policy 정식 활성화

### 4-1. ONNX 입력 명세 확정 (가장 먼저)

178 dim의 obs 구성을 **학습측 yaml/cfg**에서 그대로 가져와야 한다.  
가설로 27 dim의 추가가 다음과 같은 구성일 수 있다 (확정은 학습측 코드로):

| 후보 obs term | dim |
|---|---|
| motion_anchor_pos_b | 3 |
| motion_anchor_ori_b | 6 |
| object_pos_b (current) | 3 |
| object_ori_b (current, 6D) | 6 |
| object_lin_vel_b | 3 |
| object_ang_vel_b | 3 |
| (object ref pos_b 등 가능) | 3 |

> **중요**: 가설로 deploy.yaml을 만들지 말고, 학습측의 `tasks/.../mdp/observations.py`,
> `env_cfg.py`에서 obs 순서/차원을 1:1로 맞춰 와야 한다.

### 4-2. `sub8_45_tag/params/deploy.yaml` 뼈대

`sub8_45/params/deploy.yaml` + `dance1_subject2/params/deploy.yaml`의
`motion_anchor_ori_b`를 합친 뼈대를 그대로 복사 후 obs 항목만 정확히 채운다:

```yaml
joint_ids_map: [0,1,2,...,28]
step_dt: 0.02
stiffness: [...]   # sub8_45와 동일
damping:   [...]   # sub8_45와 동일
default_joint_pos: [...]
commands: {}
actions:
  JointPositionAction:
    clip: null
    joint_names: [.*]
    scale:  [...]
    offset: [...]
    joint_ids: null

observations:
  motion_command:
    params: {command_name: motion}
    clip: null
    scale: [1.0, ... (58)]
    history_length: 1

  motion_anchor_pos_b:
    params: {command_name: motion}
    clip: null
    scale: [1.0, 1.0, 1.0]
    history_length: 1

  motion_anchor_ori_b:
    params: {command_name: motion}
    clip: null
    scale: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    history_length: 1

  # === tag-driven obs (학습측 yaml과 1:1 맞춰서 채울 것) ===
  object_pos_b:
    params: {}
    clip: null
    scale: [1.0, 1.0, 1.0]
    history_length: 1

  object_ori_b:
    params: {}
    clip: null
    scale: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    history_length: 1

  object_lin_vel_b:
    params: {}
    clip: null
    scale: [1.0, 1.0, 1.0]
    history_length: 1

  object_ang_vel_b:
    params: {}
    clip: null
    scale: [1.0, 1.0, 1.0]
    history_length: 1

  # === proprio (sub8_45와 동일 구성/순서) ===
  base_ang_vel: { params: {}, clip: null, scale: [1.0,1.0,1.0], history_length: 1 }
  projected_gravity: { params: {}, clip: null, scale: [1.0,1.0,1.0], history_length: 1 }
  joint_pos_rel: { params: {}, clip: null, scale: [1.0,...(29)], history_length: 1 }
  joint_vel_rel: { params: {}, clip: null, scale: [1.0,...(29)], history_length: 1 }
  last_action:   { params: {}, clip: null, scale: [1.0,...(29)], history_length: 1 }
```

> 합산 dim이 **정확히 178**이 되도록 검증.  
> 학습측 cfg와 **순서**도 동일해야 함.

### 4-3. C++ obs term 등록 위치

새 obs term을 다음 두 곳 중 한 곳에 추가하면 자동 등록된다:

- `unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/observations/observations.h`  
  (proprio/공통 obs)
- `unitree_rl_mjlab/deploy/robots/g1/src/State_Mimic.cpp`  
  (mimic 전용 obs, 예: `motion_anchor_ori_b`)

추가할 term 예 (시그니처):

```cpp
REGISTER_OBSERVATION(motion_anchor_pos_b)
{
    auto loader = State_Mimic::motion;
    // ref pos: world frame from npz (loader->root_position()/torso link)
    // real pos: AprilTag로 추정한 torso pos를 외부 입력으로 받아옴
    // 둘의 차이를 현재 torso 회전으로 body frame에 투영 → 3 dim
    ...
}

REGISTER_OBSERVATION(object_pos_b)
{
    // T_torso_object = inv(T_world_torso_real) @ T_world_object_real
    ...
}

REGISTER_OBSERVATION(object_ori_b)
{
    // 6D rep of R_torso_object
    ...
}

REGISTER_OBSERVATION(object_lin_vel_b)
{
    // (선택) finite-diff 또는 외부에서 주입
    ...
}

REGISTER_OBSERVATION(object_ang_vel_b)
{
    // 동일
    ...
}
```

### 4-4. 외부 비전 → C++ IPC

C++가 AprilTag 측정값을 받을 수 있어야 한다. 옵션:

- 가장 단순: **UDP** (`localhost:PORT`)에 fixed-format 패킷 전송 (Python 송신, C++ 수신 thread)
- 좀 더 안전: **shared memory** (`boost::interprocess` 또는 `mmap`) + sequence id
- ROS 환경 있으면 그쪽 토픽 활용

저장 위치 제안:
- `ArticulationData`에 vision 필드 추가 (`p_world_torso_vision`, `R_world_torso_vision`,
  `p_world_object_vision`, `R_world_object_vision`, `vision_seq`, `vision_t`)
- subscriber thread는 `unitree_articulation.h`나 별도 `vision_input.h`에서 갱신
- 매 step에서 obs term이 이 값을 읽음 (없을 땐 hold + EMA 또는 timeout)

### 4-5. 안전 dry-run

다음 순서로 점진 활성화:

1) **policy off**, vision IPC 채널만 켜기 → C++ 콘솔에 vision 값이 매 step 잘 들어오는지 확인  
2) **policy on, motor publish off (`--dry-run`)** → action은 계산하되 모터 차단  
3) sim vs real obs 분포 비교 (`base_ang_vel`, `joint_*`, `motion_anchor_*`, `object_*`)  
4) FixStand에서 짧게만 활성화 → safety state 정상 복귀 확인  
5) 정상 deploy

---

## 5) 체크리스트

- [ ] 카메라 3대 모두 USB3.x로 인식
- [ ] cam1/cam2/cam3 각각 intrinsic 새로 계산
- [ ] `cam1↔cam2`, `cam3↔cam2` extrinsic 재계산 (mount 변경 시 필수)
- [ ] 머리 태그 9, pelvis 태그 8/7 부착 완료
- [ ] tag → torso/root z 오프셋 부호 실측 검증
- [ ] shadow CSV에 root/torso/obj 컬럼 모두 들어옴
- [ ] yaw align: torso RMSE ≤ 5cm, obj RMSE ≤ 5cm
- [ ] sub8_45 (모션-only) 폴리시 정상 동작 + shadow 동시에 동작
- [ ] sub8_45_tag onnx 입력 178 dim 구성을 학습측 cfg로 확정
- [ ] sub8_45_tag/params/deploy.yaml 작성 (dim 합 = 178)
- [ ] C++ obs term 5개 (`motion_anchor_pos_b`, `object_pos_b`, `object_ori_b`, `object_lin_vel_b`, `object_ang_vel_b`) 추가
- [ ] vision IPC 채널 (UDP/SHM) 동작 확인
- [ ] dry-run (motor off) 통과
- [ ] FixStand 안전 복귀 통과

---

## 6) 자주 헷갈리는 포인트

- **AprilTag pose는 카메라 좌표계 기준**.  
  obs로 쓰려면 cam→world(C2)→ref world→torso body frame까지 변환 체인이 정확해야 한다.
- **`motion_anchor_*`는 ref와 real의 차이를 만든 값**이지, 외부 센서로 ref를 추정하는 게 아니다.  
  ref는 항상 npz에서 읽고, real은 IMU+모터(또는 vision)로 만든다.
- 회전 표현은 학습 코드와 동일해야 한다.  
  현재 deploy의 `motion_anchor_ori_b`는 6D (회전행렬 일부 6원소). object 쪽도 같은 표현일 가능성 큼.
- AprilTag pose는 60Hz, policy는 50Hz. **timestamp 보간/홀드** 필요.
- mount/조명 바뀌면 extrinsic은 즉시 신뢰도 떨어짐 → 재캘리브.

---

## 7) 다음으로 내가 도와줄 수 있는 것

(요청하면 바로 수정/생성)

- `track_robot_and_box.py`에 `--csv-out`, head/pelvis 태그 매핑, fused torso 옵션 추가
- `sub8_45_tag/params/deploy.yaml` 뼈대 파일 생성
- `observations.h` / `State_Mimic.cpp`에 `motion_anchor_pos_b`, `object_pos_b`, `object_ori_b` 등 stub 추가
- Python → C++ UDP 브리지 최소 코드
- shadow 비교 자동 분석 스크립트(`outputs/shadow_*.csv` → ref RMSE/잠김 통계)
