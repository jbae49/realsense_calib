# Sim2Real 파이프라인 (정리본): G1 object-carrying + velocity 백업 + mimic ("오브젝트 없이" 모션 트래킹만)

이 문서는 `sim2real.md`의 내용을 **실행 순서**로 재구성하고, **mjlab**과 **unitree_rl_mjlab** 두 리포를 기준으로 **빠진 조각·주의점**을 보충한 것이다.

---

## 0. 목표를 한 문장으로

**OmniRetarget/mjlab에서 학습한 G1 object-carrying imitation policy를 실제 G1에 안전하게 올리기 위해**, 천장 RealSense + AprilTag로 `T_world_torso`, `T_world_object`, `T_torso_object`를 추정·정합하고, **velocity tracking을 기본(백업) 정책**으로 두고 **mimic policy로 전환**하는 루프까지 연결하는 sim2real 파이프라인을 만든다.

---

## 1. 두 리포의 역할 (반드시 구분)

| 구분 | **mjlab** (`mujocolab/mjlab`) | **unitree_rl_mjlab** |
|------|-------------------------------|----------------------|
| 역할 | 학습·평가 프레임워크, 문서·벤치마크 | mjlab 위에 Unitree 로봇·태스크를 얹고 **실제 배포까지** 포함 |
| 학습 CLI 예 | `uv run train Mjlab-Velocity-Flat-Unitree-G1 ...` | `python scripts/train.py Unitree-G1-Flat ...` |
| Mimic 예 | `uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name ...` | `python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=...` |
| 실기 배포 | **없음** (C++ `g1_ctrl`, DDS 없음) | `deploy/robots/g1/` — ONNX Runtime + `g1_ctrl` + FSM |
| Sim2Sim (권장 경로) | 문서상 별도 예제 중심 | **통합**: `simulate/` (unitree_mujoco) + `deploy/robots/g1/build/g1_ctrl --network=lo` |
| ONNX | upstream 정책에 따라 다름 | **학습 저장 시** `policy.onnx` + `policy.onnx.data` + ONNX 메타데이터 (`src/tasks/*/rl/runner.py`에서 `attach_metadata_to_onnx`) |

**결론:** velocity를 **mjlab만**으로 학습했다면, **배포·관측·액션 스택은 unitree_rl_mjlab 기준으로 다시 검증**해야 한다. 가능하면 **학습·export·deploy.yaml 생성 모두 같은 리포(unitree_rl_mjlab)에서** 맞추는 것이 가장 안전하다.

### 1.1 이 워크스페이스의 리포 3개 (경로 고정)

| 변수 | 경로 | 쓰는 일 |
|------|------|---------|
| `REALSENSE_CALIB` | `/home/roy/realsense_calib` | AprilTag·RealSense 캘리브·추적 스크립트 |
| `HP` | `/home/roy/realsense_calib/humanoid_project` | OmniRetarget npz, `replay`, **manipulation** 학습/play (fork) |
| `URML` | `/home/roy/unitree_rl_mjlab` | **velocity / mimic** 학습, **ONNX, `g1_ctrl`, sim2sim** (공식 upstream) |

- **박스 carrying(manipulation)** → `HP`에서만 (upstream `URML`에는 `manipulation` 태스크 없음).
- **실기 deploy·FSM·unitree_mujoco** → 반드시 **`URML`** (`/home/roy/unitree_rl_mjlab`).
- `HP`에도 `deploy/`가 있지만, 지금 받은 공식 클론 기준으로 **Step 6~9는 `URML`만** 쓴다.

---

## 2. 전체 데이터/제어 흐름 (한 장)

```
[OmniRetarget / reference .npz]
        ↓
[object frame / tag transform 검증]
        ↓
[mjlab 계열 학습: velocity (백업) + tracking/mimic]
        ↓
[play로 sim 검증 → ONNX는 unitree_rl_mjlab 저장 시 자동]
        ↓
[deploy.yaml / policy.onnx를 g1 FSM에 배치]
        ↓
[unitree_mujoco + g1_ctrl --network=lo  → sim2sim]
        ↓
[shadow: lowstate + vision + inference, lowcmd 금지]
        ↓
[실기: FixStand → Velocity → Mimic (게이트 전환)]
        ↓
[로깅 + safety monitor]
```

**AprilTag/vision**은 원문대로 policy 입력에 raw로 넣지 말고, 필터·valid 플래그·안전 모드와 함께 설계한다.  
**중요:** 현재 `unitree_rl_mjlab`의 C++ mimic 경로(`State_Mimic.cpp`)는 **모션 파일 기반 관측**(joint ref, anchor ori 등)이며, **AprilTag object pose를 관측 벡터에 넣는 코드는 포함되어 있지 않다.** object-carrying을 **학습 시점과 동일한 observation**으로 실기에 올리려면 deploy 쪽 **관측 등록·deploy.yaml·학습 MDP**를 한 세트로 확장해야 한다.

---

## 2.1 Locomotion(velocity) policy — 시뮬에서 eval·inference / “올려보기”

locomotion을 **어디서 학습했는지**에 따라 두 단계를 구분하는 게 좋다. “시뮬에서 테스트”와 “배포 스택에서 인퍼런스”가 서로 다르다.

### 층 1 — 학습 프레임워크 안에서 eval (PyTorch `.pt`, 학습과 동일한 sim·관측)

**mjlab만 쓴 경우** (`~/mjlab` 등, upstream README 기준):

```bash
cd /path/to/mjlab
# 로컬 체크포인트
uv run play Mjlab-Velocity-Flat-Unitree-G1 --checkpoint-file /path/to/model_XXXX.pt
# 또는 W&B에서 체크포인트 가져오기
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
# (선택) W&B run 안의 특정 파일명
# uv run play ... --wandb-run-path ... --wandb-checkpoint-name model_4000.pt
```

MDP만 보고 싶을 때(정책 없이 환경 동작 확인):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent zero
uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent random
```

**unitree_rl_mjlab에서 학습한 경우** (이후 배포 경로와 태스크를 맞추기 쉬움):

```bash
cd /home/roy/unitree_rl_mjlab
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/<날짜>/model_xx.pt
```

이 층은 **뷰어에서 보행·속도 추종 품질**을 보는 **정책 eval**이고, 인퍼런스는 **Python + 학습 시점 래퍼**다. 실기 `g1_ctrl`와 완전히 동일하다고 가정하면 안 된다.

### 층 2 — 배포와 동일한 inference (ONNX + `g1_ctrl` + unitree_mujoco = sim2sim)

실기에 “올리기” 직전에 할 **인퍼런스 테스트**는 이 리포 README **4.5.1**과 같다.

1. **ONNX:** 학습 시 저장된 `policy.onnx`(+`.data`)를  
   `/home/roy/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/`  
   에 두고, 그 학습과 맞는  
   `.../velocity/v0/params/deploy.yaml`  
   (`step_dt`, `joint_ids_map`, stiffness/damping, observation 이름, action scale 등)을 유지한다.
2. **빌드:** `cd /home/roy/unitree_rl_mjlab/deploy/robots/g1 && mkdir -p build && cd build && cmake .. && make`.
3. **시뮬:** `/home/roy/unitree_rl_mjlab/simulate/build/unitree_mujoco` (게임패드, `simulate/config`에서 로봇 선택).
4. **컨트롤러:** 다른 터미널에서  
   `/home/roy/unitree_rl_mjlab/deploy/robots/g1/build/g1_ctrl --network=lo`  
   → 시뮬 lowstate에 대해 **실기와 같은 ONNX Runtime 경로**로 lowcmd가 나간다. FSM에서 Velocity 상태로 두고 조이스틱 등으로 eval.

`State_RLBase.cpp`에는 `keyboard_velocity_commands` 등록 예시가 있어, 배포 yaml에서 `velocity_commands` 관측 이름을 키보드용으로 바꾸면 **키로 선속도 명령**을 넣는 sim2sim 디버깅도 가능하다(설정 변경 후 재빌드).

### mjlab에서만 학습하고 unitree로 올리고 싶을 때

**층 1**은 mjlab `play`로 가능하지만, **층 2**는 ONNX·`deploy.yaml`·관측 순서가 `g1_ctrl`과 맞아야 한다. **가능하면 `unitree_rl_mjlab`에서 동일 역할 task로 짧게 재학습·재export**하거나, 최소한 동일 env 정의로 맞춘 ONNX+yaml로 sim2sim을 통과시킨 뒤 실기로 간다.

**✓ 요약 게이트**

- 층 1: 학습 sim에서 보행·낙상·명령 응답이 목표 수준.
- 층 2: `g1_ctrl --network=lo`에서 관절 명령·주파수·이상 동작 없음 → 그다음 `--network=<실제 인터페이스>`.

---

## 3. 단계별 실행 순서 (튜토리얼): 기준 클립 **sub8 · largebox · 045**

**이 섹션만 위에서 아래로 순서대로** 따라가면 된다. 각 Step 끝 **✓ 게이트**를 통과한 뒤 다음 Step으로 간다.

### 3.0 경로·파일 고정 (복붙용)

터미널마다 한 번만 실행:

```bash
export REALSENSE_CALIB=/home/roy/realsense_calib
export HP=/home/roy/realsense_calib/humanoid_project
export URML=/home/roy/unitree_rl_mjlab
export CLIP=sub8_largebox_045_original
export NPZ=$HP/src/assets/OmniRetarget/processed/${CLIP}.npz
```

| 변수 | 실제 경로 |
|------|-----------|
| `REALSENSE_CALIB` | `/home/roy/realsense_calib` |
| `HP` | `/home/roy/realsense_calib/humanoid_project` — manipulation·replay·OmniRetarget |
| `URML` | `/home/roy/unitree_rl_mjlab` — velocity·mimic·**deploy·simulate** |
| `NPZ` | `$HP/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz` |

**리포 역할**

| 할 일 | 디렉터리 |
|-------|----------|
| npz, `replay`, manipulation 학습/play | `$HP` |
| velocity / mimic 학습, ONNX 복사, `g1_ctrl`, sim2sim | `$URML` |
| 카메라·AprilTag | `$REALSENSE_CALIB` |

**기준 npz가 없을 때만** (이미 받아 두었다면 Step 2로):

```bash
cd $HP
python scripts/omniretarget_to_npz.py --clip sub8_largebox_045_original
```

---

### 실행 순서 한 장 (Step 0 → 10)

| Step | Phase | 한 줄 목표 | 어디서 |
|------|-------|------------|--------|
| **0** | 준비 | conda + `pip install -e .` (두 리포) | `$URML`, `$HP` |
| **1** | B | npz 키 확인 + MuJoCo replay | `$HP` |
| **2** | A | RealSense + AprilTag로 `T_world_*` | `$REALSENSE_CALIB` |
| **3** | C | reference ↔ 실험실 rigid align | 수기+로그 |
| **4** | D | velocity·mimic 학습 (`URML`) + manipulation (`HP`) | `$URML` / `$HP` |
| **5** | E | play로 정책 검증 | `$URML` / `$HP` |
| **6** | F | ONNX·deploy.yaml·FSM 패키징 | **`$URML/deploy/...`** |
| **7** | G | sim2sim (`g1_ctrl --network=lo`) | **`$URML`** |
| **8** | H | shadow (lowcmd 없음) | 실기 네트워크 |
| **9** | I | 실기 velocity → mimic | **`$URML/deploy/.../build`** |
| **10** | J | 로깅·메트릭 | 공통 |

### 진행 체크리스트 (2026-05-15 기준)

- [x] **Step 0 준비 완료**: conda `unitree_rl_mjlab`, `pip install -e .`(URML/HP), 비전 패키지 설치
- [x] **실기 전제 패키지 완료**: CycloneDDS + unitree_sdk2 설치
- [x] **빌드 확인 완료**: `$URML/deploy/robots/g1/build`에서 `g1_ctrl` 빌드 성공
- [x] **Step 1 일부 완료**: `sub8_largebox_045_original.npz` 키 확인 + OBB 로그 산출
- [ ] **다음 시작점 (내일 바로 여기서 시작)**: Step 1-2 `replay` 눈검증 완료 후 체크리스트 3개 통과
- [ ] Step 2 (RealSense/AprilTag 런타임 검증)
- [ ] Step 3 (reference ↔ 실험실 align)
- [ ] Step 4~7 (학습/ONNX/FSM/sim2sim)
- [ ] Step 8~10 (shadow/실기/로깅)

**내일 시작 커맨드(복붙):**

```bash
conda activate unitree_rl_mjlab
export REALSENSE_CALIB=/home/roy/realsense_calib
export HP=/home/roy/realsense_calib/humanoid_project
export URML=/home/roy/unitree_rl_mjlab
cd $HP
python scripts/play.py replay sub8_largebox_045_original
```

---

### Step 0 — 환경 설치 (한 번만)

conda env 이름은 `unitree_rl_mjlab` (README·`URML/doc/setup_en.md` 기준).

```bash
conda create -n unitree_rl_mjlab python=3.11 -y
conda activate unitree_rl_mjlab

sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# 공식 리포 (velocity / mimic / deploy)
cd /home/roy/unitree_rl_mjlab
pip install -e .

# fork (manipulation / OmniRetarget replay) — 같은 env에서 설치 가능
cd /home/roy/realsense_calib/humanoid_project
pip install -e .
```

비전 (Step 2):

```bash
conda activate unitree_rl_mjlab
pip install opencv-python pyrealsense2 pupil-apriltags numpy
```

**실기 deploy 전제 패키지** (Step 6~9, README 4절 — 한 번 빌드):

- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

**✓ 게이트**

```bash
conda activate unitree_rl_mjlab
cd /home/roy/unitree_rl_mjlab && python scripts/list_envs.py | head
cd /home/roy/realsense_calib/humanoid_project && python scripts/list_envs.py | grep -i manip
python -c "import mujoco, mjlab; print('mjlab OK')"
test -f "$NPZ" && echo "npz OK: $NPZ"
```

---

### Step 1 — Phase B: Reference + object 뷰어 검증 (정책 없음)

`object_pos_w`가 박스 **어느 점**인지는 숫자만으로 헷갈린다. replay로 **같은 npz + 같은 largebox MJCF**를 눈으로 확인한다.

#### 1-1. npz 키 확인

```bash
conda activate unitree_rl_mjlab
cd /home/roy/realsense_calib/humanoid_project

python - <<'PY'
import numpy as np
p = "/home/roy/realsense_calib/humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz"
d = np.load(p)
print("keys:", d.files)
for k in ("object_pos_w", "object_quat_w", "body_pos_w", "joint_pos", "contact_mask"):
    if k in d.files:
        print(k, d[k].shape, d[k].dtype)
PY
```

기대: `object_pos_w` (T,3), `object_quat_w` (T,4). (`*_retargeted.npz`만 있으면 오브젝트 키가 없을 수 있음 → `omniretarget_to_npz.py` 재실행.)

#### 1-2. MuJoCo replay (로봇 + 박스)

`replay`는 **MuJoCo 창만** 연다 (viser 아님). 아래 둘 중 하나:

```bash
cd /home/roy/realsense_calib/humanoid_project

# 짧은 형태 (권장)
python scripts/play.py replay sub8_largebox_045_original

# 또는 명시적 플래그
python scripts/play.py replay --clip sub8_largebox_045_original

# npz 절대경로
python scripts/play.py replay --clip /home/roy/realsense_calib/humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz
```

성공: `Loaded <T> frames from ...` 후 MuJoCo 창. Esc로 종료.

#### 1-3. 뷰어 체크리스트

1. **Group enable** (왼쪽): group 4(비주얼) / 5(충돌 헐) 번갈아 — 크게 어긋나지 않는지.
2. 손·박스가 **파묻히거나 수 m 떠 있지 않은지** (앞·중·끝 프레임).
3. `contact_mask` True 구간에서 박스가 **붉게** 칠해지는지.
4. 박스 프레임 정의 메모:  
   `/home/roy/realsense_calib/humanoid_project/src/assets/OmniRetarget/models/largebox/largebox.xml`

#### 1-4. OBB 수치 해석 (실행 로그 예시)

`python scripts/build_object_collision_hull.py` 실행 로그 해석:

- **AABB extents:** `0.4712 x 0.4587 x 0.4079 m` (축정렬 박스, 과대 추정 가능)
- **OBB extents:** `0.3351 x 0.3377 x 0.3601 m` (최소부피 oriented box, 기준값)
- **hull/AABB = 0.425:** AABB 부피의 약 42.5%만 실제 hull이 차지 → AABB를 크기 기준으로 쓰면 오차 큼
- **inertia cross-check 정상:** `trimesh` vs MuJoCo 차이가 매우 작음 (`pos diff ~1e-9`, `eig rel ~1e-8`)

해석 요약: **sim2real에서 상자 크기 기준은 AABB가 아니라 OBB extents**를 사용한다.

주의: 해당 실행은 `nominal_mass=1.0`으로 `largebox.xml`을 덮어썼다. 기존 세팅을 유지하려면 실행 시 `--nominal-mass`를 명시하고, 질량/관성 변경 여부를 `largebox.xml`에서 확인한다.

**✓ 게이트 (Step 1)**

- [ ] `object_pos_w` / `object_quat_w` 존재
- [ ] replay 정상 재생
- [ ] “object pose = MJCF free joint 원점” 한 줄로 기록 (Step 2의 `T_object_tag`와 맞출 것)

---

### Step 2 — Phase A: 비전만 (`REALSENSE_CALIB`)

상세 수식·부착 방향: `GUIDE_tag_to_torso.md`.  
여기서는 **이미 있는 스크립트 경로**만 고정한다.

| 순서 | 스크립트 | 하는 일 |
|------|----------|---------|
| (선행) | `capture_checkerboard.py`, `capture_checkerboard_cam2.py` | 체커보드 이미지 수집 |
| | `calibrate_camera.py`, `calibrate_camera_cam2.py` | `camera*_*.npz` |
| | `calibrate_extrinsic_two_cams.py` | `camera1_to_camera2_extrinsic.npz` |
| | `register_box_tag_map.py` | `box_tag_map.npz` |
| | `calibrate_head_tag.py` | `T_tag_torso.npz` |
| 런타임 | `track_robot_and_box.py` | `T_world_*`, `T_torso_object` 실시간 |
| 검증 | `validate_head_to_torso.py` | head tag → torso 변환 검증 |

```bash
cd /home/roy/realsense_calib

# 예: head tag 캘리브 (이미 했다면 생략)
python calibrate_head_tag.py

# 예: 로봇+박스 동시 추적 (스크립트 안 CAM_SERIAL, TAG ID 확인 후)
python track_robot_and_box.py
```

고정할 변환 (코드·로그에 이름 그대로):

- `T_world_object = T_world_boxTag @ inv(T_object_tag)`
- `T_world_torso = T_world_robotTag @ inv(T_torso_robotTag)`
- `T_torso_object = inv(T_world_torso) @ T_world_object`

**`T_object_tag`** = Step 1에서 확정한 박스 rigid frame과 **동일 정의** (태그 부착 면과 일치).

**✓ 게이트 (Step 2)**

- [ ] RealSense 프레임·intrinsics 정상
- [ ] AprilTag id·품질 로그
- [ ] 정지 시 `T_torso_object` jitter·missing 허용 범위
- [ ] tag family·mm 실측·10초 샘플 로그

**로봇 안전:** L2+B damp, L2+up stand 등 **팀 매뉴얼 순서** 고정.

---

### Step 3 — Phase C: Reference ↔ 실험실 align

수기+로그 (별도 스크립트 없음).

1. `T_realWorld_refWorld = T_realWorld_torso_0 @ inv(T_refWorld_torso_0)`
2. ref torso·ref object **동일 rigid**로 변환
3. aligned 초기 object pose에 **실제 상자** 배치

**✓ 게이트:** 상자 놓을 위치가 현장에서 재현 가능; `T_realWorld_refWorld` + 사진/스케치 보관.

---

### Step 4 — Phase D: 학습

`conda activate unitree_rl_mjlab` 유지.

#### 4-1. Velocity 백업 — **`$URML`**

```bash
cd /home/roy/unitree_rl_mjlab
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

학습 run 디렉터리 예: `logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/`  
저장 시 같은 폴더에 **`policy.onnx`** (+ `.data`) 자동 export (`src/tasks/velocity/rl/runner.py`).

#### 4-2. Manipulation (sub8_largebox_045) — **`$HP`**

```bash
cd /home/roy/realsense_calib/humanoid_project
python scripts/train.py Unitree-G1-Manipulation-Simple \
  --motion_file=src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.run-name=manip_simple_sub8_045
```

다중 클립: `$HP/src/assets/OmniRetarget/motion_sets/train.json` — `$HP/doc/my_docs/how_to_use.md`.

> manipulation 정책은 **아직 `g1_ctrl` FSM에 없음**. carrying 실기는 별도 엔지니어링(§4 항목 3) 또는 sim play로 먼저 검증.

#### 4-3. Mimic / tracking — **`$URML`**

```bash
cd /home/roy/unitree_rl_mjlab
# csv가 있으면:
python scripts/csv_to_npz.py --input-file <csv> --output-name <name> --input-fps 30 --output-fps 50 --robot g1

python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/<your>.npz \
  --env.scene.num-envs=4096
```

run 폴더에 `policy.onnx`, `policy.onnx.data`, 모션명 onnx(예: `dance1_subject2.onnx`) + 메타데이터 export.

**✓ 게이트 (Step 4)**

```bash
ls /home/roy/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/*/policy.onnx 2>/dev/null | tail -1
ls /home/roy/unitree_rl_mjlab/logs/rsl_rl/g1_tracking/*/policy.onnx 2>/dev/null | tail -1
ls /home/roy/realsense_calib/humanoid_project/logs/rsl_rl/g1_manipulation/*/model_*.pt 2>/dev/null | tail -1
```

---

### Step 5 — Phase E: Play (시뮬)

`<RUN>` = run 디렉터리, `<CKPT>` = `model_XXXX.pt`.

```bash
conda activate unitree_rl_mjlab

# manipulation — HP
cd /home/roy/realsense_calib/humanoid_project
python scripts/play.py Unitree-G1-Manipulation-Simple-Play \
  --checkpoint_file=/home/roy/realsense_calib/humanoid_project/logs/rsl_rl/g1_manipulation/<RUN>/<CKPT> \
  --motion-file=/home/roy/realsense_calib/humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz \
  --viewer auto

# velocity — URML
cd /home/roy/unitree_rl_mjlab
python scripts/play.py Unitree-G1-Flat \
  --checkpoint_file=/home/roy/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/<RUN>/model_xx.pt

# mimic — URML
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/<your>.npz \
  --checkpoint_file=/home/roy/unitree_rl_mjlab/logs/rsl_rl/g1_tracking/<RUN>/model_xx.pt
```

**✓ 게이트:** 낙상/terminate 없음; 박스 추종·접촉이 reference 대비 타당.

---

### Step 6 — Phase F: Deploy 패키징 (`$URML`, README 4절)

모든 경로는 **`/home/roy/unitree_rl_mjlab`** 기준.

#### 6-0. 디렉터리 맵 (기본 포함 템플릿)

```
deploy/robots/g1/config/
├── config.yaml                          # FSM: Passive → FixStand → Velocity → Mimic_*
└── policy/
    ├── velocity/v0/
    │   ├── exported/policy.onnx[.data]  # ← 학습 산출물 복사
    │   └── params/deploy.yaml
    └── mimic/dance1_subject2/             # 예시 mimic 슬롯
        ├── exported/policy.onnx[.data]
        └── params/deploy.yaml
        └── params/dance1_subject2.npz     # motion_file
```

#### 6-1. Velocity ONNX 배치

`<VEL_RUN>` = Step 4-1 run 폴더 (예: `logs/rsl_rl/g1_velocity/2026-05-15_12-00-00`).

```bash
export URML=/home/roy/unitree_rl_mjlab
export VEL_RUN=/home/roy/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/<VEL_RUN>

cp -v $VEL_RUN/policy.onnx* \
  $URML/deploy/robots/g1/config/policy/velocity/v0/exported/
```

`params/deploy.yaml`은 리포 기본값이 **예시 학습과 다를 수 있음** → 학습 env의 obs/action scale과 맞는지 확인. (불일치 시 튀거나 즉시 넘어짐.)

키보드로 속도 명령 디버깅(sim2sim): `deploy.yaml`에서 관측 이름 `velocity_commands` → `keyboard_velocity_commands` 로 바꾼 뒤 **재빌드** (`State_RLBase.cpp` 주석 참고).

#### 6-2. Mimic 슬롯 추가 (새 모션)

예: 학습한 `sub8` 모션을 `Mimic_sub8_045` 로 올릴 때 — **`dance1_subject2` 복사 후 이름 변경**:

```bash
export URML=/home/roy/unitree_rl_mjlab
export MIMIC_NAME=sub8_045
export TRK_RUN=/home/roy/unitree_rl_mjlab/logs/rsl_rl/g1_tracking/<TRK_RUN>

# 1) 템플릿 복사
cp -r $URML/deploy/robots/g1/config/policy/mimic/dance1_subject2 \
      $URML/deploy/robots/g1/config/policy/mimic/$MIMIC_NAME

# 2) ONNX
cp -v $TRK_RUN/policy.onnx $TRK_RUN/policy.onnx.data \
  $URML/deploy/robots/g1/config/policy/mimic/$MIMIC_NAME/exported/

# 3) 모션 npz (학습에 쓴 파일과 동일 스키마)
cp -v /home/roy/unitree_rl_mjlab/src/assets/motions/g1/<your>.npz \
  $URML/deploy/robots/g1/config/policy/mimic/$MIMIC_NAME/params/${MIMIC_NAME}.npz
# dance1 템플릿이면 기존 dance1_subject2.npz 삭제 또는 motion_file 이름과 맞춤
```

#### 6-3. FSM (`config.yaml`) 수정

파일: `/home/roy/unitree_rl_mjlab/deploy/robots/g1/config/config.yaml`

1. `FSM._` 아래에 새 mimic 상태 등록 (예시 `Mimic_Dance1_subject2` 복사).
2. `Velocity.transitions`에 mimic 진입 키 추가 (기본: `Mimic_Dance1_subject2: RB + A.on_pressed`).
3. 새 mimic 블록에 설정:
   - `motion_file: config/policy/mimic/<MIMIC_NAME>/params/<MIMIC_NAME>.npz`
   - `policy_dir: config/policy/mimic/<MIMIC_NAME>/`
   - `time_start` / `time_end`

**조이스틱 전환 (기본 매핑)**

| 전환 | 입력 |
|------|------|
| Passive → FixStand | `LT + up` |
| FixStand → Velocity | `RT + A` |
| Velocity → Mimic | `RB + A` (해당 mimic 이름으로 yaml에 정의) |
| Mimic → Velocity | `RT + A` |
| Any → Passive (비상) | `LT + B` |

#### 6-4. 컨트롤러 빌드

```bash
cd /home/roy/unitree_rl_mjlab/deploy/robots/g1
mkdir -p build && cd build
cmake .. && make -j8
```

**✓ 게이트 (Step 6)**

- [ ] `velocity/v0/exported/`에 `policy.onnx`(+`.data`)
- [ ] mimic `exported/` + `params/*.npz` + `deploy.yaml` 세트
- [ ] `config.yaml` FSM에 mimic 상태·전환 키 반영
- [ ] `deploy.yaml`의 `step_dt`, `joint_ids_map`, observation 이름이 **해당 학습 run과 일치**

---

### Step 7 — Phase G: Sim2Sim (`$URML`)

**터미널 1** — unitree_mujoco:

```bash
cd /home/roy/unitree_rl_mjlab/simulate
mkdir -p build && cd build
cmake .. && make -j8
./unitree_mujoco
# 게임패드 연결; 로봇: simulate/config
```

README와 동일하게 repo 루트에서도 가능:

```bash
/home/roy/unitree_rl_mjlab/simulate/build/unitree_mujoco
```

**터미널 2** — g1_ctrl:

```bash
cd /home/roy/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=lo
```

조작 순서: 시뮬 기동 → `g1_ctrl` → 패드로 `FixStand` → `Velocity` → (선택) `Mimic`.

**✓ 게이트:** Velocity·Mimic FSM 전환 정상. `step_dt`/obs 불일치면 **실기 금지**.

---

### Step 8 — Phase H: Shadow (lowcmd 없음)

- PC 이더넷: **192.168.123.222**, netmask **255.255.255.0** (`URML/README.md` 4.3).
- lowstate + vision + ONNX forward만; **lowcmd 미발행**.

**✓ 게이트:** action 범위·NaN·지연 OK.

---

### Step 9 — Phase I: 실기 (velocity → mimic)

#### 9-1. 로봇 전원·모드 (README 4.1–4.2)

1. 로봇 **매달린 상태**로 전원 → `zero-torque` 대기.
2. `L2 + R2` → `debug mode` (관절 댐핑).

#### 9-2. 네트워크·실행

```bash
ifconfig   # 이더넷 이름 확인 (예: enp5s0)
cd /home/roy/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=enp5s0    # sim2sim은 --network=lo
```

| `network` | 용도 |
|-----------|------|
| `lo` | sim2sim (Step 7) |
| `enp5s0` 등 | 실기 (PC–로봇 이더넷) |

#### 9-3. 현장 순서

1. 지지·저게인·e-stop  
2. 패드: `FixStand` → `Velocity` 안정화  
3. 짧은 mimic (`RB+A` 등 yaml에 정의된 키)  
4. (별도) carrying / manipulation — **현재 `g1_ctrl`에는 없음**, Step 4-2는 sim·연구용

**✓ 게이트:** safety monitor (roll/pitch, 높이, 관절 속도, action jump, tag missing 등).

---

### Step 10 — Phase J: 로깅

vision raw/filtered, `T_world_*`, `T_torso_object`, q/dq/tau/q_cmd, aligned reference, 추적 메트릭, safety event — **처음엔 전부 저장 후 offline 분석**.

---

### (참고) Step 5 이후 — viser로 manipulation 보기

replay는 MuJoCo만. 학습된 정책을 브라우저 viser로:

```bash
conda activate unitree_rl_mjlab
cd /home/roy/realsense_calib/humanoid_project
python scripts/play.py Unitree-G1-Manipulation-Simple-Play \
  --checkpoint_file=/home/roy/realsense_calib/humanoid_project/logs/rsl_rl/g1_manipulation/<RUN>/<CKPT> \
  --motion-file=/home/roy/realsense_calib/humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz \
  --viewer viser
```

---

## 4. 원문 `sim2real.md` 대비 “보충·수정” 요약

1. **배포 스택의 실체는 unitree_rl_mjlab** — mjlab README의 `uv run train/play`는 upstream 예시이며, **실기 배포 경로는 이 리포의 `deploy/` + `g1_ctrl`**이다.
2. **velocity → mimic 플로우**는 이미 FSM으로 존재; 새 모션/정책은 **mimic policy 디렉터리 복사 + config.yaml 수정**이 핵심 작업이다.
3. **AprilTag + object pose를 policy에 넣는 파이프라인**은 문서의 Part 1–3 수준이 필요하지만, **현재 g1 deploy 템플릿에는 vision 입력이 없음** — carrying 학습 시 vision을 썼다면 **학습 코드·export·C++ REGISTER_OBSERVATION·deploy.yaml**을 함께 추가하는 별도 엔지니어링이 필요하다.
4. **RoboJuDo / motion_tracking_controller**는 대안; unitree_rl_mjlab을 쓰면 **그쪽 스택과 ONNX 메타·관측 정의가 다를 수 있어** 혼용 시 전부 재검증.
5. **g1_spinkick_example**의 safe transition 아이디어는 그대로 유효; npz 전처리 단계에서 stand-in/out을 넣을 것.
6. 원문 후반의 **스크립트 목록(Part 1–11)**은 설계 체크리스트로 유효 — 실제 파일은 저장소에 일괄 구현되어 있지 않을 수 있으니, 필요한 것부터 작은 스크립트로 구현하면 된다.

---

## 5. 빠른 커맨드 치트시트

```bash
export REALSENSE_CALIB=/home/roy/realsense_calib
export HP=/home/roy/realsense_calib/humanoid_project
export URML=/home/roy/unitree_rl_mjlab
conda activate unitree_rl_mjlab

# manipulation / replay
cd $HP && python scripts/play.py replay sub8_largebox_045_original

# velocity 학습 + deploy ONNX 복사
cd $URML && python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
cp logs/rsl_rl/g1_velocity/<RUN>/policy.onnx* deploy/robots/g1/config/policy/velocity/v0/exported/

# sim2sim
cd $URML/simulate/build && ./unitree_mujoco                    # 터미널 1
cd $URML/deploy/robots/g1/build && ./g1_ctrl --network=lo      # 터미널 2

# 실기
cd $URML/deploy/robots/g1/build && ./g1_ctrl --network=enp5s0
```

---

## 6. 에이전트(또는 동료)에게 넘길 최소 정보 패키지

다음이 있으면 리뷰·디버깅이 빨라진다.

- 사용 리포 **커밋 해시** (mjlab / unitree_rl_mjlab 각각).
- 학습 task id, `deploy.yaml` 최종본, ONNX 한 세트.
- 모션 npz 샘플 + object/mesh frame 근거(스크린샷 또는 짧은 영상).
- vision 로그 10–30초 + align에 쓴 `T_realWorld_refWorld`.
- sim2sim에서 재현한 버그가 있다면 그때의 `g1_ctrl` 로그와 설정.

---

이 파일은 `sim2real.md`를 대체하지 않고, **실행 순서·리포 간 갭·게이트**를 보강한 운영 문서로 쓰면 된다. 원문의 수학·로깅 항목·안전 규칙은 필요 시 `sim2real.md`와 교차 참고한다.
