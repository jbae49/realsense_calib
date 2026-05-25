# sub8_45 sim2real deployment guide (G1)

이 문서는 `sub8_largebox_045` 모션 추종을 실제 G1에 올릴 때, 처음부터 끝까지 따라갈 수 있게 정리한 실행 가이드다.

기준 코드/경로:
- 프로젝트 루트: `/home/roy/realsense_calib/unitree_rl_mjlab`
- 실기 deploy: `/home/roy/realsense_calib/unitree_rl_mjlab/deploy/robots/g1`
- FSM 설정: `/home/roy/realsense_calib/unitree_rl_mjlab/deploy/robots/g1/config/config.yaml`
- 현재 버튼 매핑(기본):
  - `LT + up` (`L2 + up`): `Passive -> FixStand`
  - `RT + A` (`R2 + A`): `FixStand -> Velocity`
  - `RB + B` (`R1 + B`): `Velocity -> Mimic_Sub8_45` (proprio + npz only)
  - `RB + Y` (`R1 + Y`): `Velocity -> Mimic_Sub8_45_TagHistory` (카메라 obs 사용, §17 참고)
  - `RB + X` (`R1 + X`): `Velocity -> Mimic_Sub8_45_NoTag_History` (no-tag, history; IMU fallback anchor_ori)
  - `LB + X` (`L1 + X`): `Velocity -> Mimic_Sub8_45_NoTag_FCW5_History` (FC-C-W5 ablation, 동일 obs 스키마)
  - `SELECT`: `FixStand/Velocity/Mimic -> Passive` (빠른 소프트 정지 복귀)

> `Mimic_Sub8_45_NoTag_History` / `_FCW5_History`는 카메라가 필요 없다. 트래커
> (`track_robot_and_box_multicam.py`)를 안 켜도 된다. `motion_anchor_ori_b`는
> `State_Mimic.cpp`의 IMU + init_quat 폴백 경로를 쓴다 (calibration/ 서브디렉터리
> 없음). 동일 npz `sub8_largebox_045_original_extended.npz` 사용.

---

## 0) 실행 순서 Quick Start (가장 먼저)

질문이 많았던 부분을 먼저 정리한다.

### 0-1. 실기(sim2real)에서 무엇을 어디서 실행하나?

- 로봇 쪽:
  1) 전원 ON
  2) zero-torque 진입 확인
  3) 컨트롤러에서 `L2 + R2`로 debug mode 진입
- 노트북 쪽:
  1) 랜선 연결
  2) `ip -br a`로 NIC 확인
  3) 로봇 IP ping 확인
  4) `g1_ctrl --network=<NIC>` 실행

중요:
- Jetson 안에서 별도 스크립트를 실행해야만 하는 구조는 아니다.
- 현재 구성에서는 노트북에서 `g1_ctrl` 실행 시 `--network=eno1`처럼 NIC만 맞춰주면 된다.

### 0-2. `g1_ctrl` 실행 직후 로봇 상태 (코드 기준)

- `g1_ctrl`는 FSM을 `Passive` 상태로 시작한다.
- `Passive` 상태는 `kp=0`, `kd`만 적용되고, 관절 목표각 `q`는 현재값 유지다.
- 체감상 "댐핑 모드"처럼 보이는 게 정상이다.

따라서 실행 직후 바로 적극 제어가 되는 게 아니라, 버튼 전환으로 상태를 올려야 한다.

### 0-3. 버튼 순서 (현재 config 기준)

1. `LT + up` (`L2 + up`) -> `FixStand`
2. `RT + A` (`R2 + A`) -> `Velocity`
3. `RB + B` (`R1 + B`) -> `Mimic_Sub8_45` (sub8 extended onnx, proprio + npz only)
4. `RB + Y` (`R1 + Y`) -> `Mimic_Sub8_45_TagHistory` (카메라 obs 사용, §17 참고)
5. 언제든 `SELECT` -> `Passive` (안전 복귀)

실무 권장:
- **정상 진입 순서**: `Passive -> FixStand -> Velocity -> Mimic_*`
- mimic을 바로 켜기보다, `Velocity`에서 자세/균형/입력 상태를 2~3초 확인 후 mimic으로 전환
- `R1+Y` 누르기 전에는 반드시 PC 트래커 (`track_robot_and_box_multicam.py --udp-publish --motion-file <raw npz>`) 가 먼저 떠있어야 함. 안 그러면 g1_ctrl 콘솔에 `[cam] warm-up TIMEOUT` 경고 뜨고 첫 step 의 카메라 obs 가 비어있게 됨 (§17-5). **R1+Y 누른 후 트래커 창에서 SPACE 를 눌러야 motion 이 진행됨** — IDLE 상태에선 정책이 frame-0 자세 유지.

### 0-4. 네트워크 확인 커맨드 (실기)

```bash
ip -br a
ping -c 3 192.168.123.161
```

기본 mimic (proprio + npz only, 카메라 무관) 실행:

```bash
cd /home/roy/realsense_calib/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=eno1 --log
```

#### `Mimic_Sub8_45_TagHistory` (v2 publisher, runtime alignment) 정확한 순서

> **새 워크플로우 (2026-05-24)**: 트래커가 NPZ 를 직접 들고 있다가 SPACE 키 누른 순간에 `T_sim_lab` 을 계산해서 즉시 PLAYBACK. 더 이상 `align_npz_to_lab.py` 로 NPZ 를 미리 변환할 필요 없음 — raw `_extended.npz` 를 그대로 사용. 이전 워크플로우(별도 CSV 캡처 → 별도 align 스크립트 → aligned NPZ deploy)는 `align_npz_to_lab.py` 의 docstring 에서 진단/legacy 용으로 보존됨.

```bash
# (전제) 로봇 전원 ON, 컨트롤러 L2+R2 (debug) 까지 들어간 상태

cd /home/roy/realsense_calib
TS=$(date +%Y%m%d_%H%M%S)

# === 1) 트래커 (v2 publisher) 시작 ============================
python track_robot_and_box_multicam.py \
  --cam1-serial 935322072654 --cam2-serial 115222071236 --cam3-serial 112322072671 \
  --cam1-calib camera1_935322072654_calibration.npz \
  --cam2-calib camera2_115222071236_calibration.npz \
  --cam3-calib camera3_112322072671_calibration.npz \
  --origin-id 1 --anchor-ids 10 --margin-min 20 \
  --detector-quad-decimate 1.5 \
  --udp-publish \
  --motion-file unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/sub8_45/params/sub8_largebox_045_original_extended.npz \
  --align-mode yaw-only \
  --csv-out outputs/sub8_45_taghist_${TS}.csv --print-every 30
# 시작 로그 확인:
#   [multicam-v2] loaded motion: ...sub8_largebox_045_original_extended.npz
#   [multicam-v2] ref ticker thread started @ 50.0 Hz (IDLE; press SPACE ...)
# FUSED 패널에 'v2 PHASE: IDLE  press SPACE when robot is at start pose' 노란 글자.

# === 2) 다른 터미널에서 g1_ctrl 시작 ============================
cd /home/roy/realsense_calib/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=eno1 --log
# 시작 로그:
#   Loaded motion file 'sub8_largebox_045_original_extended' with duration X.XXs
#   (실제 NPZ 데이터는 v2 publisher 가 owns; C++ MotionLoader_ 는 fallback/duration 용)

# === 3) 조이스틱: FSM 진입 =====================================
# L2+up   -> FixStand
# R2+A    -> Velocity
# R1+Y    -> Mimic_Sub8_45_TagHistory
#         (콘솔에 [cam] warm-up ok ... format=v2 ... 보일 때까지 대기)
# 이 시점까지 정책은 v2 IDLE 패킷을 받아 frame 0 ref pose 에 락된 obs 로 step
# → 로봇은 frame 0 자세에서 가만히 서있음 (motion 진행 X).

# === 4) 박스 위치 정렬 (선택, 권장) =============================
# 트래커 카메라 창에 노란 십자가 + "place box here" 가 표시됨 (NPZ frame-0 박스
# 위치를 현재 torso pose 기준으로 lab frame 에 투영). 박스를 그 십자가 위치에
# 정확히 두면 'obj_pos_err' 0 에 가까워져서 정책이 안정.

# === 5) PLAYBACK 시작: 트래커 창에 SPACE ========================
# 트래커의 어떤 cv 창이든 포커스에서 SPACE 를 누르면 콘솔에:
#   [multicam-v2] SPACE: PLAYBACK started, mode=yaw-only, delta_yaw=...
# FUSED 패널은 'v2 PHASE: PLAYBACK  frame N/M  pkts=...' 초록색.
# 이때부터 50 Hz 로 ref motion 이 진행되고 카메라가 본 real torso/box 가
# T_sim_lab 으로 sim frame 변환되어 6 actor obs 가 계산됨. 정책은 motion 추종.

# === 6) 비상 / 종료 ============================================
# 컨트롤러 SELECT -> Passive (즉시 zero-torque 복귀)
# 트래커 창에서 SPACE 한 번 더 -> PLAYBACK -> IDLE (ref clock reset, 다시 frame 0)
# 종료 시 [multicam-v2] v2 UDP packets sent: <N>
```

**시작 자세가 변경되면** (로봇 위치 이동, 배터리 교체 등) → 트래커도 g1_ctrl 도 재시작할 필요 없음. SELECT 로 Passive 복귀, 트래커 창에서 SPACE 한 번 (PLAYBACK→IDLE), 새 자세에서 다시 SPACE → 새 `T_sim_lab` 계산, 즉시 PLAYBACK.

**구 워크플로우 (offline align):** `align_npz_to_lab.py` + `_extended_coords_processed_v2.npz` 경로는 `--motion-file` 없이 트래커를 띄우는 v1 모드 + 옛 deploy.yaml 조합으로 여전히 동작하지만, 새 코드에선 `State_Mimic.cpp` 의 6 obs 가 v1 패킷에 대한 fallback 만 남아 항상 zero/identity 를 반환하므로 정책이 의미 있게 동작하지 않는다. v2 모드를 사용하라.

### 0-5. "터미널에 전환 안내가 자동으로 뜨나?"에 대한 답

- `g1_ctrl`가 FSM의 모든 버튼 순서를 친절히 안내해주지는 않는다.
- 따라서 실제 전환 조합은 `config.yaml`을 기준으로 이해하고 운용해야 한다.
- 즉, 실행 로그를 기다리기보다 위 `0-3` 순서대로 직접 전환한다고 생각하면 된다.

---

## 0) 먼저 답: heightmap 쓰는 학습인가?

현재 deploy 입력을 보면 **heightmap(terrain height scan) 관측을 쓰지 않는다**.

근거:
- `deploy`의 실제 policy 입력(`deploy.yaml`)에는 `base_ang_vel`, `joint_pos_rel`, `joint_vel_rel`, `last_action`, `motion_*` 등만 있고 heightmap 계열 항목이 없다.
- tracking 학습 cfg도 평면 terrain 기반이며(`plane`) 핵심은 motion command + proprioceptive obs이다.

즉, 네가 지금 준비하는 `sub8_45 mimic` 파이프라인은 기본적으로 **heightmap 없이 동작하는 구성**으로 보는 게 맞다.

---

## 1) 동료에게 받아야 하는 파일 체크리스트 (가장 중요)

`pt`만 받으면 실기 deploy가 바로 안 될 수 있다. 아래 5개를 같이 받아라.

1. **ONNX 파일**
   - `policy.onnx` (필수)
2. **deploy 파라미터**
   - `deploy.yaml` (필수)
3. **모션 npz**
   - `sub8_largebox_045_original.npz` 또는 실사용 clip
4. **obs 스펙 정보**
   - obs term 순서/차원(학습 당시)
   - 예: `motion_command`, `motion_anchor_ori_b`, `base_ang_vel`, `joint_pos_rel`, `joint_vel_rel`, `last_action`
5. **학습 기준 정보**
   - action scale/offset, joint map, anchor body, step_dt(보통 0.02), quaternion convention(`wxyz`)

권장: 동료에게 아예 아래 폴더 형태로 받기

- `/path/to/sub8_45_bundle/exported/policy.onnx`
- `/path/to/sub8_45_bundle/params/deploy.yaml`
- `/path/to/sub8_45_bundle/params/sub8_largebox_045_original.npz`

---

## 2) NPZ 사전 점검 (실행 전 필수)

### 2-1. 키/shape 확인

```bash
cd /home/roy/realsense_calib/humanoid_project
python - <<'PY'
import numpy as np
p = "src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz"
d = np.load(p)
print("file:", p)
print("keys:", list(d.keys()))
for k in ["joint_pos","joint_vel","body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w"]:
    if k in d:
        print(k, d[k].shape, d[k].dtype)
PY
```

`deploy/robots/g1/src/State_Mimic.*` 경로는 위 키를 직접 읽는다. 누락/shape mismatch면 실기에서 바로 깨진다.

### 2-2. 기본 유효성

- 프레임 수 `T`가 모든 키에서 동일해야 함
- `body_quat_w`는 정규화 상태에 가까워야 함
- `joint_pos`, `joint_vel` dof가 G1 deploy 설정과 맞아야 함(현재 29 dof 구성)

---

## 3) 좌표계/관측 설계 원칙 (global vs local)

정리:
- 정책 입력은 대부분 **로봇 기준(local/body frame)** 으로 만들어진다.
- `npz`의 `body_*_w`는 world 기반이지만, obs 빌드 단계에서 로봇 anchor 기준으로 상대화된다.

tracking 쪽에서 실제로 하는 일:
- `MotionLoader`가 `npz`를 로드
- `motion_anchor_*_b`류 관측을 현재 로봇 anchor 기준으로 변환
- 최종적으로 정책엔 local/상대 정보가 들어감

### 3-1. `motion_anchor_pos/ori`는 센서로 "reference를 추정"하는 항목이 아님

자주 헷갈리는 포인트라 명확히 적는다.

- `motion_anchor_*`의 `motion` 쪽은 센서에서 추정하는 값이 아니라, `npz` reference에서 읽은 값이다.
- 센서(IMU/encoder, 내부 state estimator)는 "현재 로봇 상태"를 제공한다.
- 관측에서 하는 일은 **reference와 현재 상태의 상대값 계산**이다.

즉:
- `motion_anchor_pos_b`: reference anchor pos와 현재 robot anchor pos의 상대 위치
- `motion_anchor_ori_b`: reference anchor ori와 현재 robot anchor ori의 상대 회전

실기 C++ mimic도 같은 구조:
- reference quat: motion file(`npz`) + joint
- current quat: robot state(root quat + torso 관련 joint)
- 둘의 상대 회전을 `motion_anchor_ori_b`로 구성

주의:
- IMU만으로 절대 위치를 장시간 정확히 적분하는 것은 드리프트 때문에 어렵다.
- 그래서 `No-State-Estimation` tracking 설정에서는 actor obs에서
  `motion_anchor_pos_b`, `base_lin_vel`를 빼는 변형이 이미 제공된다.
- 절대 위치 정합이 중요하면 AprilTag/mocap/VIO/UWB 같은 외부 기준을 보강하는 것이 안전하다.

의미:
- sim2real 성공 조건은 "절대 world 원점 일치"보다도
- **정책이 기대한 관측 파이프라인(순서/스케일/지연/history)을 실기에서도 동일하게 재현**하는 것.

---

## 4) 실기 전 Dry-run (반드시 먼저)

### 4-1. Python viewer로 reference clip 확인

```bash
cd /home/roy/realsense_calib/humanoid_project
conda run -n unitree_rl_mjlab python scripts/play.py replay sub8_largebox_045_original
```

확인:
- clip 자체가 정상 재생되는지
- 시작 자세/방향이 실험실 세팅과 크게 어긋나지 않는지

### 4-2. 정책 + 모션 결합 dry-run (가능하면)

```bash
cd /home/roy/realsense_calib/humanoid_project
conda run -n unitree_rl_mjlab python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --checkpoint_file=/ABS/PATH/model_xxxxx.pt \
  --motion-file=/home/roy/realsense_calib/humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz
```

여기서 깨지면 실기 올리면 더 위험하다.

---

## 5) PT vs ONNX: 실기 deploy에 무엇이 필요한가

`deploy/robots/g1` C++는 ONNX Runtime으로 `policy.onnx`를 읽는다.  
즉, **실기에는 ONNX가 필수**다.

정리:
- `model_*.pt`: 학습/파이썬 런타임용
- `policy.onnx`: C++ 실기 deploy용

가장 안전한 운영:
- 동료에게 `policy.onnx + deploy.yaml + npz`를 같이 요청
- `pt`만 받는 경우엔 같은 코드베이스에서 ONNX export를 추가 수행해야 함

---

## 6) deploy 폴더에 sub8_45 mimic policy 배치

기본 dance 폴더를 복제해 sub8 전용 state로 만드는 방법.

### 6-1. 폴더 만들기

```bash
mkdir -p /home/roy/realsense_calib/humanoid_project/deploy/robots/g1/config/policy/mimic/sub8_45/exported
mkdir -p /home/roy/realsense_calib/humanoid_project/deploy/robots/g1/config/policy/mimic/sub8_45/params
```

### 6-2. 파일 복사/배치

```bash
cp /ABS/PATH/policy.onnx \
  /home/roy/realsense_calib/humanoid_project/deploy/robots/g1/config/policy/mimic/sub8_45/exported/policy.onnx

cp /ABS/PATH/deploy.yaml \
  /home/roy/realsense_calib/humanoid_project/deploy/robots/g1/config/policy/mimic/sub8_45/params/deploy.yaml

cp /home/roy/realsense_calib/humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz \
  /home/roy/realsense_calib/humanoid_project/deploy/robots/g1/config/policy/mimic/sub8_45/params/sub8_largebox_045_original.npz
```

### 6-3. FSM에 sub8 state 추가

수정 파일:
- `/home/roy/realsense_calib/humanoid_project/deploy/robots/g1/config/config.yaml`

수정 포인트:
- `FSM._` 아래에 `Mimic_Sub8_45` state 추가
- `Velocity.transitions`에 진입 버튼 추가
- `Mimic_Sub8_45` 블록에 아래 지정:
  - `motion_file: config/policy/mimic/sub8_45/params/sub8_largebox_045_original.npz`
  - `policy_dir: config/policy/mimic/sub8_45`
  - `time_start`, `time_end` 필요 시 지정
  - `end_state: Velocity` 권장

주의:
- 기존 `RB + A`가 `Mimic_Dance1_subject2`에 걸려 있으니 충돌 없이 재배치해야 함.
- 예: `RB + A`를 sub8로 바꾸고 dance는 임시 비활성화.

---

## 7) 빌드/실행 (g1_ctrl)

### 7-1. 빌드

```bash
cd /home/roy/realsense_calib/humanoid_project/deploy/robots/g1
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

### 7-2. NIC 확인

```bash
ip -br a
```

robot 연결 NIC 이름(예: `enp5s0`) 확인 후 실행:

```bash
cd /home/roy/realsense_calib/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=<YOUR_ROBOT_NIC>
```

---

## 7-A) Jetson 빌드 트러블슈팅 (실제 겪은 이슈 정리)

PC에서 빌드한 `g1_ctrl`은 **x86_64**라 Jetson(**aarch64**)에서 실행이 불가하다.
따라서 코드를 Jetson으로 옮긴 뒤 Jetson에서 다시 빌드해야 하는데, 이 과정에서 마주친 이슈들과 해결법을 순서대로 정리한다.

### 7-A-0. 사전: Jetson 접속

- 로봇망에서 Jetson은 `192.168.123.164` (사용자 환경 기준).
- 노트북 NIC를 같은 서브넷에 두고 `ssh unitree@192.168.123.164`로 접속.

### 7-A-1. ONNX Runtime 라이브러리 경로

증상:
```
g1_ctrl: error while loading shared libraries: libonnxruntime.so.1: cannot open shared object file
```

원인:
- `g1_ctrl` 바이너리의 RUNPATH가 빌드 당시 절대경로로 박혀 있어, Jetson에선 그 경로가 비어 있음.

해결 (Jetson에서):
```bash
echo 'export LD_LIBRARY_PATH=$HOME/unitree_rl_mjlab/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

확인:
```bash
ldd ~/unitree_rl_mjlab/deploy/robots/g1/build/g1_ctrl | grep onnx
```

> 매번 직접 export 하는 건 피곤하니까 `~/.bashrc`에 박아두는 게 권장.

### 7-A-2. unitree_sdk2 헤더가 구버전 (`dds_wrapper` 누락)

증상 (cmake 단계 또는 빌드 초반):
```
fatal error: unitree/dds_wrapper/robots/go2/go2.h: No such file or directory
fatal error: unitree/dds_wrapper/robots/g1/g1_pub.h: No such file or directory
```

원인:
- Jetson에 미리 설치돼 있던 `unitree_sdk2`가 오래된 버전이라 `dds_wrapper` 디렉토리 자체가 없음.
- 우리 코드는 신버전 헤더(`g1_pub.h`, `g1_sub.h` 등)를 include 함.

해결 (PC -> Jetson으로 신버전 헤더 복사):
```bash
# PC에서
rsync -av /home/roy/unitree_sdk2/include/unitree/ \
  unitree@192.168.123.164:/tmp/unitree_headers/

# Jetson에서
sudo rm -rf /usr/local/include/unitree
sudo mv /tmp/unitree_headers /usr/local/include/unitree
```

### 7-A-3. `libfmt` 없음 / 버전 충돌

증상:
```
/usr/bin/ld: cannot find -lfmt
```

원인 흐름:
- `unitree_sdk2` 신버전이 `spdlog` -> `libfmt`를 끌어쓴다.
- Jetson은 인터넷이 막혀 `apt install libfmt-dev` 불가.
- PC에서 `libfmt9_*.deb`/`libfmt-dev_*.deb`를 받아 `dpkg -i` 시도 -> Jetson(Ubuntu 20.04)의
  `libc6`/`libstdc++6`보다 최신을 요구해서 의존성 충돌.

해결 (소스 빌드, 가장 안정):
```bash
# PC에서: fmt 6.1.2 소스를 Jetson으로 전송
# (구 우분투/구 libstdc++와 ABI가 잘 맞는 안정 버전이 6.x대)

scp fmt-6.1.2.zip unitree@192.168.123.164:/tmp/

# Jetson에서
cd /tmp && unzip fmt-6.1.2.zip && cd fmt-6.1.2
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DFMT_TEST=OFF
make -j"$(nproc)"
sudo make install
sudo ldconfig

# 확인: libfmt.so.6 / libfmt.so 가 /usr/local/lib에 보이면 OK
ldconfig -p | grep -i libfmt
```

> **포인트**: Ubuntu 20.04 Jetson에서는 최신 `libfmt9/10`을 deb로 깔지 말고
> **소스에서 6.1.2 빌드**가 가장 트러블이 적다.

### 7-A-4. `libunitree_sdk2.a` 가 헤더와 안 맞아서 링크 단계 실패

증상 (링크 단계, `[100%] Linking CXX executable g1_ctrl` 직후):
```
undefined reference to `std::vector<...entity_properties...>&
  org::eclipse::cyclonedds::core::cdr::get_type_props<unitree_go::msg::dds_::MotorStates_>()'
undefined reference to ...get_type_props<unitree_hg::msg::dds_::LowCmd_>()
undefined reference to ...get_type_props<unitree_hg::msg::dds_::LowState_>()
undefined reference to ...get_type_props<unitree_hg::msg::dds_::HandState_>()
```

원인:
- 7-A-2에서 **신버전 헤더**만 교체하고 정작 `libunitree_sdk2.a`는 **구버전**이 그대로 있던 상태.
- 신버전 헤더는 새 IDL 메시지(`MotorStates_`, `HandState_`, `LowState_` 등)를 선언하는데,
  구버전 .a에는 그 IDL의 CDR serialization 심볼이 없음 -> `undefined reference` 폭주.

해결 (신버전 .a로 교체, aarch64 prebuilt 그대로 사용):
```bash
# PC에서
scp /home/roy/unitree_sdk2/lib/aarch64/libunitree_sdk2.a \
  unitree@192.168.123.164:/tmp/

# Jetson에서
sudo mv /usr/local/lib/libunitree_sdk2.a /usr/local/lib/libunitree_sdk2.a.OLD
sudo cp /tmp/libunitree_sdk2.a /usr/local/lib/libunitree_sdk2.a
sudo ldconfig
```

이후 빌드 디렉토리를 깨끗이 비우고 다시 빌드:
```bash
cd ~/unitree_rl_mjlab/deploy/robots/g1/build
rm -rf *
cmake ..
make -j"$(nproc)"
```

> **교훈**: SDK 업그레이드는 항상 **헤더 + .a 세트**로 같이 한다.
> 한쪽만 갈면 컴파일은 되도 링크에서 무더기 `undefined reference`가 뜬다.

### 7-A-5. `Clock skew detected. Your build may be incomplete.` 경고

증상:
```
make[2]: warning:  Clock skew detected.  Your build may be incomplete.
```

원인:
- `scp`로 PC에서 가져온 소스 파일의 mtime이 Jetson 시스템 시각보다 미래.
- Jetson 시계가 NTP 동기화 안 돼서 PC보다 며칠~몇년 늦으면 발생.

영향:
- **빌드 결과물엔 영향 없음**. 무시 가능.

깔끔하게 없애려면:
```bash
# Jetson에서 모든 소스 mtime을 현재 시각으로 통일
find ~/unitree_rl_mjlab -exec touch {} +
```

또는 NTP 동기화:
```bash
sudo timedatectl set-ntp true
```

### 7-A-6. Jetson 1회성 환경설정 요약 (해야 했던 것 모음)

```bash
# 1) ONNX Runtime 경로 (~/.bashrc)
export LD_LIBRARY_PATH=$HOME/unitree_rl_mjlab/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib:$LD_LIBRARY_PATH

# 2) /usr/local/include/unitree  (PC의 ~/unitree_sdk2/include/unitree 와 동일하게)
# 3) /usr/local/lib/libunitree_sdk2.a  (PC의 ~/unitree_sdk2/lib/aarch64/libunitree_sdk2.a 와 동일하게)
# 4) /usr/local/lib/libfmt.so* (소스 빌드된 fmt 6.1.2)
sudo ldconfig
```

이후엔 빌드/실행이 PC에서와 동일한 절차로 돌아간다.

---

## 7-B) Jetson에서 g1_ctrl 켜는 순서 (매번 따라가는 runbook)

빌드/환경설정 다 끝난 다음, **실제로 켤 때마다 따라가는 순서**다.
처음 한 번만 하면 되는 셋업은 7-A, 매 세션마다 하는 절차는 여기.

### 7-B-0. 사전 체크 (매번 빠르게)

PC(노트북)에서:
```bash
# Jetson과 같은 로봇망(192.168.123.x)에 노트북도 들어와 있는지 확인
ip -br a
ping -c 2 192.168.123.164      # Jetson
ping -c 2 192.168.123.161      # 로봇 본체
```

### 7-B-1. 로봇 준비 (실물)

1. 로봇 전원 ON, 거치대/지면에서 안정 자세 확인
2. zero-torque 모드 진입(자동) 확인
3. 컨트롤러에서 **`L2 + R2`** 길게 -> 초록 LED -> debug mode 진입
   - 이거 안 하면 controller 입력이 `g1_ctrl`까지 안 들어옴

### 7-B-2. Jetson 접속

PC에서:
```bash
ssh unitree@192.168.123.164
```

### 7-B-3. (한 번만) 환경변수 확인

`~/.bashrc`에 7-A-1의 `LD_LIBRARY_PATH`가 들어있는지 한 번만 확인:
```bash
grep onnxruntime ~/.bashrc
```

비어있으면 7-A-1대로 추가하고 `source ~/.bashrc`.

### 7-B-4. NIC 확인

```bash
ip -br a
```

기대 출력 예:
```
eth0   UP   192.168.123.164/24    <- 로봇망 (이걸 써야 함)
wlan0  UP   192.168.0.x/24        <- 일반 와이파이 (DDS 안 됨)
```

> **NIC는 `eth0`** (현재 Jetson 환경 기준).
> 노트북에서 돌릴 때만 `enp5s0`/`eno1`이지, Jetson은 `eth0`이다.

### 7-B-5. 빌드 산출물/policy 파일 확인

```bash
ls -la ~/unitree_rl_mjlab/deploy/robots/g1/build/g1_ctrl
ls -la ~/unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/sub8_45/exported/policy.onnx
ls -la ~/unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/sub8_45/params/deploy.yaml
ls -la ~/unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/sub8_45/params/*.npz
```

4개 다 있어야 함. 하나라도 없으면 PC에서 `scp`로 다시 보낸다.

### 7-B-6. 실행

```bash
cd ~/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=eth0
```

기대 시작 로그:
- `Passive` state로 진입
- DDS subscription "connected" 류 메시지
- 에러 없이 메인 loop 진입

### 7-B-7. FSM 진입 (컨트롤러)

7-B-1에서 debug mode 들어간 컨트롤러로:

1. `L2 + ↑(up)` -> **FixStand**
   - 다리가 펴지며 정자세 진입
2. `R2 + A` -> **Velocity**
   - 로코모션 정책 활성, 좌스틱으로 저속 이동 가능
3. `R1 + B` -> **Mimic_Sub8_45**
   - sub8 mimic 정책 시작
4. 언제든 `SELECT` -> **Passive** (안전 복귀)
   - 비상 상황엔 무조건 이거 + 필요 시 e-stop

> **권장**: `FixStand` 또는 `Velocity`에서 2~3초 안정성 확인 후 `Mimic_Sub8_45`로 전환.

### 7-B-8. 자주 막히는 포인트 빠른 진단

| 증상 | 원인 후보 |
| --- | --- |
| `g1_ctrl` 켜자마자 종료, `libonnxruntime.so.1` 에러 | 7-A-1 미적용 (`LD_LIBRARY_PATH`) |
| 컨트롤러 눌러도 state 전환 안 됨 | 1) `L2+R2` 안 함 2) NIC 잘못 (`--network=wlan0`) 3) 로봇과 케이블 미연결 |
| 콘솔에 `Unknown key name: SELECT` | `config.yaml`에 `SELECT.on_pressed`가 남아있음. `back.on_pressed`로 바꿔야 함 (DSL 키 이름 다름) |
| `Input name time_step not found in observations.` | `manager_based_rl_env.h`의 `time_step` 인젝션 패치 누락 -> rebuild 필요 |
| 빌드는 OK인데 실행 시 ONNX 차원 mismatch | onnx와 deploy.yaml 세트가 다른 학습 산출물 |

### 7-B-9. 종료

1. 컨트롤러 `SELECT` -> Passive
2. 터미널 `Ctrl+C` -> `g1_ctrl` 종료
3. 로봇은 자동으로 zero-torque로 돌아감

---

## 7-C) Cable-free 운영 (wifi로만 SSH, 랜선 분리)

매번 노트북-Jetson을 랜선으로 연결하지 않고, **공유기 wifi**로만 SSH해서 운영하는 셋업이다.
로봇 ↔ Jetson은 어차피 로봇 안의 케이블(eth0)로 통신하니까 노트북-Jetson 랜선만 빼는 거.

### 7-C-0. 핵심 그림

```
[노트북 wlo1: 192.168.0.3]
        │ wifi (SSH만)
        │
   [iptime 공유기 192.168.0.1]
        │ wifi
        │
[Jetson wlan0: 192.168.0.2]   <-- SSH가 가는 곳
        │ (같은 머신 내부)
[Jetson eth0:  192.168.123.164]
        │ wired (DDS 통신)
        ▼
[로봇 192.168.123.161]
```

요점:
- **wifi는 사람이 Jetson을 보기 위한 통로**일 뿐, 로봇과 DDS는 wifi 안 거침
- 로봇과의 실시간 통신은 Jetson `eth0` <-> 로봇 wired 로만 일어남 (저지연/안정)
- 따라서 노트북-Jetson 랜선이 빠져도 로봇 제어는 영향 없음

### 7-C-1. 1회성 wifi 셋업

**(a) Jetson을 실험실 wifi에 붙이기** (랜선 SSH로 들어와서):

```bash
# 가능한 wifi 스캔
sudo nmcli dev wifi list

# 연결 (SSID/비밀번호는 본인 거)
sudo nmcli dev wifi connect "iptime" password "<WIFI_PASSWORD>"

# 자동연결 보장
nmcli con mod "iptime" connection.autoconnect yes

# Jetson의 wifi IP 확인 (메모!)
ip -4 addr show wlan0
```

> Jetson은 보통 2.4GHz `iptime`만 잡힘 (Wi-Fi 어댑터에 따라). 노트북은 5GHz `iptime5G`라도 같은 공유기면 OK.

**(b) 노트북도 같은 공유기 wifi에 붙이기**:

같은 iptime 공유기면 SSID는 `iptime`/`iptime5G` 어느 쪽이든 통합 LAN이라 무관.

**(c) "정말 같은 wifi인지" 검증** (3가지):

노트북 + Jetson 각각:
```bash
ip route | grep default
```
둘 다 `default via 192.168.0.1` 같은 식으로 같은 게이트웨이가 보이면 같은 공유기 = 같은 LAN.

노트북에서:
```bash
ping -c 3 192.168.0.2     # Jetson wlan0
```
응답 오면 통신 OK 확정.

> ping이 실패하면 공유기의 "무선 격리(AP isolation)" 설정 확인.
> 192.168.0.1 관리자 페이지에서 OFF.

### 7-C-2. SSH 별칭 정리

`~/.bashrc`에 wired/wifi 둘 다 등록:

```bash
# 노트북에서 한 번만
sed -i "s|^alias g1=.*|alias g1='ssh unitree@192.168.123.164'|" ~/.bashrc
echo "alias g1w='ssh unitree@192.168.0.2'" >> ~/.bashrc
source ~/.bashrc
```

사용:
- `g1` -> wired SSH (랜선 꽂혔을 때)
- `g1w` -> wifi SSH (랜선 빠졌을 때)

> Jetson wifi IP가 DHCP라 재부팅 후 바뀔 수 있다. 자주 바뀌면 iptime 관리자(192.168.0.1)에서 MAC 기반 고정 IP 등록 권장.
> Jetson MAC 확인: `ip link show wlan0 | awk '/ether/ {print $2}'`

### 7-C-3. tmux 안에서 g1_ctrl 띄우기 (필수)

SSH 세션이 끊겨도 g1_ctrl이 살아있게 하려면 반드시 `tmux` 사용.

```bash
g1w                                   # 또는 g1 (wired)

# Jetson 안에서
tmux new -s g1                        # 세션 생성
cd ~/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=eth0
# 정상 시작 로그 확인 (FSM: Start Passive)

# detach: Ctrl+B 누르고 D
```

이제 랜선 빼거나 wifi 잠깐 끊겨도 g1_ctrl은 Jetson 안에서 살아있다.

### 7-C-4. 다시 붙기 / 종료

```bash
# 다시 붙기 (모니터링 재개)
g1w                                   # 또는 g1
tmux attach -t g1

# 종료
# tmux attach 한 상태에서:
#   Ctrl+C  ->  g1_ctrl 종료
#   exit    ->  tmux 세션 종료
```

> Ctrl+C는 **컨트롤러로 SELECT(Passive 복귀)** 한 다음에 누르는 게 안전.

### 7-C-5. "랜선 빼면 SSH는 죽지만 g1_ctrl은 살아있다"의 이유

| 무엇 | 어디 경로 | 랜선 빼면? |
| --- | --- | --- |
| 노트북 SSH (wired용 `g1`) | 노트북 eno1 - 랜선 - Jetson eth0 | **끊김** (TCP keepalive timeout 후 세션 종료) |
| 노트북 SSH (wifi용 `g1w`) | 노트북 wlo1 - 공유기 wifi - Jetson wlan0 | 영향 없음 |
| Jetson 안의 g1_ctrl 프로세스 | Jetson 내부 | 영향 없음 (단, SSH 세션이 부모면 SIGHUP으로 같이 죽음 -> tmux로 분리 필요) |
| 로봇 ↔ Jetson DDS | 로봇 - 로봇 안 케이블 - Jetson eth0 | 영향 없음 (이건 노트북 랜선과 무관) |

그래서 **`g1w` + `tmux`** 조합이 cable-free 운영의 정답.

### 7-C-6. 자주 막히는 포인트 (cable-free)

| 증상 | 원인 / 해결 |
| --- | --- |
| `g1` 별칭만 안 됨, ping은 잘 됨 | 별칭이 옛 IP를 가리킴. 7-C-2 수정 |
| `ssh: No route to host` (wifi 시도) | 노트북 wifi 꺼짐(wlo1 DOWN), 또는 Jetson wifi 끊김 |
| `ping 192.168.0.2` 실패 | (1) Jetson wifi 안 붙음 (2) 공유기 AP isolation (3) 다른 공유기 wifi |
| 랜선 빼니까 g1_ctrl도 죽음 | tmux 안 썼음. 7-C-3 다시 |
| Jetson 재부팅 후 wifi IP 바뀜 | DHCP lease 만료. iptime 관리자에서 MAC 기반 고정 IP 등록 |
| Jetson에 wifi 어댑터는 보이는데 SSID 스캔 안 됨 | `sudo systemctl restart NetworkManager` 한 번 |

---

## 8) FSM 조작 순서 (실기)

현재 기본 매핑 기준:

1. 초기: `Passive`
2. `LT + up` (`L2 + up`) -> `FixStand`
3. `RT + A` (`R2 + A`) -> `Velocity` (기본 로코모션)
4. `RB + B` (`R1 + B`) -> `Mimic_Sub8_45`
5. 언제든 `SELECT` -> `Passive` (소프트 정지)

운영 권장:
- 무조건 `FixStand`에서 2~3초 안정 확인 후 mimic 진입
- 실험자 1명은 게임패드에서 `SELECT`만 담당
- 별도 하드웨어 e-stop은 손 닿는 위치에 항상 배치

---

## 9) mimic 시작 위치/자세 맞추기 (좌표계 정합)

핵심:
- mimic 켜는 순간 로봇 자세가 레퍼런스 시작프레임과 크게 어긋나면 실패 확률이 급상승
- 특히 yaw(방향), 발 위치, 박스 상대 위치를 먼저 맞춰야 함

권장 절차:
1. `npz` 시작 프레임의 torso/object pose 확인
2. 실험실에서 로봇/박스를 그 근처로 배치
3. vision으로 측정한 관측과 ref 간에 `T_ref_lab` 추정

사용 스크립트(이미 작성됨):
- `/home/roy/realsense_calib/compute_ref_alignment_yaw_only.py`

예시:

```bash
cd /home/roy/realsense_calib
python compute_ref_alignment_yaw_only.py \
  --obs-csv outputs/init_obs_torso_obj.csv \
  --ref-npz humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz \
  --out-json config/T_ref_lab_sub8_45.json
```

---

## 10) AprilTag가 로봇에도 반드시 필요한가?

짧게:
- **로컬 프레임에서 box를 계산하는 데 로봇 태그가 반드시 필요하지는 않다.**

케이스:
1. 내부 state estimator(imu+encoder)로 torso 상태를 신뢰할 수 있으면
   - 외부 비전은 box(world 또는 camera 기준)만 안정 추정해도 된다.
2. estimator 드리프트/초기 yaw 불확실성이 크면
   - head tag 등 외부 태그로 초기 정렬/검증을 추가하는 게 안전하다.

실무적으로는:
- 내부 estimator를 기본으로 쓰고
- AprilTag는 초기 정렬/진단/드리프트 모니터용 보조로 두는 구성이 가장 현실적이다.

---

## 11) Jetson 실기 전 점검 체크리스트

### 11-1. 시스템/성능

```bash
uname -a
cat /etc/nv_tegra_release
sudo nvpmodel -q
tegrastats
```

### 11-2. 네트워크/장치

```bash
ip -br a
ping -c 3 <robot_ip>
ls -l /dev/input/js0
```

### 11-3. 파일 배치 최종 확인

```bash
ls -la /home/roy/realsense_calib/humanoid_project/deploy/robots/g1/config/policy/mimic/sub8_45/exported
ls -la /home/roy/realsense_calib/humanoid_project/deploy/robots/g1/config/policy/mimic/sub8_45/params
```

---

## 12) state estimator sanity check (해야 함)

"그냥 쓰면 되나?"에 대한 답:  
**바로 쓰지 말고 최소 sanity는 반드시 확인**.

최소 체크:
- 정지 시 base angular velocity가 작은지(과도 노이즈/바이어스 없는지)
- 자세 유지 시 롤/피치 드리프트가 급격하지 않은지
- joystick 입력 없는 상태에서 관절/베이스 상태가 튀지 않는지
- `Velocity` 모드에서 저속 전/후진 시 바디 응답이 예측 가능하게 나오는지

이 단계가 이상하면 mimic는 거의 반드시 실패한다.

---

## 13) 실패했을 때 가장 먼저 볼 것

1. ONNX/`deploy.yaml`/`npz` 조합이 같은 학습 실험 산출물인지
2. obs term 순서/차원 mismatch 여부
3. mimic 진입 시 시작 자세/방향(yaw) mismatch
4. state estimator 품질(정지 드리프트/노이즈)
5. FSM 버튼 충돌/잘못된 state 전환
6. NIC 선택 오류 (`--network`)로 DDS 통신 불안정

---

## 14) 1회 실행용 실전 순서 (요약)

1. `npz` 키/shape 점검
2. 동료 산출물(`policy.onnx`, `deploy.yaml`) 확보
3. `deploy/.../mimic/sub8_45` 폴더 배치
4. `config.yaml`에 `Mimic_Sub8_45` 등록 + 버튼 매핑
5. `g1_ctrl` 빌드
6. NIC 확인 후 `./g1_ctrl --network=<nic>`
7. `LT+up -> RT+A`로 Velocity 안정 확인
8. 시작 자세/박스 위치 맞춤
9. `RB+A`로 mimic 진입
10. 이상 징후 시 즉시 `LT+B`, 필요 시 하드 e-stop

---

## 15) 다음 액션 (권장)

실제 운영 전에 아래 2개를 추가하면 실패율이 크게 줄어든다.

1. `sub8_45` 전용 FSM 상태를 dance와 분리(버튼 충돌 제거)
2. mimic 진입 직전 자동 pre-check(자세/IMU/통신 상태 OK일 때만 진입)

---

## 16) 참고: `unitree_rl_mjlab` task/obs 맵 (요약)

아래는 `unitree_rl_mjlab`에서 G1 기준으로 실무에서 자주 보는 task들과 obs 구성을 정리한 참고자료다.

### 16-1. G1 관련 대표 task

- `Unitree-G1-Rough` (velocity locomotion)
  - 등록 위치: `unitree_rl_mjlab/src/tasks/velocity/config/g1/__init__.py`
- `Unitree-G1-Flat` (velocity locomotion)
  - 등록 위치: `unitree_rl_mjlab/src/tasks/velocity/config/g1/__init__.py`
- `Unitree-G1-Tracking` (motion tracking)
  - 등록 위치: `unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py`
- `Unitree-G1-Tracking-No-State-Estimation` (tracking 변형)
  - 등록 위치: `unitree_rl_mjlab/src/tasks/tracking/config/g1/__init__.py`

실기 deploy(FSM)에서는 학습 task_id 대신 state 이름으로 동작한다.
- `Velocity` state (로코모션 정책)
- `Mimic_*` state (npz reference 추종 정책)
- 설정 위치: `unitree_rl_mjlab/deploy/robots/g1/config/config.yaml`

### 16-2. task별 actor obs 핵심

- `Velocity` (`Unitree-G1-Rough`/`Flat`)
  - 기본: `base_ang_vel`, `projected_gravity`, `command(twist)`, `phase`,
    `joint_pos`, `joint_vel`, `actions`
  - rough에서는 `height_scan` 포함, flat에서는 제거
  - 정의 위치: `unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py`,
    `unitree_rl_mjlab/src/tasks/velocity/config/g1/env_cfgs.py`

- `Tracking` (`Unitree-G1-Tracking`)
  - `command(motion)`, `motion_anchor_pos_b`, `motion_anchor_ori_b`,
    `base_lin_vel`, `base_ang_vel`, `joint_pos`, `joint_vel`, `actions`
  - 정의 위치: `unitree_rl_mjlab/src/tasks/tracking/tracking_env_cfg.py`

- `Tracking-No-State-Estimation`
  - 위 Tracking에서 actor obs 일부 제거:
    `motion_anchor_pos_b`, `base_lin_vel` 제외
  - 위치: `unitree_rl_mjlab/src/tasks/tracking/config/g1/env_cfgs.py`

- 실기 `Mimic` deploy (`State_Mimic`)
  - deploy 예시 obs: `motion_command`, `motion_anchor_ori_b`,
    `base_ang_vel`, `joint_pos_rel`, `joint_vel_rel`, `last_action`
  - 예시 파일: `unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/dance1_subject2/params/deploy.yaml`

### 16-3. "각 obs를 어떻게 추정하나?" (센서/연산 소스)

- `joint_pos`, `joint_vel`
  - 모터 엔코더/저수준 state에서 온다.
- `base_ang_vel`, `base_lin_vel`, `projected_gravity`
  - IMU + 내부 state estimator 기반.
- `command(twist)`
  - 속도 명령 생성기(joystick/command term)에서 생성.
- `motion_command`
  - `npz`의 `joint_pos/joint_vel`를 현재 시각(time index) 기준으로 읽음.
- `motion_anchor_pos_b`, `motion_anchor_ori_b`
  - "센서로 reference를 추정"하는 값이 아님.
  - reference anchor는 `npz`에서 읽고, robot anchor는 현재 state에서 읽은 뒤,
    두 프레임의 상대변환으로 계산한다.
  - 구현 위치:
    - `unitree_rl_mjlab/src/tasks/tracking/mdp/commands.py`
    - `unitree_rl_mjlab/src/tasks/tracking/mdp/observations.py`
- `last_action`
  - 직전 policy action을 재사용한 내부 상태 항목.

### 16-4. 실기 `Mimic`에서 orientation이 만들어지는 방식

- reference 쪽:
  - `State_Mimic::MotionLoader_`가 `npz`에서 root/joint를 읽음.
- real 쪽:
  - 로봇 root quaternion + 특정 torso 관절각(motor state)으로 현재 torso quat 구성.
- 관측:
  - 두 쿼터니언 상대회전의 회전행렬 일부(6D representation)를
    `motion_anchor_ori_b`로 넣음.
- 구현 파일:
  - `unitree_rl_mjlab/deploy/robots/g1/src/State_Mimic.cpp`
  - `unitree_rl_mjlab/deploy/robots/g1/include/State_Mimic.h`

요약하면, `unitree_rl_mjlab`은 "proprio only" 한 종류가 아니라
task마다 obs가 다르고, tracking/mimic 계열은 `npz reference + 현재 state`
를 결합한 상대표현 obs를 사용한다.

---

## 17) Tag-history 정책 (sub8_45_tag_history) deploy

기존 `sub8_45` 정책은 **proprio + npz**만 obs로 받음 (카메라 무관). 새 `sub8_45_tag_history`
정책은 actor obs 12개 중 **5개가 카메라 기반**:

| obs | 의미 | 소스 |
|---|---|---|
| `motion_anchor_pos_b` | ref torso pos (lab) − robot torso pos, 로봇 torso body frame에서 | 카메라 (torso) + npz (ref) |
| `object_pos_torso` | 박스 pos − robot torso pos, body frame | 카메라 (박스 + torso) |
| `object_ori6_torso` | 박스 quat 을 robot torso frame으로 (rot6d) | 카메라 (박스 + torso) |
| `ref_object_pos_torso` | npz의 박스 pos − robot torso pos, body frame | 카메라 (torso) + npz (ref) |
| `ref_object_ori6_torso` | npz 박스 quat을 robot torso frame으로 (rot6d) | 카메라 (torso) + npz (ref) |

→ **멀티캠 트래커가 매 프레임 torso/box pose를 UDP로 송신**, deploy 측 subscriber 가 latest snapshot 을 mutex-protect 해서 정책 step마다 읽음.

### 17-1. 아키텍처

기본 (PC-only, 권장) — 한 머신에서 트래커 + g1_ctrl 둘 다 실행:

```
[3× RealSense] -USB-> [PC]
                       ├─ track_robot_and_box_multicam.py --udp-publish
                       │     └─ UDP ASCII 17-fields packet -> 127.0.0.1:9999  (loopback, <1ms)
                       └─ g1_ctrl  (같은 PC)
                            ├─ camera_pose_subscriber  (UDP 9999 listener)
                            ├─ State_Mimic obs functions
                            ├─ ONNX policy (50 Hz)
                            └─ DDS  ─wired ethernet─> [로봇 192.168.123.161]
```

옵션 (split-machine, legacy) — 트래커는 PC, g1_ctrl 은 Jetson:

```
[3× RealSense] -USB-> [PC]                                  [Jetson]
                       └─ tracker --udp-host 192.168.123.164 ─→ camera_pose_subscriber
                                                                 └─ g1_ctrl ─→ 로봇
```

세 가지 구성 요소는 두 모드 모두에서 동일하게 사용됨:
- Publisher: `track_robot_and_box_multicam.py` (`--udp-publish` 플래그)
- Subscriber: `unitree_rl_mjlab/deploy/include/camera_pose_subscriber.h`
- Obs 함수: `unitree_rl_mjlab/deploy/robots/g1/src/State_Mimic.cpp`

> **PC-only 가 권장되는 이유**: ① UDP loopback latency 가 마이크로초 단위라 staleness/지터 우려 사라짐, ② NTP 동기화 자체가 필요 없음 (같은 monotonic clock), ③ 빌드 → 실행 사이클이 빠름 (scp 불필요), ④ gdb/perf/csv 디버깅 도구 모두 PC 에서 풍부.

### 17-2. 좌표 정합

모든 데이터가 **lab frame (camera tracker의 origin tag frame)**에서 일관되게 표현됨:
- 카메라가 보내는 torso/box pose: lab frame ✓
- npz의 ref pose: 반드시 `align_npz_to_lab.py`로 변환된 v2 NPZ 사용
  - 즉 `motion_file: .../sub8_45_extended_coords_processed_v2.npz`
- 정책의 출력 (action)은 motor command 그대로

> **주의**: 두 정책 모두 raw npz (`_extended.npz`) 와 동작. **차이는 트래커 측**: tag_history 는 트래커를 `--motion-file <raw npz>` 로 띄워 v2 publisher 모드 활성화 → SPACE 키 누르면 그 순간의 real torso pose vs NPZ frame 0 으로 `T_sim_lab` 을 계산하고 50 Hz 로 6 obs + motion_command 를 NPZ sim frame 으로 송신. 일반 sub8_45 는 v1 (legacy) — 트래커는 lab frame torso/box 만 보냄 (motion-only 정책은 어차피 카메라 obs 안 씀).

### 17-3. UDP wire format

두 wire format 이 호환됨 (subscriber 가 `v2 ` prefix 로 자동 분기):

**v1 (legacy, `--motion-file` 없을 때)**: ASCII 한 줄, 17 numeric fields, ~150 bytes:
```
<ts_ns> <torso_v> <tx> <ty> <tz> <tqw> <tqx> <tqy> <tqz> <box_v> <bx> <by> <bz> <bqw> <bqx> <bqy> <bqz>
```
- pose 는 lab frame, quat 은 (w,x,y,z). motion-only 정책 (`Mimic_Sub8_45`, `Mimic_Dance1_subject2`) 의 warm-up 신호로 쓰임 (실제 obs 값은 안 씀).

**v2 (sub8_45_tag_history)**: ASCII 한 줄, prefix `"v2 "`, ~750 bytes:
```
v2 <ts_ns> <phase 0|1> <frame_idx> <num_frames> <dof>
   <jp_0..jp_(dof-1)> <jv_0..jv_(dof-1)>
   <map_x map_y map_z> <mao_0..mao_5>
   <opt_x opt_y opt_z> <oot_0..oot_5>
   <rpt_x rpt_y rpt_z> <rot_0..rot_5>
```
- `phase` : 0=IDLE (publisher 송신 frame_idx=0, 6 obs 는 "real==ref" 즉 zero/identity 로 보냄 → 정책 stand still), 1=PLAYBACK (`T_sim_lab` 적용 + frame_idx 50Hz tick 진행).
- `dof` : 29 (G1).
- `jp / jv` : NPZ joint_pos / joint_vel (motion_command obs).
- 나머지 6 필드 : 6 actor obs (motion_anchor_pos_b/ori_b, object_pos_torso/ori6_torso, ref_object_pos_torso/ori6_torso). 모두 NPZ sim frame.

수신 측 (`g1_ctrl`) 은 자체 monotonic clock 으로 staleness 체크 → **NTP 동기화 불필요**.

#### Sample-and-hold + warm-up (2026-05-23 보강)

초기 구현은 200ms 이상 stale 시 obs를 zero/identity로 떨어뜨렸음. 이건 위험: (a) R1+Y 직후 첫 패킷 도착 전 50–200ms 동안 정책이 zero obs로 step → 발산, (b) 한 프레임만 detection이 빠져도 obs가 origin으로 점프해서 `motion_anchor_pos_b` 가 폭발. 두 단계 보강:

1. **Sample-and-hold cache**: `latest_camera_poses()` 가 "마지막 valid snapshot" 을 영구 보관. UDP 드롭/AprilTag 한 프레임 dropout 가 일어나도 가장 최근 *valid* 값을 그대로 유지. 200ms 초과 시 throttled `spdlog::warn` ("stale pose age=...ms"), 2s 초과 시 `spdlog::error` ("HARD STALE — publisher likely dead"). 단, 값을 zero로 떨어뜨리진 않음 (그게 더 위험).
2. **`enter()` warm-up**: FSM 진입 시 `camera_pose_sub.wait_for_first_packet(1500ms)` 로 첫 패킷 도착까지 블로킹. 도착하면 `[cam] warm-up ok` 로그, 타임아웃이면 `[cam] warm-up TIMEOUT` 경고 → 트래커가 안 켜져 있다는 신호. 즉 **R1+Y 누르기 전에 PC `track_robot_and_box_multicam.py --udp-publish` 가 켜져 있어야 함** (그래야 첫 step에 valid obs).

### 17-4. 빌드 절차 (PC-only 모드)

```bash
# (PC) 빌드
cd ~/realsense_calib/unitree_rl_mjlab/deploy/robots/g1/build
cmake --build .       # State_Mimic.cpp + camera_pose_subscriber.h 컴파일됨
```

> **`align_npz_to_lab.py` 는 더 이상 deploy 전제 조건이 아님.** v2 publisher 가 같은 변환을 SPACE 키 누른 시점에 runtime 으로 수행. 옛 스크립트는 진단/legacy 용으로 보존됨 (`align_npz_to_lab.py` docstring 참고).

빌드 산출물 위치:
- PC 바이너리: `~/realsense_calib/unitree_rl_mjlab/deploy/robots/g1/build/g1_ctrl`
- raw NPZ (트래커 `--motion-file` 인자): `~/realsense_calib/unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/sub8_45/params/sub8_largebox_045_original_extended.npz`

> **Split-machine (Jetson) 모드**: §7-A / §7-B 참고. PC-only 모드와 같은 코드/헤더가 그대로 쓰이고, 차이는 (a) Jetson 에 한번 빌드해두기, (b) PC 트래커에서 `--udp-host 192.168.123.164` 로 명시.

### 17-5. 실행 시퀀스 (PC-only, 권장)

```bash
# 터미널 1 — v2 publisher 시작 (NPZ owns publisher; SPACE 로 phase 전환)
cd ~/realsense_calib
python track_robot_and_box_multicam.py \
  --cam1-serial 935322072654 --cam2-serial 115222071236 --cam3-serial 112322072671 \
  --cam1-calib camera1_935322072654_calibration.npz \
  --cam2-calib camera2_115222071236_calibration.npz \
  --cam3-calib camera3_112322072671_calibration.npz \
  --origin-id 1 --anchor-ids 10 --margin-min 20 \
  --detector-quad-decimate 1.5 \
  --udp-publish \
  --motion-file unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/sub8_45/params/sub8_largebox_045_original_extended.npz \
  --align-mode yaw-only \
  --csv-out outputs/sub8_45_taghist.csv --print-every 30
# 시작 로그: [multicam-v2] loaded motion: ...
#            [multicam-v2] ref ticker thread started @ 50.0 Hz (IDLE; press SPACE ...)
```

```bash
# 터미널 2 — g1_ctrl 시작 (PC 에서 직접)
cd ~/realsense_calib/unitree_rl_mjlab/deploy/robots/g1
./build/g1_ctrl --network <PC_NIC> --log
```

콘솔 + GUI 시퀀스:
1. 터미널 1 (트래커): FUSED 패널에 `v2 PHASE: IDLE  press SPACE when robot is at start pose` 노란색.
2. 터미널 2 (g1_ctrl): 컨트롤러로 `L2+R2` (debug) → `L2+up` (FixStand) → `R2+A` (Velocity) → `R1+Y` (Mimic_Sub8_45_TagHistory).
3. 터미널 2 에 `[cam] warm-up ok: first packet received (recv_count=..., format=v2, ...)` 로그가 떠야 정상.
4. 이 시점에 정책은 IDLE 패킷 obs 로 step → 로봇은 NPZ frame 0 자세에서 가만히 서있음.
5. (선택) 트래커 카메라 창의 노란 십자가 (NPZ frame-0 박스 위치 투영) 에 박스 정렬.
6. **트래커 창 포커스에서 SPACE** → 콘솔에 `[multicam-v2] SPACE: PLAYBACK started, mode=yaw-only, delta_yaw=...`. 패널은 `v2 PHASE: PLAYBACK frame N/M` 초록색. motion 시작.
7. 비상: 컨트롤러 `SELECT` → Passive. 트래커 SPACE 한번 더 → IDLE 복귀 (ref clock reset).

### 17-6. 디버그 / 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `Observation term 'motion_anchor_pos_b' is not registered` | 빌드 누락. `cd .../g1/build && cmake --build .` 다시. |
| `[cam] warm-up TIMEOUT after 1500ms` | 트래커가 안 떠있거나 UDP 가 도착 못 함. 터미널 1 의 `UDP publish ->` 로그 + `tcpdump -i lo udp port 9999` 로 확인. |
| `[cam] warm-up ok ... format=v1` (v2 기대했는데 v1 떴음) | 트래커에 `--motion-file` 인자를 안 줬거나 `--udp-publish` 가 빠짐. 콘솔에서 `loaded motion: ...` 라인이 보이는지 확인. |
| `[cam] no v2 packet — '...' falling back to identity/zero` | v2 패킷이 stale. 트래커 CSV 의 `head_visible`/`torso_visible` % 확인. SPACE 누르기 전 IDLE 단계엔 정상 — PLAYBACK 이후라면 트래커 재시작. |
| `[multicam-v2] SPACE ignored: no valid torso latch yet` | 카메라가 head/pelvis tag 를 아직 못 잡음. 로봇이 카메라 시야 안에 있는지 + 태그 가려졌는지 확인. |
| 정책 시작 직후 흔들리거나 발산 | (a) `--align-mode full-rotation` 으로 시작 자세 roll/pitch 가 baked in 됨 → `yaw-only` 로 다시. (b) 박스가 노란 십자가 위치와 멀리 떨어져 있음 → 박스 재배치 후 SPACE. (c) SPACE 누른 시점에 로봇이 흔들리고 있었음 → SELECT 로 Passive, FixStand 다시, SPACE 다시. |
| `recv_count=0` 인 채로 진행 | 포트/host mismatch. 트래커 `--udp-host`/`--udp-port` ↔ deploy `CAMERA_POSE_BIND_ADDR`/`CAMERA_POSE_PORT` env 일치 확인. |
| obs는 정상 같은데 로봇이 가만히 있음 | (a) 트래커가 IDLE 상태 → SPACE 가 필요. (b) FSM 진입 안 됨. 컨트롤러 EM 풀려있는지(L2+R2 후), Velocity 상태 거쳤는지 점검. |

env 로 포트/bind 변경:
```bash
export CAMERA_POSE_PORT=9888           # default 9999
export CAMERA_POSE_BIND_ADDR=0.0.0.0   # default 0.0.0.0 (loopback 도 받음)
./build/g1_ctrl --log
```

### 17-7. 검증

트래커 종료 시 `[multicam-v2] v2 UDP packets sent: <N>` (v2 mode) 또는 `[multicam] UDP packets sent: <N>` (v1 mode), g1_ctrl 종료 시 `[cam]` 로그의 `recv_count` / `v2_count` / `v1_count`. 두 값이 비슷하면 손실 거의 없음. PC-only 모드에선 1:1 가까이 나와야 정상 (loopback).

UDP 도착 단독 확인:
```bash
# 터미널 1 (트래커는 끄고)
sudo tcpdump -i lo -A udp port 9999 -c 5
# 터미널 2 에서 트래커 띄우면 packet stream 이 보여야 함
```

