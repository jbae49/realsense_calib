# sub8_45 sim2real deployment guide (G1)

이 문서는 `sub8_largebox_045` 모션 추종을 실제 G1에 올릴 때, 처음부터 끝까지 따라갈 수 있게 정리한 실행 가이드다.

기준 코드/경로:
- 프로젝트 루트: `/home/roy/realsense_calib/humanoid_project`
- 실기 deploy: `/home/roy/realsense_calib/humanoid_project/deploy/robots/g1`
- FSM 설정: `/home/roy/realsense_calib/humanoid_project/deploy/robots/g1/config/config.yaml`
- 현재 버튼 매핑(기본):
  - `LT + up`: `Passive -> FixStand`
  - `RT + A`: `FixStand -> Velocity`
  - `RB + A`: `Velocity -> Mimic_Dance1_subject2`
  - `LT + B`: `Velocity/Mimic -> Passive`

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
cd /home/roy/realsense_calib/humanoid_project/deploy/robots/g1/build
./g1_ctrl --network=<YOUR_ROBOT_NIC>
```

---

## 8) FSM 조작 순서 (실기)

현재 기본 매핑 기준:

1. 초기: `Passive`
2. `LT + up` -> `FixStand`
3. `RT + A` -> `Velocity` (기본 로코모션)
4. `RB + A` -> `Mimic_*` (설정한 mimic state)
5. 언제든 `LT + B` -> `Passive` (소프트 정지)

운영 권장:
- 무조건 `FixStand`에서 2~3초 안정 확인 후 mimic 진입
- 실험자 1명은 게임패드에서 `LT + B`만 담당
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

