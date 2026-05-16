# AprilTag 셋업 튜토리얼 (카메라 2대 + 기준 태그 원점 방식)

이 문서는 `sim2real_updated.md`의 비전/태그 파트를 분리한 운영 가이드다.  
목표는 다음 순서로 **앞부분을 확실히 고정**하는 것이다.

1. 카메라 캘리브레이션(내/외부 파라미터) 완료  
2. 카메라 2대 인식/해상도/FPS 검증 완료  
3. 바닥 기준 태그를 world 원점으로 두고 다른 태그 상대 pose 추정  
4. 스무딩 전 시각화로 x/y/z 동작 확인  
5. 이후에 필터(이동평균/저역통과/칼만) 적용 여부 결정  

---

## Part A) Object Pose / AprilTag 셋업 (원점 기준 좌표계)

## 0) 시작 전에: AprilTag vs ArUco

둘 다 좋은 선택이고, 핵심은 "현재 파이프라인과 얼마나 잘 맞는가"다.

- AprilTag 장점
  - 일반적으로 원거리/사선에서 robust 하다는 평가가 많음
  - 현재 저장소가 `pupil_apriltags` + AprilTag 기준으로 이미 구성됨
  - `track_robot_and_box.py`, `detect_apriltag_two_cams_world.py` 등 기존 스크립트 재사용 가능
- AprilTag 단점
  - OpenCV 기본 내장 ArUco 대비 별도 패키지 의존성이 생김
  - 딕셔너리/태그 패밀리 관리가 익숙하지 않으면 초반 헷갈릴 수 있음
- ArUco 장점
  - OpenCV 생태계에 바로 붙음 (`cv2.aruco`)
  - Charuco 보드 기반 캘리브레이션/검증 자료가 풍부
- ArUco 단점
  - 환경/거리/블러 조건에 따라 검출 안정성이 AprilTag보다 떨어지는 케이스가 있음

권장: **지금은 AprilTag로 끝까지** 가고, 이후 필요 시 ArUco를 A/B 테스트로 비교한다.

---

## 1) 카메라 배치 전략 (z 튐 대응 포함)

질문 주신 것처럼 천장 1대만 쓰면 z축이 튀는 경우가 자주 있다(특히 가림/사선각/픽셀 해상도 한계).

- 카메라 1 (월드 기준 카메라)
  - 천장 또는 높은 위치에서 작업영역 전체를 보게 설치
  - 바닥 기준 태그(원점 태그)가 항상 보이도록 구성
- 카메라 2 (보조 카메라)
  - 천장 외에 **약간 사선 위(45도 전후)**에서 동일 영역 관측
  - z 안정화와 가림(occlusion) 완화 목적
- 깊이 가능한 카메라 활용
  - depth를 바로 pose로 쓰기보다, **검출 신뢰도 보조/거리 게이트**에 활용하면 실전에서 유리
  - 예: 태그까지 거리 급변 프레임 reject, depth 이상치 프레임 무효화

핵심 원칙: 단일 카메라 정밀도보다 **다중 시점 + 품질 게이팅**이 튐 억제에 더 효과적이다.

---

## 2) 환경 준비

```bash
conda activate unitree_rl_mjlab
cd /home/roy/realsense_calib
pip install opencv-python pyrealsense2 pupil-apriltags numpy
```

---

## 3) 터미널에서 카메라 2대 인식 확인

### 3-1. 장치 인식

```bash
lsusb | rg -i "intel|realsense"
rs-enumerate-devices
```

### 3-2. 카메라별 지원 해상도/FPS 확인

```bash
rs-enumerate-devices
```

여기서 각 시리얼별로 color/depth stream profile(예: 640x480@30, 1280x720@30 등)을 확인한다.  
참고: `-s`는 short summary라서 해상도/FPS가 생략된다.

#### 우리 카메라 스택 (현재 장비 기준)

- Camera A: `Intel RealSense D435`
  - Serial: `115222071236`
  - Firmware: `5.16.0.1` (권장: `5.17.0.10`)
  - IMU: 없음 (`IMU_Unknown`)
- Camera B: `Intel RealSense D435i`
  - Serial: `935322072654`
  - Firmware: `5.16.0.1` (권장: `5.17.0.10`)
  - IMU: 있음 (`BMI055`)

운영에서 중요하게 볼 프로파일(두 카메라 공통):

- Color: `640x480` / `848x480` / `960x540` @ 최대 `60 Hz`
- Color: `1280x720` / `1920x1080` @ 최대 `30 Hz`
- Depth: `640x480`, `848x480` @ 최대 `90 Hz`
- Depth: `1280x720` @ 최대 `30 Hz`

AprilTag 운영 권장 (policy 50 Hz 기준):

- 전제: policy/제어가 `50 Hz`이면, 비전 업데이트도 가능하면 `>=50 Hz`가 유리
- 시작점: `640x480 @ 60` (`BGR8`)로 먼저 테스트
- 확인 항목: 실제 검출 FPS, CPU 점유율, 포즈 지터(x/y/z 표준편차)
- 안정적이면: `640x480 @ 60` 유지
- 정확도 부족하면: `848x480 @ 60` 시도
- 연산이 버거우면: `848x480 @ 30` 또는 `640x480 @ 30`로 타협

운영 전략 요약:

- 동기성 우선: 50Hz 제어와 가까운 측정 업데이트 확보 (`>=50 Hz` 권장)
- 안정성 우선: 검출 FPS 저하/CPU 병목이 보이면 해상도 또는 FPS를 낮춰 실시간성 유지

실험 메모(발표용):

- `fps 30` 대비 `fps 60`에서 태그 인식 끊김이 훨씬 적었고, 빠른 움직임에서도 추적 연속성이 더 좋았다.
- 우리 policy 제어 주기가 `50 Hz`이므로, 비전 업데이트는 `50 Hz`보다 큰 설정을 우선 사용한다.
- 현재 운영 1순위 테스트 설정: `960x540 @ 60` (환경/부하에 따라 `848x480 @ 60`도 사용 가능).

### 3-3. Extrinsic 전에 해야 할 사전 점검 (중요)

`python calibrate_extrinsic_two_cams.py`를 돌리기 전에, 먼저 "각 카메라가 실험 위치에서 태그를 안정적으로 보는지"를 확인한다.

태그 크기 설정은 커맨드 문자열 대신 JSON 파일로 관리한다:

```bash
cat config/tag_sizes.json
```

현재 기준:

- 기본 태그 크기: `0.077 m`
- ID `0`, `1`: `0.145 m`

1) 카메라2(월드 후보, D435 `115222071236`) 단독 확인

```bash
cd /home/roy/realsense_calib
python detect_apriltag_with_axes.py \
  --serial 115222071236 \
  --calib camera2_115222071236_calibration.npz \
  --tag-config config/tag_sizes.json \
  --width 640 --height 480 --fps 60
```

2) 카메라1(D435i `935322072654`) 단독 확인

```bash
cd /home/roy/realsense_calib
python detect_apriltag_with_axes.py \
  --serial 935322072654 \
  --calib camera1_935322072654_calibration.npz \
  --tag-config config/tag_sizes.json \
  --width 640 --height 480 --fps 60
```

설명:

- 지금은 `detect_apriltag_with_axes.py`가 인자를 받아서 카메라별로 재사용 가능하다
- 즉, 스크립트를 따로 만들 필요 없이 `--serial`, `--calib`만 바꿔 같은 체크를 수행하면 된다
- 태그별 실제 크기는 `config/tag_sizes.json`에서 관리한다
- 필요하면 임시로 `--tag-size-map`을 추가해 CLI에서 덮어쓸 수 있다

3) 바닥 원점 태그 인식 확인 (설정 전에 먼저)

```bash
cd /home/roy/realsense_calib
python detect_apriltag_two_cams.py \
  --tag-config config/tag_sizes.json \
  --width 640 --height 480 --fps 60
```

체크 기준:

- 바닥 원점으로 쓸 태그 ID가 최소 한 카메라에서 계속 안정 검출되는가
- 두 카메라를 함께 볼 때도 원점 태그가 시야 가장자리에서 자주 잘리지 않는가
- 사람/물체가 잠깐 가려도 재검출이 빠르게 복귀하는가

이 단계 통과 후에:

```bash
python calibrate_extrinsic_two_cams.py \
  --tag-id 1 \
  --tag-config config/tag_sizes.json \
  --num-samples 100 \
  --width 960 --height 540 --fps 60
```

---

## 4) 카메라 캘리브레이션 (intrinsic/extrinsic)

운영 원칙(중요):

- 캘리브레이션은 **운영할 해상도/FPS를 먼저 고정**하고, 그 설정으로 수행한다.
- 현재 운영 기준: `960x540 @ 60`
- 이유:
  - intrinsic은 해상도/리사이즈 조건에 민감함
  - extrinsic은 이론상 물리 변환이지만 실제 추정은 픽셀 검출 품질에 의존함
  - 따라서 운영 설정과 캘리브 설정이 다르면 z축/스케일 오차가 커질 수 있음

권장 순서:

1. cam1/cam2 체커보드 수집 (`960x540@60`)  
2. cam1/cam2 intrinsic 계산  
3. 같은 설정으로 cam1->cam2 extrinsic 계산

```bash
cd /home/roy/realsense_calib

# (1) 체커보드 수집 - cam1 (D435i)
python capture_checkerboard.py \
  --serial 935322072654 \
  --save-dir checker_images_cam1_960 \
  --width 960 --height 540 --fps 60

# (1) 체커보드 수집 - cam2 (D435)
python capture_checkerboard.py \
  --serial 115222071236 \
  --save-dir checker_images_cam2_960 \
  --width 960 --height 540 --fps 60

# (2) 카메라별 intrinsic - cam1
python calibrate_camera.py \
  --images-glob "checker_images_cam1_960/*.png" \
  --output camera1_935322072654_calibration.npz \
  --serial 935322072654 \
  --checkerboard-cols 7 --checkerboard-rows 10 --square-size 0.025

# (2) 카메라별 intrinsic - cam2
python calibrate_camera.py \
  --images-glob "checker_images_cam2_960/*.png" \
  --output camera2_115222071236_calibration.npz \
  --serial 115222071236 \
  --checkerboard-cols 7 --checkerboard-rows 10 --square-size 0.025

# (3) 카메라 간 extrinsic
python calibrate_extrinsic_two_cams.py \
  --tag-id 1 \
  --tag-config config/tag_sizes.json \
  --num-samples 200 \
  --width 960 --height 540 --fps 60
```

`calibrate_extrinsic_two_cams.py`가 하는 일:

- 목적: 두 카메라 좌표계를 하나로 맞추는 고정 변환 `T_c2_c1` 계산
- 의미: camera1에서 본 포즈를 camera2(world) 좌표계로 옮길 수 있게 함
- 방법: 두 카메라가 같은 AprilTag를 동시에 볼 때
  - `T_c1_tag`, `T_c2_tag`를 각각 추정
  - `T_c2_c1 = T_c2_tag @ inv(T_c1_tag)`를 샘플마다 계산
  - 여러 샘플 평균 + 회전행렬 재직교화(SVD) 후 저장
- 산출물: `camera1_to_camera2_extrinsic.npz` (키: `T_c2_c1`, 시리얼, 샘플 수 등)
- 주의: extrinsic 계산 시 스트림 해상도/FPS(`--width --height --fps`)를 실제 운영과 맞춰서 실행

왜 필요한가:

- camera2를 world로 둘 때, camera1 검출 결과를 world로 통일해 합칠 수 있음
- 2대 카메라 융합(가림 완화, z 안정화)의 필수 전제
- `detect_apriltag_two_cams_world.py`가 이 파일을 읽어 월드 변환에 사용

언제 다시 해야 하나:

- 카메라 위치/각도/삼각대가 조금이라도 바뀐 경우
- 렌즈/마운트 체결이 바뀐 경우
- 같은 태그를 볼 때 두 카메라 world 좌표가 지속적으로 어긋나는 경우

산출물 예:

- `camera1_*_calibration.npz`
- `camera2_*_calibration.npz`
- `camera1_to_camera2_extrinsic.npz`

검증:

```bash
python detect_apriltag_two_cams_world.py
```

이 스크립트에서 camera2를 world로 두고 camera1 검출 결과를 world로 변환해 비교한다.

---

## 5) 바닥 태그를 world 원점으로 두는 방법

요구하신 방식(바닥 태그 1개를 원점으로 고정)은 매우 실용적이다.

바닥을 origin으로 쓰는 이유(발표용):

- 실험 공간의 절대 기준점을 고정해, 카메라 위치가 조금 바뀌어도 좌표계 해석이 일관된다.
- 로봇/물체/태그의 상대 위치를 동일 기준에서 비교할 수 있어, sim2real 정렬 설명이 쉬워진다.
- 특히 다중 카메라 융합 시에도 최종 결과를 하나의 바닥 기준 좌표계로 표현할 수 있다.

관찰 메모(발표용):

- 바닥 origin 설정 후 단일 카메라만 사용해도, 현재 환경에서는 `z`축 오차가 생각보다 나쁘지 않음을 확인했다.
- 다만 가림/사선/빠른 동작에서는 듀얼 카메라 융합이 여전히 더 안정적이므로 운영 기본은 2카메라로 유지한다.

### 5-1. 좌표계 정의

- 원점 태그 ID를 예: `TAG_ORIGIN = 100`으로 지정
- `world` 좌표계 = 원점 태그 좌표계
  - x/y 방향은 태그 부착 방향에 맞춰 물리적으로 정함
  - z는 바닥 법선 위 방향으로 맞춤(태그 인쇄 방향 주의)

### 5-2. 프레임별 상대 pose

카메라에서 읽은 pose를 `T_cam_tag`라 할 때:

- `T_world_tag_i = inv(T_cam_tag_origin) @ T_cam_tag_i`

즉, origin 태그가 보이는 프레임에서는 다른 모든 태그 위치를 origin 기준 상대좌표로 바로 계산할 수 있다.

중요한 맥락(카메라 2대 사용 시):

- 최종 원점이 바닥 태그여도, camera1 데이터를 함께 쓰려면 먼저 camera1을 camera2/world로 옮겨야 함
- 변환 체인 예:
  - `T_c2_tag = T_c2_c1 @ T_c1_tag`  (camera1 -> camera2/world)
  - `T_floor_tag_i = inv(T_c2_tag_origin) @ T_c2_tag_i`  (world -> 바닥 원점)
- 그래서 `calibrate_extrinsic_two_cams.py`의 `T_c2_c1`이 필요하다
- 요약하면: **1 -> 2(world) -> 바닥원점** 체인으로 좌표계를 통일해야 듀얼 카메라 융합이 일관된다

### 5-3. ID 0 원점 기준 시각화 (선 + 상대 xyz)

가능하다. `detect_apriltag_with_axes.py`에 `--origin-id`를 주면,

- ID 0을 원점으로 사용
- 원점 태그 중심에서 다른 태그 중심으로 선(하늘색) 표시
- 화면 왼쪽에 `rel[0->N] x,y,z`를 실시간 출력

카메라2(월드 후보) 기준 예시:

```bash
cd /home/roy/realsense_calib
python detect_apriltag_with_axes.py \
  --serial 115222071236 \
  --calib camera2_115222071236_calibration.npz \
  --tag-config config/tag_sizes.json \
  --origin-id 0 \
  --width 640 --height 480 --fps 60
```

카메라1도 동일하게 `--serial`, `--calib`만 교체하면 된다.

헷갈림을 줄이려면, 원점 전용 스크립트를 사용한다:

```bash
cd /home/roy/realsense_calib
python detect_apriltag_with_origin_coords.py \
  --serial 115222071236 \
  --calib camera2_115222071236_calibration.npz \
  --tag-config config/tag_sizes.json \
  --origin-id 1 \
  --width 640 --height 480 --fps 60
```

이 스크립트는 각 태그 아래 좌표를 전부 origin 기준으로만 표시한다.
(origin 태그는 항상 `rel: [0, 0, 0]`)

### 5-4. 축 정의 (AprilTag / 카메라 좌표계)

`pupil_apriltags` 기준으로 태그 좌표계는 다음이다.

- 태그 `+X`: 태그 오른쪽 방향
- 태그 `+Y`: 태그 아래 방향
- 태그 `+Z`: 태그 면의 법선(태그에서 카메라 쪽으로 나오는 방향)

카메라(OpenCV) 좌표계는 보통 다음 기준으로 해석한다.

- 카메라 `+X`: 이미지 오른쪽
- 카메라 `+Y`: 이미지 아래
- 카메라 `+Z`: 카메라 앞 방향

그래서 스크립트 오버레이에서 색은:

- 빨강 = X, 초록 = Y, 파랑 = Z

즉 `--origin-id 0`일 때 보이는 `rel[0->N]`은
`inv(T_cam_tag0) @ T_cam_tagN`의 평행이동 성분이며, "태그0 기준으로 태그N이 어디에 있는지"를 뜻한다.

### 5-5. 실전 팁

- origin 태그는 가능하면 크게 인쇄하고, 항상 시야에 들어오게 배치
- origin 미검출 시 이전 world 정합을 짧게 hold하거나, 보조 카메라로 대체
- 바닥 반사/광택이 있으면 검출 튐이 커지므로 매트 처리 권장

### 5-6. Section 5 -> 6 사이 필수 확인 (듀얼카메라 + 원점 변환 + 융합)

Step 6(로봇/박스 추정 파이프라인)로 넘어가기 전에 아래를 먼저 확인한다.

1) camera2를 world 기준으로 고정  
2) camera1 검출을 `T_c2_c1`로 camera2 좌표계로 변환  
3) camera1/2 결과를 품질(decision margin) 가중으로 융합  
4) 마지막에 origin 기준으로 변환

식:

- `T_c2_tag_from_cam1 = T_c2_c1 @ T_c1_tag`
- `T_origin_tag_i = inv(T_c2_tag_origin) @ T_c2_tag_i`

실행 커맨드 (ID 1을 원점으로):

```bash
cd /home/roy/realsense_calib
python detect_apriltag_two_cams_origin_fusion.py \
  --cam1-serial 935322072654 \
  --cam2-serial 115222071236 \
  --cam1-calib camera1_935322072654_calibration.npz \
  --cam2-calib camera2_115222071236_calibration.npz \
  --extrinsic camera1_to_camera2_extrinsic.npz \
  --tag-config config/tag_sizes.json \
  --origin-id 1 \
  --width 960 --height 540 --fps 60
```

시각화 창 3개:

- `Camera1 -> transformed to C2`: camera1 검출을 C2(world)로 변환한 뒤 origin 상대좌표 표시
- `Camera2 (C2/world)`: camera2 직접 검출 기반 origin 상대좌표 표시
- `Fused Origin Coordinates (C2/world)`: 융합 결과 기준 상대좌표 표시

이 단계의 합격 기준:

- 두 카메라 창에서 같은 태그의 origin 상대좌표가 큰 틀에서 일치
- 융합 창 좌표가 단일 카메라보다 덜 튐
- origin 태그 가림 상황에서도 최소 한 카메라가 유지되면 추정이 빠르게 복귀

---

## 6) 로봇 torso / 박스 pose 추정 파이프라인

현재 저장소 기준 핵심 파일:

- `calibrate_head_tag.py`: head tag -> torso 고정변환 산출
- `register_box_tag_map.py`: 박스 다중 태그 맵 산출
- `track_robot_and_box.py`: 실시간 `T_world_torso`, `T_world_box`, 상대 pose 출력
- `validate_head_to_torso.py`: head->torso 변환 검증

실행:

```bash
cd /home/roy/realsense_calib
python calibrate_head_tag.py
python register_box_tag_map.py
python validate_head_to_torso.py
python track_robot_and_box.py
```

`track_robot_and_box.py`는 화면 오버레이 + 콘솔 출력으로 상대좌표를 바로 보여주므로,
스무딩 전 시각 점검에 바로 쓸 수 있다.

---

## Part B) Robot 좌표/레퍼런스 모션 확인 (OmniRetarget global frame 정합)

Part A에서 object pose/원점 좌표계를 고정했다면, 이제 OmniRetarget reference에서
로봇 좌표와 물체 좌표를 함께 확인해 좌표계 해석을 맞춘다.

### B-1. sub8_largebox_045 reference replay (robot + object)

```bash
cd /home/roy/realsense_calib/humanoid_project
python scripts/play.py replay sub8_largebox_045_original
```

재생 조작키:

- `Space`: pause / resume
- `Right` 또는 `N`: 다음 프레임 1칸 (pause 상태에서)
- `Left` 또는 `B`: 이전 프레임 1칸 (pause 상태에서)

중요:

- object가 안 보이면 replay 모드가 맞는지 먼저 확인 (`scripts/play.py replay ...`)
- MuJoCo 왼쪽 아래 `Group enable`에서 아래를 켠다
  - group 4: object visual
  - group 5: object collision

### B-2. MuJoCo GUI에서 로봇/물체 좌표 보기

핵심 정리(오해 방지):

- `qpos`와 `xpos` 모두 **world(global) 기준** 값이다.
- 즉 `xpos`가 pelvis 기준(local)인 것은 아니다.
- pelvis 기준 좌표가 필요하면 `inv(T_world_pelvis) @ T_world_body`처럼 직접 변환해야 한다.

`Watch` 패널에서 `Field`/`Index`를 바꿔 아래 값을 확인한다.

위치:

- floating base(로봇 root freejoint) 위치:
  - x: `0`, y: `1`, z: `2`
- object_joint 위치:
  - x: `36`, y: `37`, z: `38`

자세(orientation):

- floating base quaternion (`w,x,y,z`):
  - `qpos[3:7]`
- object quaternion (`w,x,y,z`):
  - `qpos[39:43]`
- body world quaternion을 보려면 `Field = xquat` + body id 기반 인덱스를 사용

실제 확인한 body id (`xpos` 참고용):

- `pelvis` body id = `1` (xpos 시작 인덱스 `3`)
- `torso_link` body id = `16` (xpos 시작 인덱스 `48`)
- `object` body id = `31` (xpos 시작 인덱스 `93`)

참고:

- `qpos`는 조인트 상태(루트/객체 free joint 포함) 확인에 직관적
- `xpos`는 body world position이며 body id 기반 인덱스로 확인
- quaternion은 항상 정규화된 회전값(`w,x,y,z`)으로 해석하고, 부호가 동시에 뒤집혀도 같은 회전을 의미할 수 있다

축(axes) 시각화:

- MuJoCo 왼쪽 패널 `Visualization`에서 frame 관련 토글(body/geom frame)을 켜면 축을 볼 수 있다.
- replay에서 프레임 멈춤(`Space`) 후 `Right/Left`로 한 프레임씩 넘기면서 축과 수치를 같이 확인하면 해석이 쉽다.

### B-3. 왜 이 단계가 필요한가

- OmniRetarget `processed/*.npz`의 `body_pos_w`, `object_pos_w`는 global/world 기준 값이다.
- 따라서 시각화에서 로봇과 물체를 같은 global frame으로 동시에 확인해야,
  이후 실험실 origin 좌표계와 정합할 때 축/부호/오프셋 오류를 줄일 수 있다.

---

## 7) 스무딩 전에 먼저 해야 할 시각화 테스트

필터 적용 전 반드시 "원본 신호 상태"를 본다.

권장 테스트:

- 정지 테스트(10~20초)
  - 태그/로봇/박스를 고정하고 x,y,z 드리프트를 측정
- 직선 이동 테스트
  - x만 바뀌게 움직여서 y,z cross-coupling 확인
- 회전 테스트
  - yaw 회전 시 z가 같이 흔들리는지 확인
- 가림 테스트
  - 사람/물체로 부분 가림했을 때 재검출 복귀 시간 확인

통과 기준 예시(초기):

- 정지 상태 z 표준편차가 요구 정밀도 이내
- 미검출 이후 0.5~1.0초 내 복귀
- 카메라 1/2 추정치 차이가 허용범위 이내

---

## 8) 스무딩 기법 정리 (이동평균 / 저역통과 / 칼만)

### 8-1. 이동평균 (Moving Average)

- 장점: 가장 단순, 구현 쉬움
- 단점: 지연(latency) 증가, 급격한 동작에서 둔해짐
- 추천: 초기 노이즈 레벨 파악용

### 8-2. 저역통과 필터 (Low-pass, EMA)

- 장점: 계산 가벼움, 튜닝 직관적(alpha)
- 단점: alpha 설정에 따라 지연-노이즈 tradeoff 큼
- 추천: 실시간 제어 직전의 1차 안정화

### 8-3. 칼만 필터 (Kalman)

- 장점: 위치/속도 모델 기반으로 dropout 대응에 강함
- 단점: 모델/노이즈 공분산 튜닝 필요
- 추천: 미검출/가림/다중카메라 융합까지 갈 때

실무 순서:

1. 무필터 baseline 기록  
2. EMA부터 적용  
3. 필요 시 칼만으로 업그레이드

---

## 9) 카메라 기준 마커 상대위치 라이브러리 정리

현재 파이프라인에 바로 쓸 수 있는 선택지:

- `pupil_apriltags`
  - AprilTag ID + pose(`pose_R`, `pose_t`)를 바로 얻음
  - 현재 저장소와 가장 잘 맞음
- OpenCV `cv2.aruco`
  - ArUco 마커/보드 기반 pose 추정 (`estimatePoseSingleMarkers` 계열)
  - ArUco 전환 시 유효
- ROS 계열 (`apriltag_ros`)
  - ROS2 사용 시 토픽 기반 좌표계 관리가 편함
  - 지금 구조가 ROS 비의존이라면 우선순위는 낮음

지금 단계에서는 **`pupil_apriltags` 유지**가 가장 빠르고 안전하다.

---

## 10) FOV/노출/셔터 체크 (놓치기 쉬운 항목)

화각(FOV) 자체를 "설정값"으로 바꾸기보다는, 실제 캘리브레이션과 해상도 선택으로 간접 관리한다.

- 해상도/FPS를 올리면 검출 정밀도는 좋아지나 지연/연산량 증가
- 자동 노출이 심하게 흔들리면 ID 튐 증가 -> 가능하면 고정 노출/게인 테스트
- 모션 블러가 크면 셔터/조명 개선 필요
- 렌즈 왜곡이 큰 FoV에서는 intrinsic 품질이 낮으면 z 튐이 커짐

체크 기준:

- 운영 해상도/FPS로 **다시 캘리브레이션**했는가
- 운영 조명에서 태그 코너가 선명한가
- 천장/사선 카메라 모두에서 origin 태그가 안정 검출되는가

---

## 11) 이번 단계 완료 게이트 (다음 단계로 넘어가기 전)

- [ ] 카메라 2대 시리얼/스트림 프로파일 문서화 완료
- [ ] intrinsic/extrinsic 산출물 재생성 및 검증 완료
- [ ] 바닥 origin 태그 기준 상대좌표 계산 확인
- [ ] `track_robot_and_box.py`에서 x,y,z/자세 시각화 정상
- [ ] 무필터 baseline 로그(정지/이동/회전/가림) 확보
- [ ] EMA 또는 칼만 적용 전/후 비교 계획 수립

---

## 12) OmniRetarget 최종 목표와의 연결 (앞부분 완료 후 진행)

최종 목표가 OmniRetarget policy 탑재인 만큼, 앞단(비전)이 끝나면 아래를 순서대로 이어간다.

1. OmniRetarget 데이터의 박스 기준점이 **정중앙**인지 재확인  
2. 박스 orientation(쿼터니언/축정의)와 실험실 태그 프레임 일치 검증  
3. head tag -> torso 변환(`T_tag_torso`) 품질 재검증  
4. `T_world_torso`, `T_world_object`, `T_torso_object` 로그를 학습/배포 관측 정의와 대조  

특히 "정중앙 + orientation + torso 기준" 3개가 동시에 맞아야, 나중에 정책이 붙었을 때 틀어짐을 줄일 수 있다.
