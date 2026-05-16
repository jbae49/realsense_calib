# AprilTag → torso_link 변환 가이드

## 전체 그림

```
시뮬레이션:  simulator → torso_link pose (직접 줌)
실제 로봇:   camera → AprilTag(머리) → torso_link pose (변환 필요)

둘 다 동일한 observation:
  obs = torso_pos - box_pos  (상대 위치)
  obs += torso_orientation   (회전)
```

## 필요한 변환

```
T_cam_tag        카메라가 읽은 AprilTag pose (매 프레임)
T_tag_torso      태그 → torso_link 고정 변환 (한 번 세팅)
─────────────────────────────────────────────
T_cam_torso = T_cam_tag @ T_tag_torso
```

## T_tag_torso 구성

```
T_tag_torso = inv(T_torso_tag)
T_torso_tag = T_torso_headlink @ T_headlink_tagtop @ T_tagtop_tag

여기서:
  T_torso_headlink  = URDF head_joint        = [0.004, 0, -0.054] (고정)
  T_headlink_tagtop = 메쉬 꼭대기 좌표       = [0.001, 0, 0.526]  (고정)
  T_tagtop_tag      = 태그 부착 방향 rotation = R_mounting         (측정 필요)
```

## Step-by-step

### Step 1: 태그 부착 방향 파악 (R_mounting)

AprilTag 좌표계 규칙:
- z축: 태그 표면에서 바깥으로 나옴 (법선 방향)
- x축: 태그의 오른쪽
- y축: 태그의 아래쪽

#### Case A: 태그를 위를 향하게 (수평으로 머리 위에 올려놓음)
- 태그 z축 = 위 = robot z축
- 태그 x축 = 로봇 앞 (태그 글자가 앞을 향하게)
→ R_mounting = Rx(-90°) (카메라 convention에 맞게 조정 필요)

#### Case B: 태그를 앞을 향하게 (이마에 붙임)
- 태그 z축 = 앞 = robot x축
→ R_mounting = I 또는 Ry(90°)

### Step 2: 캘리브 스크립트로 검증 (calibrate_head_tag.py)

### Step 3: 런타임에서 사용 (track_robot_and_box.py)
