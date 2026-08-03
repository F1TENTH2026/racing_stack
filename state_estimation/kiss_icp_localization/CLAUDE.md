# CLAUDE.md

KISS-ICP 스타일 map-scan 정합 localization (Livox MID360 + 내장 IMU).
fast_livo localization과 동일 인터페이스(`/livox/lidar`, `/livox/imu`, `/initialpose`)의 경량 대체재.

## Build & Test

```bash
colcon build --packages-select kiss_icp_localization
colcon test  --packages-select kiss_icp_localization   # core gtest (합성 방 정합 수렴)

# ROS 레벨 smoke test (합성 맵 + 시뮬 스캔/IMU, ~10s)
python3 test/smoke_test.py

# 실데이터 검증 (bag 47s + hall_0609 맵, 실시간 재생 ~60s)
python3 test/bag_validation.py [bag_dir] [map_pcd]
```

테스트 스크립트는 노드 바이너리를 직접 exec한다 — `ros2 run`은 래퍼라
terminate해도 노드가 살아남아 다음 실행을 오염시킴 (토픽 이중 발행).

## Run

```bash
ros2 launch kiss_icp_localization localization.launch.py map:=hall_0609
```

`map:=`은 maps root 아래 디렉토리명 (`<root>/<map>/cloudGlobal.pcd` 로드).
root는 `KISS_LOC_MAPS_ROOT` → `FASTLIVO_MAPS_ROOT` → 리포 내 stack_master/maps 순.

출력: `/kiss_loc/odometry` (스캔 보정 + IMU rate 전파, 단일 토픽),
`/kiss_loc/scan_aligned`, `/kiss_loc/map` (transient_local), TF `map→base_link`.

## Architecture

- **`VoxelHashMap`** — prior map의 정적 복셀 해시 (복셀당 점 수 상한, 점+normal).
  NN 탐색은 KISS-ICP처럼 인접 27복셀 고정 — 반경을 늘리면 비용 폭증 (125복셀로
  실시간 불가했음). `EstimateNormals()`: 복셀+인접 PCA로 normal 추정
  (fast_livo save_map은 normal 필드를 0으로 씀 — 직접 계산해야 함)
- **`AlignScanToMap`** — robust ICP (Gauss-Newton + Geman-McClure), normal 있으면
  point-to-plane(`J_w = pw×n`, 부호 주의), 없으면 point-to-point. OpenMP 병렬.
  ridge(1e-6) + step clamp로 퇴화 기하 폭주 방지. 3D 모드(`localization_2d=false`) 정합.
- **`AlignScanToTrackSdf`** — 2D 모드(`localization_2d=true`) 정합. 지면밴드 슬랩을 2D로
  평탄화 후 `TrackMask` SDF(벽=zero-level)에 scan-to-SDF. 점 residual=SDF값, Jacobian=∇SDF.
  **3-DoF(x,y,yaw)만** 추정하고 z/roll/pitch는 예측에서 유지 → SE(3) 반환이라 다운스트림
  (발산게이트·속도·전파) 무변경. Geman-McClure + ridge(1e-6) + step clamp(기존과 동일).
  2D 모드는 map_2p5d.pcd 불필요(매칭에 안 씀). 정합 맵은 **`loc_2d_map_yaml`**(전용
  yaml+pgm, >127=주행영역·경계=벽) — 비우면 `track_map_yaml`로 폴백. detection용
  track_mask와 분리된 별도 TrackMask(`loc_mask_`)로 로드. roll/pitch는 ground_lidar.yaml
  지면법선으로 만든 `R_level_`(마운트 틸트)로 고정해 스캔을 수평 투영.
- **`AdaptiveThreshold`** — KISS-ICP 적응 임계: 모델(예측) 오차 RMS가 탐색 반경
- **`LocalizationNode`** — IMU gyro만 사용 (가속도계 미사용): deskew 회전,
  스캔 간 예측 회전, IMU rate 전파. 병진은 constant-velocity 모델 + EMA
- **`BevDetector`** — per-frame BEV 검출. z밴드 crop(법선방향) 후 **2단계 map
  subtraction**: ① prior map(2.5d) 근접 점 제거 `GetClosestNeighbor` ② `TrackMask`
  밖(법선수직/in-plane) 점 제거. DBSCAN 클러스터 + centroid 트래킹으로 moving 판정.
- **`TrackMask`** (header-only) — GLIM `map_track.{pgm,yaml}`(점유격자 free 영역 =
  주행 트랙) 로드 + Euclidean distance transform(2-pass) 사전계산. `DistOutsideTrack`
  으로 track 밖 수평거리 반환(>margin이면 제거). 트랙 밖 오탐의 주 원인 차단.

### GLIM 맵 산출물 (glim_map_pipeline.py)

`map.pcd`(전체 3D) / `map_2p5d.pcd`(지면제거+z밴드 = localization·stage1이 로드) /
`map_2d.*`(nav 점유격자) / `map_track.*`(2D 트랙마스크 = free 영역, stage2 필터용).
트랙마스크는 **벽이 닫혀야** 레인만 잡힘(flood-fill) — 개방환경이면 밖으로 샘.

### 시간축 (중요)

livox 드라이버 `use_system_timestamp=true`면 **header가 100ms 누적 '끝' 시각**
(publish 시 now()). per-point timestamp는 디바이스 uptime이라 header와 직접 비교
불가 → 첫 점 기준 상대화 후 `stamp_at_scan_end`(기본 true)로 윈도우를 header 뒤로
앵커. 이거 틀리면 deskew/예측이 0.1s 어긋나고 IMU 커버리지 대기로 지연 +0.1s.

### Localization 안전장치 (KISS-ICP 원본에 없는 부분)

- **발산 게이트**: 정합 결과가 예측에서 `reject_trans`(2m)/`reject_rot_deg`(30°)
  이상 점프하면 기각하고 예측으로 coast. 단 `reject_recover_count`(3) 연속 기각이면
  예측이 틀린 것으로 보고 정합 결과로 재앵커 (영구 coast 발산 방지)
- **속도 상한 + 가속도 상한** (`max_velocity`, `max_accel`): 복도 퇴화 구간에서
  종방향 미관측 → 속도 추정 폭주 → 예측 overshoot 피드백을 물리 한계로 차단
  (bag에서 휠 실측 6.1 m/s인데 추정 13.5 m/s까지 부풀었던 사례)

### 검증 기준치 (bag_validation.py, hall_0609 + 2026-06-10 bag)

lock fraction 100%, scan-to-map 잔차 mean 6.5cm (맵 밀도/센서 노이즈 바닥),
ICP p50 23ms / p95 47ms, 수렴률 80%, max |v| 5.9 m/s (휠 실측 6.10), 추적 중
최대 보정 스냅 ~0.45m. 초기 포즈 수렴 스냅(첫 1s)은 점프 기준에서 제외.
`print_stats:=true`면 STAT 라인(key=value)이 찍히고 스크립트가 파싱해 통계 출력.
추가 파라미터는 CLI로 오버라이드 가능: `python3 test/bag_validation.py BAG MAP k:=v ...`

### 튜닝에서 배운 것 (되돌리지 말 것)

- `convergence_eps` 1e-5는 NN 재할당 진동 때문에 절대 수렴 안 함 → 항상
  max_iterations를 다 돌았음. 2e-3로 수렴률 5.6%→80%, ICP 33→23ms, 잔차 동일
- `adaptive_range`를 실내 스케일(15)로 줄이면 임계가 조여져 공격 주행에서
  lock 100%→83% 붕괴. KISS-ICP가 max_range를 쓰는 건 의도된 견고성 마진
- scan_voxel 0.25 + map 50pts/voxel은 잔차 6.5→6.1cm에 ICP 2.5배 — 손해.
  잔차는 5-6cm가 측정 바닥이라 해상도로는 더 안 내려감
- deskew 끄면 잔차 +25%, 점프 +35% — gyro 통합 기여 실측 확인됨
- 2D scan-to-SDF 주의: ① 트랙이 벽으로 **밀폐**돼야 SDF 경계가 안 샘(map_track flood-fill
  전제와 동일) ② 직선 복도는 종방향 미관측(벽 ∇SDF가 횡방향뿐) → 기존 ridge·속도/가속 상한·
  발산게이트로 완화 ③ 초기포즈가 벽에서 너무 멀면 수렴 basin 밖 → `/initialpose` 재앵커
  ④ 2D 모드 STAT 잔차는 평균 |SDF|(점-벽 거리)라 voxel 잔차와 의미가 다름
- 2D 진동(jitter): 소수 점(~67)+이산 SDF로 scan별 yaw가 ~1-3° 떨림. **완전 덮어쓰기**
  (`T_=result.pose`)가 매끄러운 gyro를 버려 진동. → `loc_2d_corr_gain`(complementary 필터,
  발산게이트 뒤에서 `blendPlanar(T_pred,fix,gain)`). **0.3 권장**(진동 ~절반↓, conv 0.97 유지),
  ≤0.15는 lag로 역효과. 주의: 점 수 늘리면(scan_voxel↓) 비벽 점이 끼어 yaw 진동 **악화**,
  vel_smoothing↑(속도 감쇠)도 예측 lag로 **악화** — 진동은 속도가 아니라 절대포즈를 저역통과해야 함.

## 주의

- `initial_pose` 파라미터는 맵 원점 출발 가정. 다른 위치 출발이면 RViz
  `/initialpose`로 재앵커 (z는 현재 추정값 유지)
- 합성 데이터로 ICP 테스트할 때 맵·스캔을 같은 격자에서 샘플링하지 말 것 —
  lattice locking으로 ~1° 바이어스에 갇힘 (실데이터에 없는 병리). 연속 표면
  무작위 샘플링 사용 (test_core.cpp 참조)
