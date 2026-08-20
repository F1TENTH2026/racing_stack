# 실차 레이스 빠른 명령어

## 1. 정적 장애물 레이스 실행

`<맵_이름>`을 실제로 사용할 맵 디렉터리 이름으로 바꾼다.

```bash
ros2 launch stack_master race.launch.xml map:=<맵_이름> race_mode:=static_qualify
```

`sim`의 기본값은 `false`이므로 실차에서는 따로 입력하지 않는다.

## 2. 주행 중 전체 속도 배율 변경

아래 숫자는 `speed_scaling.yaml`에 설정된 모든 섹터 배율에 다시 곱해지는
전역 배율이다. 예를 들어 섹터 값 `1.2`에 `0.8`을 적용하면 최종 배율은
`0.96`이다.

```bash
ros2 param set /speed_sector_tuner phase_multiplier 0.8
```

원래 섹터 설정으로 복구:

```bash
ros2 param set /speed_sector_tuner phase_multiplier 1.0
```
