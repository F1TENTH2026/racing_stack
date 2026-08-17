#!/usr/bin/env python3
"""race_supervisor — debug:=off 로 띄웠을 때, 조이스틱 RB 로 디버그 출력을 끊는 노드.

왜 노드로 만들었나
------------------
런치 파일은 런타임 이벤트에 반응할 수 없다. ROS 2 launch 의 이벤트 핸들러는
프로세스의 시작/종료만 다루고, "조이스틱 버튼이 눌리면" 같은 조건은 표현할 수
없다. 그래서 /joy 를 구독하는 별도 노드가 필요하다.

동작
----
debug:=on   : 이 노드는 아무 것도 하지 않는다(아예 기동되지 않는다).
debug:=off  : 기동 직후에는 마커가 그대로 나온다 — RViz 로 2D Pose Estimate 를
              여러 번 하고 결과를 눈으로 확인해야 하기 때문이다.
              조이스틱 RB(buttons[5]) 를 누르면 그때 한 번만:

                1. 대상 노드들의 viz_rate_hz 를 0 으로 설정 (파라미터 서비스)
                2. 순수 디버그 노드 종료 (lap_analyser, keyboard_joy)
                3. /debug_enabled 에 false 를 latch 발행
                   - 각 노드가 디버그 로그 파일 쓰기를 멈추는 신호
                   - 랩탑에서 RViz 를 자동으로 닫고 싶으면 이 토픽을 구독하면 된다
                     (RViz 는 랩탑에서 뜨므로 차에서 직접 죽일 수 없다)

              이후로는 되돌리지 않는다(one-way latch). 주행 중에 실수로 디버그가
              다시 켜지는 일이 없어야 하기 때문이다.

버튼 매핑은 simple_mux_node 와 같은 것을 쓴다:
    buttons[4] = LB -> 사람(조이스틱) 주행
    buttons[5] = RB -> 자율주행
LB/RB 로 실제 주행 모드를 바꾸는 것은 simple_mux 의 몫이고, 이 노드는 RB 를
"레이스 시작" 신호로만 읽는다. 둘은 서로를 모른다.
"""
import subprocess

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

# 자율주행 시작 버튼. simple_mux_node._handle_joy 와 같은 인덱스.
RB_BUTTON = 5

# viz_rate_hz 를 0 으로 내릴 노드들. 여기 없는 노드나 파라미터를 선언하지 않은
# 노드는 그냥 무시된다(서비스 호출이 실패해도 다음 노드로 넘어간다).
VIZ_NODES = (
    "/state_machine",
    "/recovery_spliner_node",
    "/start_planner",
    "/static_avoidance_planner",
    "/planner_change",
    "/controller_manager",
    "/tracking",
)

# 주행에 전혀 관여하지 않는 순수 디버그/텔레메트리 노드. 프로세스째 종료한다.
# 패턴은 pgrep -f 정규식이며, 대괄호를 씌워 이 명령을 실행하는 셸 자신이
# 매치되지 않게 한다(그렇지 않으면 pkill 이 자기 부모를 죽인다).
DEBUG_PROCS = (
    ("lap_analyser", "[l]ap_analyser"),
    ("keyboard_joy", "[k]eyboard_joy_node"),
    ("rqt_reconfigure", "[r]qt_reconfigure"),
)


class RaceSupervisor(Node):
    def __init__(self):
        super().__init__("race_supervisor")

        self.declare_parameter("debug", True)
        self.declare_parameter("joy_topic", "/joy")
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value

        # latch: RViz/랩탑 헬퍼가 나중에 붙어도 마지막 상태를 받을 수 있어야 한다.
        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.debug_pub = self.create_publisher(Bool, "/debug_enabled", latched)
        self.debug_pub.publish(Bool(data=self.debug))

        self._fired = False
        if self.debug:
            # debug:=on 이면 감시할 것이 없다. 구독조차 만들지 않는다.
            self.get_logger().info("[race_supervisor] debug:=on — 디버그 출력 유지, 감시 안 함")
            return

        joy_topic = self.get_parameter("joy_topic").get_parameter_value().string_value
        self.create_subscription(Joy, joy_topic, self._joy_cb, 10)
        self.get_logger().info(
            f"[race_supervisor] debug:=off — {joy_topic} 의 RB(buttons[{RB_BUTTON}]) 대기 중. "
            "지금은 RViz 로 pose estimate 가능. RB 를 누르면 디버그를 끊는다.")

    def _joy_cb(self, msg: Joy):
        if self._fired or len(msg.buttons) <= RB_BUTTON:
            return
        if not msg.buttons[RB_BUTTON]:
            return
        self._fired = True
        self.get_logger().warn("[race_supervisor] RB 감지 — 디버그 출력을 끊는다 (되돌리지 않음)")
        self._silence_viz()
        self._kill_debug_procs()
        self.debug_pub.publish(Bool(data=False))
        self.get_logger().warn("[race_supervisor] 완료. 이후 주행 토픽만 발행된다.")

    def _silence_viz(self):
        """대상 노드들의 viz_rate_hz 를 0 으로. 응답을 기다리지 않는다 —
        여기서 블로킹하면 RB 를 누른 순간 이 노드의 실행기가 멈춰버린다."""
        for node_name in VIZ_NODES:
            try:
                cli = self.create_client(SetParameters, f"{node_name}/set_parameters")
                if not cli.service_is_ready():
                    # 없는 노드(예: ot_planner 선택에 따라 안 뜬 플래너)는 조용히 넘어간다.
                    if not cli.wait_for_service(timeout_sec=0.2):
                        continue
                req = SetParameters.Request()
                req.parameters = [
                    Parameter("viz_rate_hz", Parameter.Type.DOUBLE, 0.0).to_parameter_msg()]
                cli.call_async(req)
            except Exception as e:                       # 감독자가 죽으면 주행이 위험하다
                self.get_logger().error(f"[race_supervisor] {node_name} viz 정지 실패: {e}")

    def _kill_debug_procs(self):
        for label, pattern in DEBUG_PROCS:
            try:
                # check=False: 매치되는 프로세스가 없으면 pkill 이 1 을 반환하는데,
                # 그건 "이미 안 떠 있다"는 뜻이라 정상이다.
                subprocess.run(["pkill", "-f", pattern], check=False, timeout=2.0)
                self.get_logger().info(f"[race_supervisor] 종료 요청: {label}")
            except Exception as e:
                self.get_logger().error(f"[race_supervisor] {label} 종료 실패: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RaceSupervisor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGTERM(런치가 노드를 정리할 때 보낸다)이면 rclpy 의 시그널 핸들러가 이미
        # 컨텍스트를 내려놓은 상태라, 아래에서 shutdown 을 또 부르면
        # "rcl_shutdown already called" 로 트레이스백이 찍힌다. rclpy.ok() 로 확인.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
