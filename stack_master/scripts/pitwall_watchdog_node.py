#!/usr/bin/env python3
"""pitwall_watchdog — 차가 디버그를 끊으면 랩탑의 RViz 를 같이 닫는 노드.

RViz(pitwall)는 조종수 랩탑에서 뜨므로 차에서 직접 죽일 수 없다. 대신 차의
race_supervisor 가 조이스틱 RB 를 받으면 /debug_enabled 에 false 를 latch 로
발행하고, 이 노드가 랩탑에서 그것을 받아 로컬의 RViz 를 닫는다.

pitwall.launch.xml 에 함께 들어가므로 평소처럼

    ros2 launch stack_master pitwall.launch.xml

만 하면 된다. 끄고 싶으면 watchdog:=false.

latch(TRANSIENT_LOCAL) 구독인 이유: 차가 먼저 RB 를 받은 뒤에 랩탑에서 pitwall 을
띄우는 순서도 있을 수 있다. 그때도 마지막 상태를 받아야 "이미 레이스 중"임을 알고
바로 닫는다.

기본적으로 자기 자신과 rqt 도 함께 정리하고 RViz 를 닫는다. 주행 중에 조종수
랩탑이 계속 마커를 구독하고 있으면 차의 노드들이 get_subscription_count() > 0 을
보고 마커를 계속 만들기 때문에, 랩탑을 닫는 것은 단순한 편의가 아니라 차의 CPU 를
실제로 덜어주는 일이다.
"""
import os
import signal
import subprocess

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool

# pgrep -f 정규식. 대괄호를 씌워 이 패턴을 담고 있는 자기 자신(및 부모 셸)이
# 매치되지 않게 한다 — 그렇지 않으면 워치독이 스스로를 먼저 죽인다.
TARGETS = (
    ("rviz2", "[r]viz2"),
    ("rqt_reconfigure", "[r]qt_reconfigure"),
)


class PitwallWatchdog(Node):
    def __init__(self):
        super().__init__("pitwall_watchdog")
        self.declare_parameter("shutdown_self", True)
        self.shutdown_self = self.get_parameter("shutdown_self").value

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Bool, "/debug_enabled", self._cb, latched)
        self._fired = False
        self.get_logger().info(
            "[pitwall_watchdog] /debug_enabled 감시 중. 차에서 RB 를 누르면 RViz 를 닫는다.")

    def _cb(self, msg: Bool):
        if msg.data:
            # debug 가 살아 있다는 뜻이므로 아무 것도 하지 않는다.
            return
        if self._fired:
            return
        self._fired = True
        self.get_logger().warn("[pitwall_watchdog] 차가 디버그를 끊었다 — RViz 를 닫는다")
        for label, pattern in TARGETS:
            try:
                subprocess.run(["pkill", "-f", pattern], check=False, timeout=2.0)
                self.get_logger().info(f"[pitwall_watchdog] 종료 요청: {label}")
            except Exception as e:
                self.get_logger().error(f"[pitwall_watchdog] {label} 종료 실패: {e}")
        if self.shutdown_self:
            # 자기가 속한 launch 프로세스 그룹째 정리해 터미널을 돌려준다.
            self.get_logger().info("[pitwall_watchdog] pitwall 런치를 종료한다")
            try:
                os.kill(os.getppid(), signal.SIGINT)
            except OSError:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = PitwallWatchdog()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGTERM 이면 rclpy 시그널 핸들러가 이미 컨텍스트를 내렸다. rclpy.ok() 확인.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
