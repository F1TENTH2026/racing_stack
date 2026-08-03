#!/usr/bin/env python3
# node for sysid experiments. Drives the car through predefined excitation
# profiles so vehicle dynamics can be identified from the recorded rosbags.

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64


class IDController(Node):

    def __init__(self):
        super().__init__('id_controller')

        # publishers
        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped, 'drive_topic', 10)
        # create publishers to vesc electric-RPM (speed) and servo commands
        # (bridge ackermann to vesc node and directly talk to vesc driver)
        self.erpm_pub = self.create_publisher(
            Float64, '/vesc/commands/motor/speed', 10)
        self.servo_pub = self.create_publisher(
            Float64, '/vesc/commands/servo/position', 10)
        self.current_pub = self.create_publisher(
            Float64, '/vesc/commands/motor/current', 10)
        self.brake_pub = self.create_publisher(
            Float64, '/vesc/commands/motor/brake', 10)

        # parameters
        self.declare_parameter('experiment', 5)

        # 1 - send const vesc command
        self.declare_parameter('const_erpm', 3000.0)
        self.declare_parameter('const_servo', 0.415)

        # 2 - accelerate then decelerate via current commands
        self.declare_parameter('const_curr', 40.0)
        self.declare_parameter('const_brake', 50.0)
        self.declare_parameter('accel_time', 1.5)
        self.declare_parameter('decel_time', 1.5)

        # 3 - accelerate then decelerate via acceleration commands
        self.declare_parameter('const_accel', 5.0)
        self.declare_parameter('const_decel', -5.0)

        # 4 - drive with const erpm and increase servo position
        self.declare_parameter('angle_time', 20.0)
        self.declare_parameter('start_pos', -0.1)
        self.declare_parameter('end_pos', 0.4)

        # 5 - drive with const motor speed and increase steering angle (uses
        # angle_time from above)
        self.declare_parameter('start_angle', 0.1)
        self.declare_parameter('end_angle', 0.4)
        self.declare_parameter('const_speed', 3.0)

        # 6 - follow given acceleration profile and record IMU data
        self.declare_parameter('acc_profile', 3)

        # 7 - bang bang control on the servo with constant speed
        self.declare_parameter('period', 3.0)
        self.declare_parameter('repetitions', 5)
        self.declare_parameter('bangbang_steer', 0.1)

        self.experiment = self.get_parameter('experiment').value

        self.const_erpm = self.get_parameter('const_erpm').value
        self.const_servo = self.get_parameter('const_servo').value

        self.const_curr = self.get_parameter('const_curr').value
        self.const_brake = self.get_parameter('const_brake').value
        self.accel_time = self.get_parameter('accel_time').value
        self.decel_time = self.get_parameter('decel_time').value

        self.const_accel = self.get_parameter('const_accel').value
        self.const_decel = self.get_parameter('const_decel').value

        self.angle_time = self.get_parameter('angle_time').value
        self.start_pos = self.get_parameter('start_pos').value
        self.end_pos = self.get_parameter('end_pos').value

        self.start_angle = self.get_parameter('start_angle').value
        self.end_angle = self.get_parameter('end_angle').value
        self.const_speed = self.get_parameter('const_speed').value

        self.acc_profile = self.get_parameter('acc_profile').value

        self.period = self.get_parameter('period').value
        self.repetitions = self.get_parameter('repetitions').value
        self.bangbang_steer = self.get_parameter('bangbang_steer').value

        # flags for "log once" behaviour (rospy.loginfo_once replacement)
        self._logged_once = set()

        self.get_logger().warn("Starting experiment #" + str(self.experiment))

        self.start_time = self.get_time()
        # 30 Hz control loop (rospy.Rate(30) replacement)
        self.timer = self.create_timer(1.0 / 30.0, self.loop)

    def get_time(self):
        # seconds as float, mirrors rospy.get_time()
        return self.get_clock().now().nanoseconds * 1e-9

    def loginfo_once(self, msg):
        if msg not in self._logged_once:
            self._logged_once.add(msg)
            self.get_logger().info(msg)

    def loop(self):
        if self.experiment == 1:
            self.send_const_vesc_cmd()
        elif self.experiment == 2:
            self.accel_decel()
        elif self.experiment == 3:
            self.drive_accel_decel()
        elif self.experiment == 4:
            self.increase_servo_position()
        elif self.experiment == 5:
            self.increase_steering_angle()
        elif self.experiment == 6:
            self.acc_profile_recorder()
        elif self.experiment == 7:
            self.bang_bang_servo()

    def send_const_vesc_cmd(self):
        erpm_msg = Float64(data=float(self.const_erpm))
        servo_msg = Float64(data=float(self.const_servo))
        self.erpm_pub.publish(erpm_msg)
        self.servo_pub.publish(servo_msg)

    def accel_decel(self):
        servo_msg = Float64(data=float(self.const_servo))
        self.servo_pub.publish(servo_msg)
        if (self.get_time() - self.start_time < self.accel_time):
            current_msg = Float64(data=float(self.const_curr))
            self.current_pub.publish(current_msg)
            self.get_logger().info("accelerating")
        elif (self.get_time() - self.start_time < self.accel_time + self.decel_time):
            brake_msg = Float64(data=float(self.const_brake))
            self.brake_pub.publish(brake_msg)
            self.get_logger().info("decelerating")

    def drive_accel_decel(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = 0.0
        drive_msg.drive.steering_angle_velocity = 0.0
        drive_msg.drive.speed = 0.0
        # used as a flag to indicate the ackermann controller to use
        # acceleration instead of speed
        drive_msg.drive.jerk = 512.0

        if (self.get_time() - self.start_time < 3):  # wait for 3s
            self.loginfo_once("experiment starting")
        elif (self.get_time() - self.start_time < self.accel_time + 3):
            drive_msg.drive.acceleration = float(self.const_accel)
            self.get_logger().info("accelerating")
            self.cmd_pub.publish(drive_msg)
        elif (self.get_time() - self.start_time < self.accel_time + self.decel_time + 3):
            drive_msg.drive.acceleration = float(self.const_decel)
            self.cmd_pub.publish(drive_msg)
            self.get_logger().info("decelerating")
        else:
            self.loginfo_once("experiment over")

    def increase_servo_position(self):
        time_frac = (self.get_time() - self.start_time) / self.angle_time
        if time_frac <= 1:
            angle = self.end_pos * time_frac + self.start_pos * (1 - time_frac)
            servo_msg = Float64(data=float(angle))
            erpm_msg = Float64(data=float(self.const_erpm))
            self.erpm_pub.publish(erpm_msg)
            self.servo_pub.publish(servo_msg)
        else:
            self.loginfo_once("experiment over")

    def increase_steering_angle(self):
        time_frac = (self.get_time() - self.start_time) / self.angle_time
        if time_frac <= 1:
            angle = self.end_angle * time_frac + self.start_angle * (1 - time_frac)
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.steering_angle = float(angle)
            drive_msg.drive.steering_angle_velocity = 0.0
            drive_msg.drive.speed = float(self.const_speed)
            drive_msg.drive.acceleration = 0.0
            drive_msg.drive.jerk = 0.0
            self.cmd_pub.publish(drive_msg)
            if time_frac >= 0.9:
                self.get_logger().warn("Ending soon, stop the bag")
        else:
            self.loginfo_once("experiment over")

    def acc_profile_recorder(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = 0.0
        drive_msg.drive.steering_angle_velocity = 0.0

        if self.acc_profile == 1:  # Step response to speed
            speed = 3.0
            duration = 4
            if (self.get_time() - self.start_time < 2):  # stand still 2s
                drive_msg.drive.speed = 0.0
                drive_msg.drive.jerk = 0.0
                drive_msg.drive.acceleration = 0.0
                self.cmd_pub.publish(drive_msg)
                self.loginfo_once("experiment starting")
            elif (self.get_time() - self.start_time > duration + 2):  # stand still again
                drive_msg.drive.speed = 0.0
                drive_msg.drive.jerk = 0.0
                drive_msg.drive.acceleration = 0.0
                self.cmd_pub.publish(drive_msg)
                self.loginfo_once("experiment over")
            else:  # step input to 3 m/s after 2 seconds have passed
                drive_msg.drive.speed = speed
                drive_msg.drive.jerk = 0.0
                drive_msg.drive.acceleration = 0.0
                self.cmd_pub.publish(drive_msg)
                self.get_logger().info("accelerating")
        elif self.acc_profile == 2:  # Max acc and braking test
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.steering_angle_velocity = 0.0
            drive_msg.drive.speed = 0.0
            drive_msg.drive.jerk = 512.0
            if (self.get_time() - self.start_time < self.accel_time):
                drive_msg.drive.acceleration = 100.0
                self.get_logger().info("accelerating")
                self.cmd_pub.publish(drive_msg)
            elif (self.get_time() - self.start_time < self.accel_time + self.decel_time):
                drive_msg.drive.acceleration = -100.0
                self.cmd_pub.publish(drive_msg)
                self.get_logger().info("decelerating")
            else:
                self.loginfo_once("experiment over")
        elif self.acc_profile == 3:  # speed steps at speed
            speed1 = 3.0
            speed2 = 5.0
            duration = 6
            if (self.get_time() - self.start_time < 2 or
                    self.get_time() - self.start_time > 2 + duration):  # stand still
                drive_msg.drive.speed = 0.0
                drive_msg.drive.jerk = 0.0
                drive_msg.drive.acceleration = 0.0
                self.cmd_pub.publish(drive_msg)
                self.loginfo_once("standing still")
            elif (self.get_time() - self.start_time > duration / 3 + 2 and
                    self.get_time() - self.start_time < duration / 3 * 2 + 2):  # accelerate to speed 2
                drive_msg.drive.speed = speed2
                drive_msg.drive.jerk = 0.0
                drive_msg.drive.acceleration = 0.0
                self.cmd_pub.publish(drive_msg)
                self.get_logger().info("accelerating further")
            elif (self.get_time() - self.start_time > duration / 3 * 2 + 2 and
                    self.get_time() - self.start_time < duration + 2):  # decelerate to speed 1 again
                drive_msg.drive.speed = speed2
                drive_msg.drive.jerk = 0.0
                drive_msg.drive.acceleration = 0.0
                self.cmd_pub.publish(drive_msg)
                self.get_logger().info("decelerating")
            elif (self.get_time() - self.start_time > duration + 2):  # stand still again
                drive_msg.drive.speed = 0.0
                drive_msg.drive.jerk = 0.0
                drive_msg.drive.acceleration = 0.0
                self.cmd_pub.publish(drive_msg)
                self.loginfo_once("experiment over")
            else:  # step input to speed 1 after 2 seconds have passed
                drive_msg.drive.speed = speed1
                drive_msg.drive.jerk = 0.0
                drive_msg.drive.acceleration = 0.0
                self.cmd_pub.publish(drive_msg)
                self.get_logger().info("accelerating")
        else:
            self.get_logger().info("invalid speed profile")
            drive_msg.drive.speed = 1.0
            drive_msg.drive.jerk = 512.0
            if (self.get_time() - self.start_time < self.accel_time):
                drive_msg.drive.acceleration = float(self.const_accel)
                self.get_logger().info("accelerating")
                self.cmd_pub.publish(drive_msg)
            elif (self.get_time() - self.start_time < self.accel_time + self.decel_time):
                drive_msg.drive.acceleration = float(self.const_decel)
                self.cmd_pub.publish(drive_msg)
                self.get_logger().info("decelerating")
            else:
                self.loginfo_once("experiment over")

    def bang_bang_servo(self):
        time_frac = (self.get_time() - self.start_time) / self.period

        steer_sign = -1
        if time_frac % 1 < 0.5:
            steer_sign = 1

        if time_frac <= 1 * self.repetitions:
            angle = steer_sign * self.bangbang_steer

            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.steering_angle = float(angle)
            drive_msg.drive.steering_angle_velocity = 0.0
            drive_msg.drive.speed = float(self.const_speed)
            drive_msg.drive.acceleration = 0.0
            drive_msg.drive.jerk = 0.0
            self.cmd_pub.publish(drive_msg)
        else:
            self.loginfo_once("experiment over")


def main(args=None):
    rclpy.init(args=args)
    node = IDController()
    node.get_logger().info('id Controller is running')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
