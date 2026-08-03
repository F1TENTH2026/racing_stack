import rclpy
from rclpy.node import Node

import numpy as np

from f110_msgs.msg import WpntArray
from frenet_conversion.frenet_converter import FrenetConverter
from frenet_conversion_msgs.srv import (
    Frenet2Glob,
    Frenet2GlobArr,
    Glob2Frenet,
    Glob2FrenetArr,
)


class FrenetConversionServer(Node):
    """
    ROS2 service server that converts coordinates between the global cartesian
    frame and the frenet frame defined by ``/global_waypoints``.

    Port of the ROS1 C++ ``frenet_conversion_server`` node. The ``PerceptionOnly``
    parameter selects between the default service names and the perception
    variants, mirroring the original behaviour.
    """

    def __init__(self):
        super().__init__('frenet_conversion_server')

        self.declare_parameter('PerceptionOnly', False)
        self.perception_only = self.get_parameter('PerceptionOnly').get_parameter_value().bool_value

        self.converter = None
        self.has_global_trajectory = False

        self.init_subscribers_publishers()

        self.get_logger().info('[Frenet Conversion] Waiting for global waypoints...')
        self.get_logger().info('[Frenet Conversion] Frenet Conversion Server ready.')

    def init_subscribers_publishers(self):
        self.global_trajectory_sub_ = self.create_subscription(
            WpntArray,
            '/global_waypoints',
            self.global_trajectory_callback,
            10)

        self.get_logger().info('[Frenet Conversion] PERCEPTION ONLY: %d' % int(self.perception_only))

        if not self.perception_only:
            g2f = 'convert_glob2frenet_service'
            g2fa = 'convert_glob2frenetarr_service'
            f2g = 'convert_frenet2glob_service'
            f2ga = 'convert_frenet2globarr_service'
        else:
            g2f = 'convert_glob2frenet_perception_service'
            g2fa = 'convert_glob2frenetarr_perception_service'
            f2g = 'convert_frenet2glob_perception_service'
            f2ga = 'convert_frenet2globarr_perception_service'

        self.convert_glob2frenet_server_ = self.create_service(
            Glob2Frenet, g2f, self.glob2frenet_conversion_callback)
        self.convert_glob2frenetarr_server_ = self.create_service(
            Glob2FrenetArr, g2fa, self.glob2frenetarr_conversion_callback)
        self.convert_frenet2glob_server_ = self.create_service(
            Frenet2Glob, f2g, self.frenet2glob_conversion_callback)
        self.convert_frenet2globarr_server_ = self.create_service(
            Frenet2GlobArr, f2ga, self.frenet2globarr_conversion_callback)

    def global_trajectory_callback(self, msg):
        waypoint_array = msg.wpnts
        waypoints_x = [waypoint.x_m for waypoint in waypoint_array]
        waypoints_y = [waypoint.y_m for waypoint in waypoint_array]
        waypoints_psi = [waypoint.psi_rad for waypoint in waypoint_array]
        self.converter = FrenetConverter(
            np.array(waypoints_x), np.array(waypoints_y), np.array(waypoints_psi))
        if not self.has_global_trajectory:
            self.get_logger().info('Global waypoints received.')
        self.has_global_trajectory = True

    def glob2frenet_conversion_callback(self, request, response):
        if self.converter is None:
            self.get_logger().warn('[Frenet Conversion] No global trajectory yet.')
            return response
        frenet = self.converter.get_frenet([request.x], [request.y])
        idx = self.converter.get_closest_index([request.x], [request.y])
        response.s = float(frenet[0, 0])
        response.d = float(frenet[1, 0])
        response.idx = int(idx[0])
        return response

    def glob2frenetarr_conversion_callback(self, request, response):
        if self.converter is None:
            self.get_logger().warn('[Frenet Conversion] No global trajectory yet.')
            return response
        x = list(request.x)
        y = list(request.y)
        frenet = self.converter.get_frenet(x, y)
        idx = self.converter.get_closest_index(x, y)
        response.s = [float(s) for s in frenet[0]]
        response.d = [float(d) for d in frenet[1]]
        response.idx = [int(i) for i in idx]
        return response

    def frenet2glob_conversion_callback(self, request, response):
        if self.converter is None:
            self.get_logger().warn('[Frenet Conversion] No global trajectory yet.')
            return response
        glob = self.converter.get_cartesian(request.s, request.d)
        response.x = float(glob[0])
        response.y = float(glob[1])
        return response

    def frenet2globarr_conversion_callback(self, request, response):
        if self.converter is None:
            self.get_logger().warn('[Frenet Conversion] No global trajectory yet.')
            return response
        glob = self.converter.get_cartesian(np.array(request.s), np.array(request.d))
        response.x = [float(x) for x in np.atleast_1d(glob[0])]
        response.y = [float(y) for y in np.atleast_1d(glob[1])]
        return response


def main(args=None):
    rclpy.init(args=args)

    frenet_conversion_server = FrenetConversionServer()
    rclpy.spin(frenet_conversion_server)

    frenet_conversion_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
