import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy


class GridFilter:
    """Erosion-based occupancy lookup. ROS2 port: pass the owning rclpy `node`
    so it can subscribe to the map topic and log; or feed maps via map_callback()."""

    def __init__(self, node=None, map_topic=None, debug=False):
        self.node = node
        self.resolution = None      # m/pixel
        self.origin = None          # (x, y)
        self.map_data = None
        self.image = None           # OccupancyGrid -> OpenCV image
        self.eroded_image = None
        self.kernel_size = 3
        self.debug = debug
        self.map_topic = map_topic
        if self.node is not None and self.map_topic:
            self.subscribe_to_map(self.map_topic)
        else:
            self._log("No node/map topic provided; feed maps via map_callback().")

    def _log(self, msg):
        if self.node is not None:
            self.node.get_logger().info(str(msg))
        else:
            print(f"[GridFilter] {msg}")

    def subscribe_to_map(self, map_topic):
        self._log(f"Subscribing to map topic: {map_topic}")
        # nav2 map_server / gym_bridge latch /map (TRANSIENT_LOCAL); a default
        # VOLATILE subscriber never receives the one-shot latched map, so
        # eroded_image stays None and is_point_inside() is always False.
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self._sub = self.node.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, map_qos)

    def map_callback(self, msg):
        if self.image is None:
            self.resolution = msg.info.resolution
            self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            width, height = msg.info.width, msg.info.height
            image = np.array(msg.data, dtype=np.int8).reshape((height, width))
            # 255: obstacle, 0: free
            self.image = np.where(image == 100, 0, 255).astype(np.uint8)
            self.update_image()
            self._log("Map image initialized.")

    def set_erosion_kernel_size(self, size):
        self.kernel_size = size
        self.update_image()

    def update_image(self):
        if self.image is None:
            self._log("Map image not initialized.")
            return
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        self.eroded_image = cv2.erode(self.image, kernel)

    def world_to_pixel(self, x, y):
        px = int((x - self.origin[0]) / self.resolution)
        py = int((y - self.origin[1]) / self.resolution)
        return px, py

    def is_point_inside(self, x, y):
        if self.eroded_image is None:
            return False
        px, py = self.world_to_pixel(x, y)
        if px < 0 or py < 0 or px >= self.eroded_image.shape[1] or py >= self.eroded_image.shape[0]:
            return False
        return self.eroded_image[py, px] == 255
