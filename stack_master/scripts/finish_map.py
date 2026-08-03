#!/usr/bin/env python3
"""
finish_map: one-click end-to-end Cartographer map finalization.

Triggered by RViz "2D Goal Pose" (/goal_pose), it:
  0. queries the active trajectory id from Cartographer (no longer hardcoded 0)
  1. finishes that trajectory
  2. writes the .pbstream (auto-versioned so existing maps are never overwritten)
  3. renders an occupancy grid to PNG + YAML via nav2_map_server/map_saver_cli
  4. rebuilds the workspace so the new map is immediately available

All Cartographer service calls use rclpy async clients + add_done_callback so the
chain runs entirely inside the single rclpy.spin() loop (no re-entrant spinning).
"""

import rclpy
from rclpy.node import Node
from cartographer_ros_msgs.srv import (
    FinishTrajectory, WriteState, GetTrajectoryStates)
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import subprocess
import sys
import os
import zlib
import struct


class FinishMapNode(Node):
    def __init__(self):
        super().__init__('finish_map_node')

        # Declare and get parameters
        self.declare_parameter('map_name', 'my_map')
        self.declare_parameter('maps_dir', '')
        # Must match cartographer_occupancy_grid_node's -resolution (launch: 0.05)
        self.declare_parameter('resolution', 0.05)

        self.map_name = self.get_parameter('map_name').value
        self.maps_dir = self.get_parameter('maps_dir').value
        self.resolution = self.get_parameter('resolution').value

        if not self.maps_dir:
            self.get_logger().error('maps_dir parameter is required!')
            sys.exit(1)

        # Create maps directory if it doesn't exist
        os.makedirs(self.maps_dir, exist_ok=True)

        self.get_logger().info(f'Map name: {self.map_name}')
        self.get_logger().info(f'Maps directory: {self.maps_dir}')

        # Create service clients for Cartographer
        self.finish_trajectory_client = self.create_client(
            FinishTrajectory, '/finish_trajectory')
        self.write_state_client = self.create_client(
            WriteState, '/write_state')
        self.trajectory_states_client = self.create_client(
            GetTrajectoryStates, '/get_trajectory_states')

        # Wait for services to be available
        self.get_logger().info('Waiting for Cartographer services...')
        if not self.finish_trajectory_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/finish_trajectory service not available')
        if not self.write_state_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/write_state service not available')
        if not self.trajectory_states_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/get_trajectory_states service not available '
                                   '(will fall back to trajectory id 0)')

        # Resolved at save time: the actual on-disk basename used for the
        # pbstream / png / yaml (may be auto-versioned, e.g. "<map>_v2").
        self.map_basename = self.map_name

        # Subscribe to /goal_pose topic from RViz
        # Use RELIABLE QoS to prevent message loss over network
        goal_pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        self.goal_pose_subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            goal_pose_qos)

        # Flag to trigger map saving
        self.should_save = False

        # Timer to check if we should save
        self.timer = self.create_timer(0.1, self.check_and_save)

        self.get_logger().warning("\n" + "="*60)
        self.get_logger().warning("Click '2D Goal Pose' in RViz to finish mapping and save the map")
        self.get_logger().warning("="*60 + "\n")

    def goal_pose_callback(self, msg: PoseStamped):
        """Callback when goal pose is received from RViz"""
        self.get_logger().info('Goal pose received from RViz. Triggering map save...')
        self.should_save = True

    def check_and_save(self):
        """Check if we should save the map"""
        if self.should_save:
            self.timer.cancel()
            self.finish_and_save_map()

    def finish_and_save_map(self):
        """Finish trajectory and save map"""
        self.get_logger().info('Starting map save process...')

        # Step 1: Query the active trajectory id, then finish it. Done async so
        # we never re-enter rclpy.spin() from within a callback.
        if self.trajectory_states_client.service_is_ready():
            self.get_logger().info('Step 1/3: Querying active trajectory id...')
            states_future = self.trajectory_states_client.call_async(
                GetTrajectoryStates.Request())
            states_future.add_done_callback(self.trajectory_states_callback)
        else:
            self.get_logger().warn(
                '/get_trajectory_states not ready, falling back to trajectory id 0')
            self.finish_trajectory(0)

    def trajectory_states_callback(self, future):
        """Pick the most recent trajectory id from the states response."""
        trajectory_id = 0
        try:
            response = future.result()
            ids = response.trajectory_states.trajectory_id
            if len(ids):
                trajectory_id = ids[-1]
        except Exception as e:
            self.get_logger().error(
                f'get_trajectory_states failed ({e}); falling back to id 0')
        self.finish_trajectory(trajectory_id)

    def finish_trajectory(self, trajectory_id):
        """Finish the given trajectory id."""
        self.get_logger().info(f'Step 1/3: Finishing trajectory {trajectory_id}...')
        finish_req = FinishTrajectory.Request()
        finish_req.trajectory_id = int(trajectory_id)

        finish_future = self.finish_trajectory_client.call_async(finish_req)
        finish_future.add_done_callback(self.finish_trajectory_callback)

    def finish_trajectory_callback(self, future):
        """Callback when finish trajectory is done"""
        try:
            result = future.result()
            if result is not None:
                self.get_logger().info('Trajectory finished successfully')
                self.save_pbstream()
            else:
                self.get_logger().error('Failed to finish trajectory')
        except KeyboardInterrupt:
            raise  # Re-raise KeyboardInterrupt to shutdown properly
        except Exception as e:
            self.get_logger().error(f'Error finishing trajectory: {str(e)}')

    def save_pbstream(self):
        """Save the pbstream file (auto-versioned to avoid overwriting maps)."""
        # Pick a basename that doesn't collide: <map>, then <map>_v2, _v3, ...
        basename = self.map_name
        pbstream_path = os.path.join(self.maps_dir, f'{basename}.pbstream')
        version = 2
        while os.path.exists(pbstream_path):
            basename = f'{self.map_name}_v{version}'
            pbstream_path = os.path.join(self.maps_dir, f'{basename}.pbstream')
            version += 1
        self.map_basename = basename
        if basename != self.map_name:
            self.get_logger().warn(
                f'{self.map_name}.pbstream exists; saving as {basename}.pbstream instead')

        self.get_logger().info(f'Step 2/3: Saving pbstream to {pbstream_path}...')

        write_req = WriteState.Request()
        write_req.filename = pbstream_path
        write_req.include_unfinished_submaps = True

        write_future = self.write_state_client.call_async(write_req)
        write_future.add_done_callback(self.write_state_callback)

    def write_state_callback(self, future):
        """Callback when write state is done"""
        try:
            result = future.result()
            if result is not None:
                self.get_logger().info('Pbstream saved successfully')
                self.save_occupancy_grid()
            else:
                self.get_logger().error('Failed to save pbstream')
        except KeyboardInterrupt:
            raise  # Re-raise KeyboardInterrupt to shutdown properly
        except Exception as e:
            self.get_logger().error(f'Error saving pbstream: {str(e)}')

    def save_occupancy_grid(self):
        """Render the occupancy grid (PNG + YAML) from the saved pbstream.

        We render from the pbstream rather than snapshotting the live /map_carto
        topic: right after FinishTrajectory the occupancy_grid_node may be
        mid-rebuild and publish only a partial grid, which produced truncated /
        all-unknown maps. The pbstream always holds the full optimized map.

        cartographer_pbstream_to_ros_map only emits PGM, so we convert that to
        PNG (the format the rest of the stack's maps use) and drop the PGM.
        """
        map_path = os.path.join(self.maps_dir, self.map_basename)
        pbstream_path = os.path.join(self.maps_dir, f'{self.map_basename}.pbstream')
        pgm_path = f'{map_path}.pgm'
        png_path = f'{map_path}.png'

        self.get_logger().info(
            f'Step 3/3: Rendering occupancy grid from pbstream to '
            f'{png_path} / {map_path}.yaml...')

        try:
            # cartographer_pbstream_to_ros_map writes <map_filestem>.pgm + .yaml
            cmd = [
                'ros2', 'run', 'cartographer_ros',
                'cartographer_pbstream_to_ros_map',
                '-pbstream_filename', pbstream_path,
                '-map_filestem', map_path,
                '-resolution', str(self.resolution),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0 and os.path.exists(pgm_path):
                # Convert the rendered PGM -> PNG and discard the PGM.
                self._pgm_to_png(pgm_path, png_path)
                os.remove(pgm_path)
                # The tool writes an absolute path into the yaml's `image:` field;
                # rewrite it to the relative PNG basename so the map stays portable
                # after colcon copies it into install/.
                self._fix_yaml_image_path(f'{map_path}.yaml',
                                          f'{self.map_basename}.png')
                self.get_logger().info(f'Map image saved to {png_path}')
                self.get_logger().info(f'Map YAML saved to {map_path}.yaml')
            else:
                self.get_logger().error(
                    f'Failed to render map from pbstream: '
                    f'{result.stderr or result.stdout}')
        except subprocess.TimeoutExpired:
            self.get_logger().error('pbstream_to_ros_map timed out')
        except Exception as e:
            self.get_logger().error(f'Error rendering map image: {str(e)}')

        self.get_logger().info("\n" + "="*60)
        self.get_logger().info(f"Map saving complete!")
        self.get_logger().info(f"  - {pbstream_path}")
        self.get_logger().info(f"  - {png_path}")
        self.get_logger().info(f"  - {map_path}.yaml")
        self.get_logger().info("="*60 + "\n")

        self.rebuild_workspace()

    @staticmethod
    def _pgm_to_png(pgm_path, png_path):
        """Convert an 8-bit PGM to grayscale PNG using only the stdlib.

        The image libraries (Pillow/OpenCV) are unreliable in the conda runtime
        this node launches under, so we read the PGM (cartographer emits binary
        P5; ascii P2 handled too) and hand-encode a non-interlaced 8-bit
        grayscale PNG with zlib — no third-party deps.
        """
        with open(pgm_path, 'rb') as f:
            data = f.read()

        idx = 0

        def next_token():
            nonlocal idx
            # skip whitespace and '#' comment lines between header fields
            while idx < len(data):
                c = data[idx:idx + 1]
                if c.isspace():
                    idx += 1
                elif c == b'#':
                    while idx < len(data) and data[idx:idx + 1] not in (b'\n', b'\r'):
                        idx += 1
                else:
                    break
            start = idx
            while idx < len(data) and not data[idx:idx + 1].isspace():
                idx += 1
            return data[start:idx]

        magic = next_token()
        width = int(next_token())
        height = int(next_token())
        maxval = int(next_token())
        idx += 1  # single whitespace separator precedes the pixel data

        if magic == b'P5':
            if maxval > 255:
                raise ValueError('16-bit PGM not supported')
            pixels = data[idx:idx + width * height]
            if len(pixels) < width * height:
                raise ValueError('truncated PGM pixel data')
        elif magic == b'P2':
            pixels = bytes(int(v) for v in data[idx:].split()[:width * height])
        else:
            raise ValueError(f'unsupported PGM magic {magic!r}')

        def chunk(tag, payload):
            return (struct.pack('>I', len(payload)) + tag + payload +
                    struct.pack('>I', zlib.crc32(tag + payload) & 0xffffffff))

        raw = bytearray()
        for y in range(height):
            raw.append(0)  # filter type 0 (None) per scanline
            raw.extend(pixels[y * width:(y + 1) * width])

        with open(png_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)))
            f.write(chunk(b'IDAT', zlib.compress(bytes(raw), 9)))
            f.write(chunk(b'IEND', b''))

    def _fix_yaml_image_path(self, yaml_path, image_basename):
        """Replace the yaml `image:` value with a relative basename."""
        try:
            with open(yaml_path, 'r') as f:
                lines = f.readlines()
            with open(yaml_path, 'w') as f:
                for line in lines:
                    if line.startswith('image:'):
                        f.write(f'image: {image_basename}\n')
                    else:
                        f.write(line)
        except Exception as e:
            self.get_logger().warn(f'Could not rewrite yaml image path: {e}')

    def rebuild_workspace(self):
        """Reinstall only stack_master so the new map is immediately available.

        The map lives in the stack_master package, so only that package needs to
        be (re)installed — no need to rebuild the whole workspace (much faster,
        and avoids unrelated packages aborting the build).
        """
        # maps_dir = .../ws/src/racing_stack/stack_master/maps/<map_name>
        # workspace root is 5 levels up
        ws_dir = os.path.abspath(os.path.join(self.maps_dir, '../../../../..'))
        self.get_logger().info(f'Rebuilding stack_master at: {ws_dir}')

        cmd = (
            'colcon build --symlink-install --packages-select stack_master'
            ' --cmake-args -DCMAKE_BUILD_TYPE=Release'
            ' && source install/setup.bash'
        )
        try:
            result = subprocess.run(
                ['bash', '-c', cmd],
                cwd=ws_dir,
                timeout=300
            )
            if result.returncode == 0:
                self.get_logger().info('Workspace rebuilt successfully. New map is now available.')
            else:
                self.get_logger().error(f'colcon build failed (exit code {result.returncode})')
        except subprocess.TimeoutExpired:
            self.get_logger().error('colcon build timed out (>5 min)')
        except Exception as e:
            self.get_logger().error(f'Error during colcon build: {str(e)}')

        # Shutdown the node
        self.get_logger().info('Shutting down finish_map node...')
        raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    node = FinishMapNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
