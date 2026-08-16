from typing import Union
import numpy as np
from scipy.interpolate import CubicSpline

class FrenetConverter:
    def __init__(self, waypoints_x: np.array, waypoints_y: np.array, waypoints_psi: np.array = None):
        # Coerced once here so the hot paths can index and broadcast without
        # re-converting; callers pass lists as well as arrays.
        self.waypoints_x = np.asarray(waypoints_x, dtype=float)
        self.waypoints_y = np.asarray(waypoints_y, dtype=float)
        self.waypoints_psi = waypoints_psi
        self.waypoints_s = None
        self.spline_x = None
        self.spline_y = None
        self.raceline_length = None
        self.waypoints_distance_m = 0.1 # [m]
        self.iter_max = 3
        self.closest_index = None

        self.build_raceline()

    def build_raceline(self):
        # Cumulative arc length, vectorised. This used to be a Python loop calling
        # np.linalg.norm on a 2-element list per waypoint -- ~1000 interpreter
        # round-trips per rebuild, and frenet_odom_republisher rebuilds the converter
        # every time /global_waypoints arrives (0.5 Hz), not just at startup.
        # sqrt(dx*dx + dy*dy) rather than np.hypot: hypot's stabilised algorithm
        # differs from np.linalg.norm in the last ulp, and that 1e-14 on the spline
        # knots is visible after cubic extrapolation. This form is bit-identical to
        # the norm-in-a-loop it replaces, and np.cumsum matches the sequential
        # accumulation bit for bit, so the raceline is unchanged exactly.
        dx = np.diff(self.waypoints_x)
        dy = np.diff(self.waypoints_y)
        self.waypoints_s = np.concatenate(([0.0], np.cumsum(np.sqrt(dx * dx + dy * dy))))
        self.spline_x = CubicSpline(self.waypoints_s, self.waypoints_x)
        self.spline_y = CubicSpline(self.waypoints_s, self.waypoints_y)
        # One spline over both coordinates. Evaluating it returns (n, 2) in a single
        # scipy call, halving the per-evaluation Python overhead on the projection
        # path, which evaluates position and derivative 4 times per get_frenet().
        # spline_x / spline_y are kept because they are part of this class's surface.
        self._spline_xy = CubicSpline(
            self.waypoints_s, np.column_stack((self.waypoints_x, self.waypoints_y)))
        self.raceline_length = self.waypoints_s[-1]
        # Actual mean waypoint spacing. MUST match the raceline or the approx-s
        # seed (index*spacing) lands on the wrong part of the track and the
        # projection diverges (was hardcoded 0.1 -> broke any non-0.1 m line).
        if len(self.waypoints_x) > 1:
            self.waypoints_distance_m = self.raceline_length / (len(self.waypoints_x) - 1)

    def _nearest_idx(self, x, y) -> np.ndarray:
        """Index of the nearest raceline waypoint for each query point.

        The shared kernel behind get_closest_index() and get_approx_s(), which were
        two copies of this same argmin. Three things the old copies did that this
        does not:

          * np.tile materialised a full len(waypoints) x len(x) copy of the raceline
            on every call; broadcasting needs no copy at all.
          * np.linalg.norm([dist_x.T, dist_y.T], axis=0) built a Python list of two
            arrays, which numpy then had to stack into a third array before touching
            any arithmetic.
          * that norm took a square root of every element. argmin over distance and
            over squared distance pick the same index, so the sqrt was computed and
            discarded.

        Returns one index per query point, shaped like the old argmin(..., axis=1).
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        dx = self.waypoints_x[:, None] - x[None, :]
        dy = self.waypoints_y[:, None] - y[None, :]
        return np.argmin(dx * dx + dy * dy, axis=0)

    def get_frenet(self, x, y, s=None) -> np.array:
        # Compute Frenet coordinates for a given (x, y) point.
        # One nearest-waypoint search, not two: get_closest_index() and
        # get_approx_s() computed the identical argmin over the whole raceline, and
        # this called both. The index is what get_closest_index() returned; the
        # arc length at that index is what get_approx_s() returned.
        idx = self._nearest_idx(x, y)
        self.closest_index = idx  # kept updated for get_frenet_velocities()
        if s is None:
            # TRUE cumulative arc length at the nearest waypoint (not index*spacing,
            # which would assume uniform 0.1 m spacing).
            s = self.waypoints_s[idx]
        s, d = self.get_frenet_coord(x, y, s)

        return np.array([s, d])

    def get_approx_s(self, x, y) -> float:
        """
        Finds the s-coordinate of the given point by finding the nearest waypoint.
        """
        # Use the TRUE cumulative arc-length at the nearest waypoint (not
        # index*spacing, which assumes uniform 0.1 m spacing).
        return self.waypoints_s[self._nearest_idx(x, y)]

    def get_frenet_velocities(self, vx, vy, theta) -> np.array:
        """
        Returns the Frenet velocities for the given Cartesian velocities.

        Args:
            vx (float): x-velocity
            vy (float): y-velocity
            theta (float): orientation of the vehicle

        Returns:
            np.array: [s_dot, d_dot] Frenet velocities
        """
        if self.closest_index is None:
            raise ValueError("FRENET CONVERTER: closest index is None, call get_closest_index first.")
        if self.waypoints_psi is None:
            raise ValueError("FRENET CONVERTER: waypoints_psi is None, construct with psi to use frenet velocities.")
        delta_psi = theta - self.waypoints_psi[self.closest_index]
        s_dot = vx * np.cos(delta_psi) - vy * np.sin(delta_psi)
        d_dot = vx * np.sin(delta_psi) + vy * np.cos(delta_psi)

        return np.array([s_dot, d_dot])

    def get_closest_index(self, x, y) -> int:
        """
        Finds the index of the closest waypoint to the given point.

        Args:
            x (float): x-coordinate of the point
            y (float): y-coordinate of the point

        Returns:
            int: index of the closest waypoint
        """
        self.closest_index = self._nearest_idx(x, y)
        return self.closest_index


    def get_frenet_coord(self, x, y, s, eps_m=0.01) -> float:
        """
        Finds the s-coordinate of the given point, considering the perpendicular
        projection of the point on the track.

        Args:
            x (float): x-coordinate of the point
            y (float): y-coordinate of the point
            s (float): estimated s-coordinate of the point
            eps_m (float): maximum error tolerance for the projection. Default is 0.01.

        Returns:
            The s-coordinate of the point on the track.
        """
        # Check if point is on the estimated s perpendicular to the track

        _, projection, d = self.check_perpendicular(x, y, s, eps_m)
        for i in range(self.iter_max):
            cand_s = (s + projection)%self.raceline_length
            _, cand_projection, cand_d = self.check_perpendicular(x, y, cand_s, eps_m)
            #print(f"candidate projection: {cand_projection}; projection: {projection}; d: {d} cand_d: {cand_d}")
            cand_projection = np.clip(cand_projection, -self.waypoints_distance_m/(2*self.iter_max), self.waypoints_distance_m/(2*self.iter_max))
            updated_idxs = np.abs(cand_projection) <= np.abs(projection)
            d[updated_idxs] = cand_d[updated_idxs]
            s[updated_idxs] = cand_s[updated_idxs]
            projection[updated_idxs] = cand_projection[updated_idxs]

        return s, d

    def check_perpendicular(self, x, y, s, eps_m=0.01) -> Union[bool, float]:
        if np.any(np.isnan(s)):
            raise ValueError("BUB FRENET CONVERTER: S is nan")

        # Two scipy calls instead of four: the combined spline returns x and y
        # together. The derivative is still taken at s % raceline_length while the
        # position is taken at raw s -- that asymmetry is what get_derivative() and
        # the direct spline_x(s)/spline_y(s) calls did before, and changing it would
        # move the projection for any s outside [0, raceline_length).
        s = np.asarray(s, dtype=float)
        pos = self._spline_xy(s)                                   # (n, 2)
        der = self._spline_xy(s % self.raceline_length, 1)         # (n, 2)

        # Unit tangent, then the parallel and perpendicular components of the
        # track-to-point vector. Written out rather than assembled into (2, n)
        # arrays for einsum: same arithmetic, without building four intermediates
        # per call on a path that runs four times per get_frenet().
        # Divide rather than multiply by a reciprocal: the two differ in the last
        # ulp, and dividing is what the np.linalg.norm form here did, so this stays
        # bit-identical to it.
        tx, ty = der[:, 0], der[:, 1]
        norm = np.sqrt(tx * tx + ty * ty)
        tx = tx / norm
        ty = ty / norm

        # obtain vector from the track to the point
        x_vec = np.asarray(x, dtype=float).reshape(-1) - pos[:, 0]
        y_vec = np.asarray(y, dtype=float).reshape(-1) - pos[:, 1]

        # projection of point_to_track on the tangent, and on its left normal
        proj = tx * x_vec + ty * y_vec
        d = -ty * x_vec + tx * y_vec

        # TODO commented out because of computational efficiency
        # eps_m * point_to_track_norm is needed to make it scale invariant
        # check_perpendicular becomes effectively cos(angle) <= eps_m

        # point_to_track_norm = np.linalg.norm(point_to_track, axis=0)
        # check_perpendicular = np.abs(proj) <= eps_m * point_to_track_norm
        check_perpendicular = None

        return check_perpendicular, proj, d

    def get_derivative(self, s) -> np.array:
        """
        Returns the derivative of the point corresponding to s on the chosen line.

        Args:
            s: parameter which is used to evaluate the spline
            line: argument used to choose the line. Can be 'int', 'mid', 'out'. Default is 'mid'.

        Returns:
            der: dx/ds, dy/ds
        """
        s = s%self.raceline_length

        der = [self.spline_x(s, 1), self.spline_y(s, 1)]

        return der


    def get_cartesian(self, s: float, d: float) -> np.array:
        """
        Convert Frenet coordinates to Cartesian coordinates

        Args:
            s (float): longitudinal coordinate
            d (float): lateral coordinate

        Returns:
            np.array: [x, y] Cartesian coordinates
        """
        x = self.spline_x(s)
        y = self.spline_y(s)
        psi = self.get_derivative(s)
        psi = np.arctan2(psi[1], psi[0])
        x += d * np.cos(psi + np.pi / 2)
        y += d * np.sin(psi + np.pi / 2)

        return np.array([x, y])


    def get_e_psi(self, x: float, y:float, yaw:float) -> float:
        """
        Calculate E_psi: the heading error between vehicle yaw and track direction.

        Args:
            x (float): vehicle x position
            y (float): vehicle y position
            yaw (float): vehicle yaw angle (radians)

        Returns:
            float: heading error E_psi (in radians, between -pi and pi)
        """
        s = self.get_approx_s(np.array([x]), np.array([y]))[0]
        psi = np.arctan2(*self.get_derivative(s)[::-1])  # dy/ds, dx/ds → arctan2(dy, dx)

        e_psi = yaw - psi
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

        return e_psi
