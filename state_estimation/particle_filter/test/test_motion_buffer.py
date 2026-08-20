from collections import deque

from particle_filter.particle_filter import ParticleFiler


def test_time_weighted_zero_order_hold_integral():
    samples = deque([(0.0, 1.0), (0.5, 3.0), (0.75, 2.0)])
    # 1*0.5 + 3*0.25 + 2*0.25
    assert ParticleFiler._integrate_samples(samples, 0.0, 1.0, 99.0) == 1.75


def test_nonpositive_interval_has_no_motion():
    samples = deque([(1.0, 4.0)])
    assert ParticleFiler._integrate_samples(samples, 2.0, 1.0, 4.0) == 0.0
