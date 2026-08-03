from setuptools import find_packages, setup

package_name = 'spliner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nicolas',
    maintainer_email='nicolas@todo.todo',
    description='Local spline-based overtaking planner (ported from ROS1 UNICORN spliner)',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'spliner_node = spliner.spliner_node:main',
            'static_avoidance_node = spliner.static_avoidance_node:main',
            'start_spline_node = spliner.start_spline_node:main',
            'start_spline_node_v2 = spliner.start_spline_node_v2:main',
        ],
    },
)
