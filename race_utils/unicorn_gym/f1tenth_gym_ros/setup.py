from setuptools import setup
import os
from glob import glob

package_name = 'f1tenth_gym_ros'


def data_files_in(pattern):
    # glob but skip directories (e.g. launch/__pycache__): setuptools can't
    # copy a directory as a data file ("doesn't exist or not a regular file").
    return [p for p in glob(pattern) if os.path.isfile(p)]


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), data_files_in('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.xacro')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Billy Zheng',
    maintainer_email='billyzheng.bz@gmail.com',
    description='Bridge for using f1tenth_gym in ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'gym_bridge = f1tenth_gym_ros.gym_bridge:main',
            # (opponent_controller moved to the standalone `opponent` package)
            # absorbed from the former standalone opponent_publisher package
            'obstacle_publisher = f1tenth_gym_ros.obstacle_publisher:main',
            'collision_detector = f1tenth_gym_ros.collision_detector:main'
        ],
    },
)
