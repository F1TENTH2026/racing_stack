import os
from glob import glob
from setuptools import setup

package_name = 'virtual_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='js',
    maintainer_email='kwonc@unist.ac.kr',
    description='Virtual objects (opponents + static obstacles) injected into the '
                'perception stack (LiDAR overlay + tracking merge) for sim and real',
    license='MIT',
    entry_points={
        'console_scripts': [
            'opponent_vehicle = virtual_perception.opponent_vehicle:main',
            'opponent_controller = virtual_perception.opponent_controller:main',
            'static_obstacle_manager = virtual_perception.static_obstacle_manager:main',
            'scan_overlay = virtual_perception.scan_overlay:main',
            'tracking_merger = virtual_perception.tracking_merger:main',
        ],
    },
)
