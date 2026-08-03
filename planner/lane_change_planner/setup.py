from setuptools import find_packages, setup

package_name = 'lane_change_planner'

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
    maintainer='jobalchi',
    maintainer_email='dltjdwns9871@naver.com',
    description='Lane-change avoidance planner (UNICORN f1tenth stack)',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'change_avoidance_node = lane_change_planner.change_avoidance_node:main',
            'update_waypoints = lane_change_planner.update_waypoints:main',
        ],
    },
)
