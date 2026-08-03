from setuptools import setup
import os
from glob import glob

package_name = 'id_controller'

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
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jonathan Becker',
    maintainer_email='JonathanBecker.Tech@gmail.com',
    description='The controller for sysid experiments',
    license='MIT',
    entry_points={
        'console_scripts': [
            'controller_node = id_controller.controller_node:main',
        ],
    },
)
