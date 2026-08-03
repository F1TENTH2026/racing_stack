from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'frenet_conversion_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gnone',
    maintainer_email='eghignone@pbl.ee.ethz.ch',
    description='Service server converting between global and frenet coordinates',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'frenet_conversion_server_node = frenet_conversion_server.frenet_conversion_server_node:main'
        ],
    },
)
