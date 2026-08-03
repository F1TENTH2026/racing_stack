from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lap_analyser'

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
    maintainer='HMCL',
    maintainer_email='shineejoon@unist.ac.kr',
    description='Analyses lap times, lateral error and distance to track boundary for F1TENTH racing',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lap_analyser = lap_analyser.lap_analyser:main',
        ],
    },
)
