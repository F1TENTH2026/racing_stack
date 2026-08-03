from setuptools import setup
import os
from glob import glob

package_name = 'steering_lookup'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'cfg'),
            glob('cfg/*.csv') + glob('cfg/*.npy')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jonathan Becker',
    maintainer_email='jonbecke@ethz.ch',
    description='The steering_lookup library',
    license='MIT',
    entry_points={
        'console_scripts': [
        ],
    },
)
