from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sector_tuner'

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
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HMCL',
    maintainer_email='shineejoon@unist.ac.kr',
    description='Sector velocity scaler and sector slicing GUI for F1TENTH racing',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sector_tuner = sector_tuner.sector_tuner:main',
            'sector_slicer = sector_tuner.sector_slicer:main',
        ],
    },
    scripts=["scripts/finish_sector.sh"],
)
