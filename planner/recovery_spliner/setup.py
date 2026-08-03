from setuptools import find_packages, setup

package_name = 'recovery_spliner'

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
    maintainer='jeongsangryu',
    maintainer_email='ryujs@unist.ac.kr',
    description='Recovery spline planner for the UNICORN f1tenth racing stack',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'recovery_spliner_node = recovery_spliner.recovery_spliner_node:main',
        ],
    },
)
