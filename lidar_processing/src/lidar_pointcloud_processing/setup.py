from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'lidar_pointcloud_processing'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'requirements.txt']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Configurable LiDAR PointCloud2 preprocessing for ROS 2 Humble',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'pointcloud_processing_node = '
            'lidar_pointcloud_processing.pointcloud_node:main',
            'pointcloud_inspector = '
            'lidar_pointcloud_processing.pointcloud_inspector:main',
            'inspect_bag = lidar_pointcloud_processing.tools.inspect_bag:main',
            'offline_experiments = '
            'lidar_pointcloud_processing.tools.offline_experiments:main',
            'visualize_frame = lidar_pointcloud_processing.tools.visualize_frame:main',
        ],
    },
)
