from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'geometric_pointcloud_analysis'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Ground segmentation, clustering, normals and bounding boxes for LiDAR clouds',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'geometric_analysis_node = '
            'geometric_pointcloud_analysis.geometric_analysis_node:main',
            'offline_geometry = geometric_pointcloud_analysis.tools.offline_geometry:main',
        ],
    },
)
