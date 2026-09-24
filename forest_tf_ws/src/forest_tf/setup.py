from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'forest_tf'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hitech',
    maintainer_email='hitech@localhost',
    description='TF tools for the Forest quadruped dataset',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'forest_tf_node = forest_tf.forest_tf_node:main',
            'static_tf_republisher = forest_tf.static_tf_republisher:main',
        ],
    },
)