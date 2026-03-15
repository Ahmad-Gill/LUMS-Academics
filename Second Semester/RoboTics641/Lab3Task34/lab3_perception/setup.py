from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'lab3_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # ✅ THIS LINE installs launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahmadgill',
    maintainer_email='your_email@example.com',
    description='Lab3 perception package',
    license='Apache License 2.0',
    tests_require=['pytest'],
entry_points={
    'console_scripts': [
        'turtlebot3_mapping = lab3_perception.turtlebot3_mapping:main',
        'dynamic_tf_broadcaster = lab3_perception.dynamic_tf_broadcaster:main',
        'turtlebot3_circle = lab3_perception.turtlebot3_circle:main',
        'turtlebot3_go2goal = lab3_perception.turtlebot3_go2goal:main',
        'turtlebot3_laserscan = lab3_perception.turtlebot3_laserscan:main',
    ],
},
)