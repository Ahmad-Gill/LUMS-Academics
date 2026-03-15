from setuptools import setup
from glob import glob

package_name = 'robotics_lab'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahmad',
    maintainer_email='ahmad@example.com',
    description='Robotics lab exercises',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'publisher = robotics_lab.publisher:main',
            'moveStraight = robotics_lab.moveStraight:main',
            'moveToGoal = robotics_lab.moveToGoal:main',
            'moveSquare = robotics_lab.moveSquare:main',
            'moveCircle = robotics_lab.moveCircle:main',
'chase = robotics_lab.chase:main',
'move_square = robotics_lab.moveSquare:main',  # Add this
            'move_circle = robotics_lab.moveCircle:main',  # Add this
        ],
    },
)
