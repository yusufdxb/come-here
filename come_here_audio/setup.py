from setuptools import find_packages, setup

package_name = 'come_here_audio'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/audio_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/audio.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yusuf Guenena',
    maintainer_email='yusuf.a.guenena@gmail.com',
    description='Audio perception for come-here system.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'audio_node = come_here_audio.audio_node:main',
        ],
    },
)
