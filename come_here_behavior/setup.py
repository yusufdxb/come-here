from setuptools import find_packages, setup

package_name = 'come_here_behavior'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/behavior_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yusuf Guenena',
    maintainer_email='yusuf.a.guenena@gmail.com',
    description='Behavior state machine for come-here system.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'behavior_node = come_here_behavior.behavior_node:main',
            'go2_bridge_node = come_here_behavior.go2_bridge_node:main',
        ],
    },
)
