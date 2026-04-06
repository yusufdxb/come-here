from setuptools import find_packages, setup

package_name = 'come_here_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/perception_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yusuf Guenena',
    maintainer_email='yusuf.a.guenena@gmail.com',
    description='Visual person detection for come-here system.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'perception_node = come_here_perception.perception_node:main',
        ],
    },
)
