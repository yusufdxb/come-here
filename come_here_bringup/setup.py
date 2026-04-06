from setuptools import find_packages, setup

package_name = 'come_here_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/come_here.launch.py']),
        ('share/' + package_name + '/config', ['config/come_here_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yusuf Guenena',
    maintainer_email='yusuf.a.guenena@gmail.com',
    description='Launch and config for come-here system.',
    license='MIT',
)
