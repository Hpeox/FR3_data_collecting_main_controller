from setuptools import find_packages, setup

package_name = 'main_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'pyzmq'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='hpx.peipei@gmail.com',
    description='Main controller for multi-sensor imitation-learning data capture',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'main_controller = main_controller.main:main',
        ],
    },
)
