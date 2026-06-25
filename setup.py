from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

package_name = 'main_controller'


def _repo_root_from_setup() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_integrated_repo(repo_root: Path) -> None:
    required = (
        Path('FT300S'),
        Path('XenseTacSensor'),
        Path('RealSense') / 'launch',
    )
    missing = [str(path) for path in required if not (repo_root / path).exists()]
    if missing:
        joined = ', '.join(missing)
        raise RuntimeError(
            'MainController must be built inside the integrated repo; '
            f'missing {joined} under {repo_root}'
        )


class build_py(_build_py):
    """Build command that records the integrated repository root."""

    def run(self):
        repo_root = _repo_root_from_setup()
        _validate_integrated_repo(repo_root)
        super().run()
        package_dir = Path(self.build_lib) / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        hint_path = package_dir / '_repo_root_hint.py'
        hint_path.write_text(
            '"""Build-time repository root hint for MainController."""\n\n'
            f'REPO_ROOT_HINT = {str(repo_root)!r}\n',
            encoding='utf-8',
        )


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'pyzmq', 'matplotlib'],
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
    cmdclass={'build_py': build_py},
    entry_points={
        'console_scripts': [
            'main_controller = main_controller.main:main',
        ],
    },
)
