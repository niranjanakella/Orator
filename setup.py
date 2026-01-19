"""Setup configuration for nj-orator package"""

from setuptools import setup, find_packages
import os

# Read version from package
def get_version():
    """Get version from nj_orator package"""
    try:
        with open(os.path.join('nj_orator', '__init__.py'), 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return '1.0.0'

# Read long description from README
def get_long_description():
    """Get long description from README"""
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return "Orator TTS - Text-to-speech with custom hotkeys"

# Read requirements
def get_requirements():
    """Get requirements from requirements.txt"""
    try:
        with open('requirements.txt', 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception:
        return []

setup(
    name='nj-orator',
    version=get_version(),
    description='Orator TTS - Text-to-speech with custom hotkeys',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Niranjan Akella',
    url='https://github.com/niranjanakella/Orator',
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
    include_package_data=True,
    package_data={
        'nj_orator': [
            'orator_menu_icon.png',
        ],
    },
    install_requires=get_requirements() + ['click>=8.0.0'],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'orator=nj_orator.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: MacOS',
    ],
)

