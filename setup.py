# setup.py

from setuptools import setup, find_packages

setup(
    name='energy-net-env',
    version='0.1.0',
    description='A custom Gymnasium environment for energy network simulation.',
    author='Itay Segev',
    author_email='itaysegev11@gmail.com',
    packages=find_packages(),
    install_requires=[
        'gymnasium>=0.26.0',
        'numpy',
        'pandas',
        'pyyaml',  # If using YAML for configurations
        # Add other dependencies as needed
    ],
    entry_points={
        'gymnasium.envs': [
            'EnergyNetEnv-v0 = energy_net_env.energy_net_env:EnergyNetEnv',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update as per your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify your Python version
)