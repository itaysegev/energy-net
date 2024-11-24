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
        'numpy>=1.24.0',
        'pandas',
        'stable-baselines3==2.0.0',
        'sb3-contrib==2.0.0',
        'sphinx-autodoc-typehints==1.23.0',
        'box2d-py==2.3.8',
        'pybullet',
        'optuna~=3.0',
        'pytablewriter~=0.64',
        'pyyaml>=5.1',
        'cloudpickle>=1.5.0',
        'plotly',
        'rliable==1.2.0',
        'wandb',
        'huggingface_sb3>=2.2.5',
        'seaborn',
        'tqdm',
        'rich',
        'moviepy',
        'ruff',
        'inflection',
        'ipykernel',
        'ipywidgets',
        'jupyter',  
        'tensorboard',
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