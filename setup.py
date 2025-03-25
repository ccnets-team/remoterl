from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Core package dependencies (without simulator-specific ones)
install_requires = [
    "websocket-client",
    "boto3",
    "sagemaker",
    "msgpack",
    "requests",
    "gymnasium[mujoco]>=1.0.0",
    "ray[rllib]>=2.4.0",
    "torch",
    "typer",
    "pyyaml",
]

setup(
    name="remoterl",
    version="1.0.2",
    packages=find_packages(), 
    include_package_data=True,
    package_data={
        "remoterl.cli": ["*.yaml"],  # explicitly specify the cli subpackage
    },
    entry_points={
        "console_scripts": [
            "remoterl=remoterl.cli.cli:app",
        ],
    },
    install_requires=install_requires,
    author="JunHo Park",
    author_email="junho@ccnets.org",
    url="https://github.com/ccnets-team/remoterl",
    description="Remote RL for training and inference on AWS SageMaker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Dual Licensed (REMOTE RL COMMERCIAL LICENSE or GNU GPLv3)",
    keywords="remote rl reinforcement-learning sagemaker",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
