from setuptools import setup

setup(
    name="calcli",
    version="6.9",
    packages=["calcli"],
    entry_points={
        "console_scripts": ["calcli = calcli.__main__:main"],
    },
    description="A CLI Calculator using argparse",
    author="Pranish Lama",
    python_requires=">=3.6",
)