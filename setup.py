from setuptools import setup, find_packages

setup(
    name="icua_analysis",
    version="0.1.0",
    description="Data analysis code for the first icua experiment",
    author="Dr. Benedict Wilkins",
    author_email="",
    url="https://github.com/dicelab-rhul/icua-experiment-1-analysis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
    ],
    python_requires=">=3.7",
)
