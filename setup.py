from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "ReALOG - RelAtional Learning On Graphs."
LONG_DESCRIPTION = "ReALOG - A python package for training GNNs on relational data."

# Setting up
setup(
    name="realog",
    version=VERSION,
    author="Martin Jurkovic, Valter Hudovernik, Uros Kozole",
    author_email="martin.jurkovic19@gmail.com, valter.hudovernik@gmail.com, kozole.uros@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
    "networkx==3.3",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "scikit-learn==1.4.2",
    "sdv==1.12.1",
    "torch==2.3.0",
    "torch_geometric==2.5.3",
    "tensorboard==2.16.2",
    "tensorboardX==2.6.2",
        ],
    keywords=["python", "realog", "gnn", "graph neural networks", "relational data"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)