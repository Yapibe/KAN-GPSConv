from setuptools import setup, find_packages

setup(
    name="KAN-GPSConv",
    version="0.1",
    author="Yuval Hanuka, Yair Pickholz Berliner",
    author_email="yuvalhanukamail.tau.ac.il, Pickholz.yair@gmail.com",
    description="A project exploring KAN layers integration into GPS networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch-geometric",
        "numpy",
        "scikit-learn",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
