from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pdigy",
    version="0.1.0",
    author="pdigy",
    description="A pathology digital image compression and storage format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pdigy/pdigy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "Pillow>=8.0.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "openslide-python>=1.1.2",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "matplotlib",
        ],
    },
    include_package_data=True,
    license="MIT",
)
