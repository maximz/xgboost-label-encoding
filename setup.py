#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.md") as history_file:
    history = history_file.read()

requirements = ["scikit-learn", "xgboost", "numpy", "typing_extensions"]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Maxim Zaslavsky",
    author_email="maxim@maximz.com",
    name="xgboost_label_encoding",
    description="Xgboost Label Encoding",
    packages=find_packages(
        include=["xgboost_label_encoding", "xgboost_label_encoding.*"]
    ),
    python_requires=">=3.8",
    version="0.0.2",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/maximz/xgboost-label-encoding",
    zip_safe=False,
)
