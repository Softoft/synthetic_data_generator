from setuptools import setup, find_packages

setup(
    name="synthetic_data_generator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of the synthetic data generator package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
