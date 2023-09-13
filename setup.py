import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chemotools",
    version_config=True,
    setup_requires=["setuptools-git-versioning"],
    author="Pau Cabaneros Lopez",
    author_email="pau.cabaneros@gmail.com",
    description="Package to integrate chemometrics in scikit-learn pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paucablop/chemotools",
    project_urls={
        "Bug Tracker": "https://github.com/paucablop/chemotools/issues/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
    ],
    include_package_data=True,
    package_data={'': ['tests/resources/*.csv',
                       'datasets/data/*.csv']}
)