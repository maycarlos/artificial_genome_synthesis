from setuptools import find_packages, setup

setup(
    name="ags",
    packages=find_packages(),
    # package_dir = {"": "./artificial_genome_synthesis"},
    version="0.2.11",
    description="Networks for artifical genome synthesis",
    author="Elmer",
    entry_points={"console_scripts": ["ags = artificial_genome_synthesis.main:main"]},
    license="",
)
