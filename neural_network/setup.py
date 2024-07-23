import setuptools

from setup_utils import get_requirements, get_dev_requirements

requirements = get_requirements()
dev_requirements = get_dev_requirements()

setuptools.setup(
    name="neural_network",
    version="0.1",
    description="Custom neural network package (Experimental package)",
    python_requires=">=3.7",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    install_requires=[requirements],
    extras_require={"dev": [dev_requirements]},
)
