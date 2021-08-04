import glob
from setuptools import setup
from setuptools import find_packages

if __name__== "__main__":
    setup(
        name="tronn",
        version='0.5.0',
        description="neural net tools for gene regulation models",
        author="Daniel Kim",
        author_email="danielskim@stanford.edu",
        url="https://github.com/kundajelab/tronn",
        license="MIT",
        install_requires=["numpy", "tensorflow-gpu", "six", "networkx==2.2", "seaborn==0.9.0"],
        packages=find_packages(),
        scripts=['bin/tronn'] + glob.glob("R/*.R") + glob.glob("R/ggr/*R") + glob.glob("scripts/*py") + glob.glob("scripts/ggr/*py")
    )
