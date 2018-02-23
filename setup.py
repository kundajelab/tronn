from setuptools import setup
from setuptools import find_packages

if __name__== "__main__":
    setup(
        name="tronn",
        version='0.2.0',
        description="neural net tools for gene regulation models",
        author="Daniel Kim",
        author_email="danielskim@stanford.edu",
        url="https://github.com/kundajelab/tronn",
        license="MIT",
        install_requires=["numpy", "tensorflow-gpu", "six"],
        packages=find_packages(),
        package_data={"tronn":"data/*.json"},
        scripts=[
            'bin/tronn',
            "R/plot.pwm_x_position.R",
            "R/plot.example_x_pwm.R",
            "R/plot_metrics_curves.R"]
    )
