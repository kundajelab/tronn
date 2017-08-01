from setuptools import setup
from setuptools import find_packages

if __name__== "__main__":
    setup(name="tronn",
          version='0.1.0',
          description="neural net tools for gene regulation models",
          author="Daniel Kim",
          author_email="danielskim@stanford.edu",
          url="https://github.com/kundajelab/tronn",
          license="MIT",
          install_requires=["numpy", "tensorflow-gpu", "six"],
          packages=find_packages(),
          package_data={"tronn":"data/*.json"},
          scripts=['bin/tronn',
                   "R/plot_metrics_curves.R",
                   'R/run_region_clustering.R',
                   'R/make_network_grammar.R',
                   'R/make_network_grammar_v2.R',
                   'R/filter_w_rna.R',
                   'R/run_rgreat.R']
    )
