from setuptools import setup
from setuptools import find_packages

if __name__== '__main__':
    setup(name='tronn',
          version='0.0.1',
          description='neural net tools for gene regulation models',
          author='Daniel Kim',
          author_email='danielskim@stanford.edu',
          url='https://github.com/kundajelab/tronn',
          license='MIT',
          install_requires=['numpy', 'tensorflow-gpu', 'six'],
          packages=find_packages())
