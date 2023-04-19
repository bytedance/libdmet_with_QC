from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'pyscf',
    'matplotlib',
    'numpy==1.21',
    'openfermion==1.3.0',
    'openfermionpyscf==0.5',
    'fqe==0.2.0',
    'projectq==0.7.3',
    'julia==0.5.7'
]


setup(name='libdmet_qc',
      version='0.1',
      description='libDMET with solvers of quantum computing chemistry',
      author='ByteDance',
      author_email='',
      packages=find_packages(),
      install_requires=REQUIRED_PACKAGES,
      platforms=['any'],
     )
