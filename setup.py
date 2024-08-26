'''
Created on 25 November 2022
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
# from distutils.core import setup

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='AstronomyCalc',
      version='1.0.0',
      author='Sambit Giri',
      author_email='sambit.giri@gmail.com',
      packages=find_packages("src"),
      package_dir={"": "src"},
      package_data={'AstronomyCalc': ['input_data/*', 'input_data/**/*']},
      install_requires=requirements,
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
)
