from setuptools import setup

setup(
   name='pyREM',
   version='0.1.0',
   author='Ronniy C. Joseph',
   author_email='ronniy.joseph@icrar.org',
   packages=['pyrem'],
   license='LICENSE',
   description='a python library containing all sorts of tools to understand EoR Experiments',
   long_description=open('README.md').read(),
   install_requires=[
      "numpy >= 1.15.1",
      "scipy >= 1.2",
      "matplotlib >= 3.0.0",
      "powerbox >= 0.5.0",
      "numba >= 0.40.0"
   ],
)