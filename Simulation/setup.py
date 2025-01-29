from setuptools import setup

setup(
   name='Simulation',
   version='1.2',
   description='Simulates Noise-Driven Adaptation and Evolution',
   author='Güney Erin Tekin',
   author_email='guney.erin@gmail.com',
   packages=['Simulation'],  #same as name
   install_requires=['numpy', 'numba', 'matplotlib','tqdm','scipy'], #external packages as dependencies
)