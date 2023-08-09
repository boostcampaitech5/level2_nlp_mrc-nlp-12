from setuptools import setup, find_packages


setup(
    name='odqa', 
    version='0.1',
    description='Python package',
    packages=find_packages(where='code'),
    package_dir={'': 'code'},
)
