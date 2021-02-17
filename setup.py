"""A setuptools based setup module.
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='pong_rl',
    version='0.0.1',
    description='A sample Python project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/min0ru/pong_rl',
    author='min0ru',
    author_email='2010min0ru@gmail.com',
    packages=find_packages(where='pong_rl'),
    python_requires='>=3.6, <4',

    install_requires=[
        'gym[atari]',
        'tensorflow==2.4.1',
        'numpy==1.19.2',
        'scipy==1.6.0'
    ],

    extras_require={
        'dev': [
            'ipython',
            'jedi<0.18',
            'jupyter',
            'jupyterlab',
        ],
    },
)
