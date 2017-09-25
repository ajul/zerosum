import setuptools
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name = 'zerosum',
    version = '0.0.0',
    description = 'Balances zero-sum games so that they have a specified Nash equilibrium.',
    long_description = long_description,
    url = 'https://github.com/ajul/zerosum',

    author = 'Albert Julius Liu',
    author_email = 'ajul1987@gmail.com',

    license = 'BSD-3-Clause',

    classifiers = [
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',

        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

    keywords = 'game-theory game-design numerical-optimization',

    packages = setuptools.find_packages(exclude=['examples', 'tests']),

    install_requires = ['numpy', 'scipy'],
    extras_require = {
        'excel': ['xlwings', 'pypiwin32'],
    },
)