from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

requires=[
    'numpy',
    'scipy',
    'nibabel',
    'pandas',
    'nose'
]

setup(
    # Application name:
    name='neuropower',

    # Version number:
    version='0.2.2',

    # Author details
    author='Joke Durnez',
    author_email = 'joke.durnez@gmail.com',

    # Packages
    packages=find_packages(),

    #Details
    description='A package to perform power analyses for neuroimaging data',
    long_description=readme(),
    url='http://github.com/neuropower/neuropower-core',
    license='MIT',
    keywords='statistics power fmri neuroimaging inference samplesize',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7'
    ],
    test_suite='nose.collector',
    tests_require=requires,
    install_requires=requires
    )
