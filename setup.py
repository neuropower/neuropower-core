from setuptools import setup

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

setup(name='neuropower',
    version='0.1',
    description='A package to perform power analyses for neuroimaging data',
    long_description=readme(),
    url='http://github.com/neuropower/neuropower-core',
    author='Joke Durnez',
    author_email = 'joke.durnez@gmail.com',
    license='MIT',
    packages=['neuropower'],
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
