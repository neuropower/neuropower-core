from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    # Application name:
    name='neuropower',

    # Version number:
    version='0.2.5',

    # Author details
    author='Joke Durnez',
    author_email = 'joke.durnez@gmail.com',
    maintainer = 'Joke Durnez',
    maintainer_email = 'joke.durnez@gmail.com',

    # Packages
    packages=find_packages(),

    #Details
    url = 'https://github.com/neuropower/neuropower-core',
    description='A package to perform power analyses for neuroimaging data',
    long_description=readme(),
    license='MIT',
    keywords='statistics power fmri neuroimaging inference samplesize',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7'
    ],
    test_suite='nose.collector',
    install_requires=[
        'numpy==1.11.0',
        'scipy==0.17.0',
        'nibabel==2.0.2',
        'pandas==0.18.1',
        'nose==1.3.7'
        ]
    )
