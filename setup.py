import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='coltrane',
    version='0.1.0',
    author='Piotr Rarus',
    author_email='piotr.rarus@gmail.com',
    description='ML framework to ease research process. Built on top of scikit-learn.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/piotr-rarus/coltrane',
    packages=setuptools.find_packages(
        exclude=[
            "tests",
        ]
    ),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'scikit-multilearn',
        'imbalanced-learn',
        'matplotlib'
        'seaborn',
        'joblib',
        'tqdm',
        'pytest',
        'pytest_cov',
        'lazy_property',
        'colorama',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)
