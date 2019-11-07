import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='coltrane',
    version='0.0.2',
    author='Piotr Rarus',
    author_email='piotr.rarus@gmail.com',
    description='Just another ML framework. Built on top of scikit-learn.',
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
        'matplotlib',
        'seaborn',
        'tqdm',
        'lazy',
        'colorama',
        'austen'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)
