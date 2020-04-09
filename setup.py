import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='coltrane',
    version='0.1.1',
    author='Piotr Rarus',
    author_email='piotr.rarus@gmail.com',
    description='Just another ML framework. Built on top of scikit-learn.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/piotr-rarus/coltrane',
    packages=setuptools.find_packages(
        exclude=[
            "*test",
        ]
    ),
    install_requires=[
        'austen==0.2.7',
        'colorama==0.4.3',
        'joblib==0.14.1',
        'jupyter==1.0.0',
        'lazy==1.4',
        'numpy==1.18.2',
        'pandas==1.0.3',
        'plotly==4.6.0',
        'scikit-learn==0.22.2',
        'tqdm==4.45.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)
