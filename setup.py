from setuptools import setup, find_packages

setup(
    name='resolvent4py',
    version='0.1.0',
    packages=find_packages(include=['resolvent4py', 'resolvent4py.*']),
    install_requires=[
        'numpy', 
        'scipy',
        'petsc4py',
        'slepc4py'
    ],
    description='A package for numerical linear algebra and optimization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Alberto Padovan',
    author_email='padovan3@illinois.edu',
    url='https://github.com/albertopadovan/resolvent4py',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

