from setuptools import setup, find_packages

setup(
    name='IHSetLim',
    version='1.1.4',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'xarray',
        'numba',
        'pandas',
        'datetime',
        'spotpy',
        'IHSetCalibration @ git+https://github.com/defreitasL/IHSetCalibration.git',
        'fast_optimization @ git+https://github.com/defreitasL/fast_optimization.git'
    ],
    author='Changbin Lim',
    author_email='limcs@unican.es',
    description='IH-SET Lim et al. (2022)',
    url='https://github.com/IHCantabria/IHSetLim',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)