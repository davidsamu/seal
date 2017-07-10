from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='seal',
      version='0.1',
      license='GNU GPLv3',
      description='Spiking and Spectral Electrophysiology data Analysis Library.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)'
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering'
      ],
      keywords='neurophysiology electrophysiology spike spectral analysis',
      url='http://github.com/davidsamu/seal',

      author='David Samu',
      author_email='daavid.samu@gmail.com',

      packages=['seal', 'seal/analysis', 'seal/object', 'seal/test', 'seal/util'],
      install_requires=[
        'numpy',       # core
        'scipy',
        'pandas',      # data analysis
        'sklearn',
        'matplotlib',  # plotting
        'seaborn',
        'quantities',  # neurophy
        'neo',
        'elephant',
      ],

      test_suite='nose.collector',
      tests_require=['nose'],

      scripts=['bin/quality_test'],

      include_package_data=True,
      zip_safe=False
)
