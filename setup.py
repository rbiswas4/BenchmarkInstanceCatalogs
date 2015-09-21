from distutils.core import setup

setup(# package information
      name="benchmarkInstanceCatalogs",
      version="0.0.1dev",
      description='Utilities to benchmark queries to InstanceCatalogs',
      long_description=''' ''',
      # What code to include as packages
      packages=['benchmarkInstanceCatalogs'],
      package_dir={'benchmarkInstanceCatalogs':'benchmarkInstanceCatalogs'},
      # What data to include as packages
      include_package_data=True,
      package_data={'benchmarkInstanceCatalogs': ['example_data/*.dat']}
      )
