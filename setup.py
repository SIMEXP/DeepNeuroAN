from setuptools import setup

setup(name='DeepNeuroAN',
      version='1.0',
      description='Deep registration for f-MRI preprocessing',
      url='https://github.com/SIMEXP/DeepNeuroAN',
      author='Loic TETREL',
      author_email='loic.tetrel.pro@gmail.com',
      license='MIT',
      packages=['deepneuroan'],
      scripts=['bin/deepneuroan'],
      install_requires=[
          'tensorflow',
          'nilearn',
      ],
      include_package_data=True,
      zip_safe=False)
