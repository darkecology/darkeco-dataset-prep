from setuptools import find_packages, setup

setup(
    name='cajundata',
    version="0.0.1",
    description='Cajun data set scripts',
    packages=find_packages(),
    url='https://github.com/darkecology/cajundata',
    author='Dan Sheldon',
    author_email='sheldon@cs.umass.edu',
    install_requires=[
        'pandas>=1.0.0'
        'pvlib>=0.9.0',
    ],
    keywords='radar aeroecology bird migration',
    license='MIT'
)
